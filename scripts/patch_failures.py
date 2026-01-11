import asyncio
import json
import logging
import glob
import os
import aiohttp
from typing import Dict, List

# logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# prompt templates and persona setup
BUSINESS_PERSONA = "You are a business decision maker synthesizing loan recommendations."
SYSTEM_INSTRUCTIONS = "You are a business decision agent. Return ONLY valid JSON, no markdown, no other text."
PORT = 8005

class VLLMClientAsync:
    # helper class to talk to vllm server
    def __init__(self, base_url: str = f"http://localhost:{PORT}/v1"):
        self.base_url = base_url.rstrip('/')
        self.api_key = "EMPTY"

    async def generate(self, prompt: str, system_prompt: str = None, config: Dict = None) -> str:
        url = f"{self.base_url}/completions"
        
        # formatting for llama 3
        if system_prompt:
             full_prompt = (
                 f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                 f"{system_prompt}<|eot_id|>"
                 f"<|start_header_id|>user<|end_header_id|>\n\n"
                 f"{prompt}<|eot_id|>"
                 f"<|start_header_id|>assistant<|end_header_id|>\n\n"
             )
        else:
             full_prompt = (
                 f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                 f"{prompt}<|eot_id|>"
                 f"<|start_header_id|>assistant<|end_header_id|>\n\n"
             )

        payload = {
            "model": config.get('model', "meta-llama/Llama-3.2-3B-Instruct"),
            "prompt": full_prompt,
            "temperature": config.get('temperature', 0.1),
            "max_tokens": config.get('max_tokens', 1000),
            "stop": ["<|eot_id|>", "<|end_of_text|>", "```\n\n"]
        }
        
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"Error calling vLLM: {response.status} - {text}")
                        return ""
                    data = await response.json()
                    return data['choices'][0]['text']
            except Exception as e:
                logger.error(f"Exception calling vLLM: {e}")
                return ""

class PatchWorker:
    # worker class to handle the logic of fixing failed decisions
    def __init__(self, client: VLLMClientAsync):
        self.client = client

    def _safe_float(self, value, default=0.0):
        if value is None: return default
        try: return float(value)
        except (ValueError, TypeError): return default

    def _safe_int(self, value, default=0):
        if value is None: return default
        try: return int(float(value))
        except (ValueError, TypeError): return default
    
    def _format_business_prompt(self, application_data: str, agent_recommendations: list) -> str:
        # building the list of previous agent recommendations
        formatted_recs = []
        for i, rec in enumerate(agent_recommendations):
            if rec is None: rec = {}
            name = rec.get('agent_name', 'Unknown')
            decision = rec.get('approval_decision', 'N/A')
            app_type = rec.get('approval_type', 'N/A')
            rate = self._safe_float(rec.get('interest_rate'), 0.0)
            conf_prob = self._safe_int(rec.get('confidence_probability'), 0)
            conf_level = rec.get('confidence_level', 'N/A')
            reasoning = rec.get('reasoning', {}).get('approval_decision_reason', 'N/A')
            
            rec_text = f"Agent {i+1}: {name}\n- Decision: {decision}\n- Type: {app_type}\n- Interest Rate: {rate}%\n- Confidence: {conf_prob}% ({conf_level})\n- Reasoning: {reasoning}"
            formatted_recs.append(rec_text.strip())
        
        recommendations_text = "\n\n".join(formatted_recs)
        
        # main synthesis prompt logic
        prompt = f"""{SYSTEM_INSTRUCTIONS}

{BUSINESS_PERSONA}

LOAN APPLICATION:
{application_data}

AGENT RECOMMENDATIONS:
{recommendations_text}

TASK:
Synthesize a final business decision considering all agent perspectives.

Return ONLY valid JSON matching this structure (no markdown, no backticks):
{{
  "agent_name": "Business Decision",
  "loan_type": "personal_loan",
  "approval_decision": "approve or deny",
  "approval_type": "STANDARD_TERMS or SUBOPTIMAL_TERMS or MANUAL_REVIEW or DENIAL",
  "interest_rate": 8.5,
  "confidence_probability": 75,
  "confidence_level": "high",
  "agent_influence": {{
    "risk_manager_weight": 0.25,
    "regulatory_weight": 0.25,
    "data_science_weight": 0.25,
    "consumer_advocate_weight": 0.25,
    "primary_influence": "Name of most influential agent"
  }},
  "reasoning": {{
    "synthesis_rationale": "Why this decision",
    "weight_justification": "Why these weights",
    "risk_assessment": "Final risk view"
  }}
}}

CRITICAL: Return ONLY the JSON object. Weights must sum to 1.0.
"""
        return prompt

    async def patch_decision(self, record: Dict) -> Dict:
        # get application data and previous outputs
        initial_prompt = record.get('initial_prompt')
        all_agent_outputs = record.get('all_agent_outputs', [])
        
        prompt = self._format_business_prompt(initial_prompt, all_agent_outputs)
        
        # loop until we get a valid response (max 5 tries)
        for attempt in range(5):
            response_text = await self.client.generate(
                prompt=prompt,
                system_prompt=None, 
                config={"temperature": 0.2 + (attempt * 0.1), "max_tokens": 1500}
            )
            
            # clean up any markdown blocks if the model included them
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            try:
                decision_json = json.loads(cleaned_text)
                
                # final check for confidence value
                conf = decision_json.get("confidence_probability")
                if conf is None or conf == 0:
                    raise ValueError("Zero confidence generated")
                
                logger.info("Successfully synthesized valid business decision.")
                return decision_json
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
        
        logger.error("Failed to patch decision after 5 attempts.")
        return None

async def process_file(filepath: str, worker: PatchWorker):
    # scan a file for records that need patching
    logger.info(f"Scanning {os.path.basename(filepath)}...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return

    results = data.get("results", [])
    modified = False
    
    for idx, r in enumerate(results):
        needs_patch = False
        biz = r.get("business_decision", {})
        
        # logic for identifying bad records
        conf = biz.get("confidence_probability")
        if conf is None or not isinstance(conf, (int, float)) or conf <= 0:
            needs_patch = True
            logger.info(f"Record {idx}: Found Invalid Confidence ({conf})")
        
        reasoning = str(biz.get("reasoning", ""))
        if "System Error" in reasoning or "Parsing Failure" in reasoning:
            needs_patch = True
            logger.info(f"Record {idx}: Found Error String in reasoning")
            
        if needs_patch:
            # run the patcher
            logger.info(f"Patching record {idx}...")
            new_decision = await worker.patch_decision(r)
            if new_decision:
                r["business_decision"] = new_decision
                modified = True
            else:
                logger.error(f"Could not patch record {idx}. Keeping original error.")

    if modified:
        # save the file back if we changed anything
        logger.info(f"Saving patched file: {filepath}")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        logger.info(f"No changes needed for {filepath}")

async def main():
    # just scans all sequential files in the folder
    directory = "sequential_inference"
    pattern = os.path.join(directory, "sequential_*.json")
    files = sorted(glob.glob(pattern))
    
    client = VLLMClientAsync()
    worker = PatchWorker(client)
    
    for f in files:
        await process_file(f, worker)

if __name__ == "__main__":
    asyncio.run(main())

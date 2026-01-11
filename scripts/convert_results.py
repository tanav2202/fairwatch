
import asyncio
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
import aiohttp
import tqdm.asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BUSINESS_PERSONA = "You are a business decision maker synthesizing loan recommendations."
SYSTEM_INSTRUCTIONS = "You are a business decision agent. Return ONLY valid JSON, no markdown, no other text."

class VLLMClientAsync:
    """Minimal Async Client for vLLM (copied/adapted from vllm_client_async.py)"""
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.base_url = base_url.rstrip('/')
        self.api_key = "EMPTY"

    async def generate(self, prompt: str, system_prompt: str = None, config: Dict = None) -> str:
        url = f"{self.base_url}/completions"
        
        # PROMPT FORMATTING (Llama 3)
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
            "temperature": config.get('temperature', 0.1), # Low temp for synthesis
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

class LegacyConverter:
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
        """Exact logic from BusinessDecisionAgent"""
        formatted_recs = []
        # Note: In the sequential logic, we usually just list them in order of appearance
        # But BusinessDecisionAgent.py had a fixed list 'Risk Manager', 'Regulatory Compliance'...
        # Wait, the example output shows Agent 1, Agent 2, Agent 3... matching the execution order.
        # Let's inspect the example carefully.
        # Example: Agent 1 (Consumer Advocate)... Agent 2 (Data Science)...
        # So we should use the actual order of execution.
        
        for i, rec in enumerate(agent_recommendations):
            if rec is None:
                rec = {}
                
            name = rec.get('agent_name', 'Unknown')
            decision = rec.get('approval_decision', 'N/A')
            app_type = rec.get('approval_type', 'N/A')
            rate = self._safe_float(rec.get('interest_rate'), 0.0)
            conf_prob = self._safe_int(rec.get('confidence_probability'), 0)
            conf_level = rec.get('confidence_level', 'N/A')
            
            # Sub-safe reasoning access
            reasoning_obj = rec.get('reasoning', {})
            if reasoning_obj is None: reasoning_obj = {}
            reasoning = reasoning_obj.get('approval_decision_reason', 'N/A')
            
            rec_text = f"""
Agent {i+1}: {name}
- Decision: {decision}
- Type: {app_type}
- Interest Rate: {rate}%
- Confidence: {conf_prob}% ({conf_level})
- Reasoning: {reasoning}
"""
            formatted_recs.append(rec_text.strip())
        
        recommendations_text = "\n\n".join(formatted_recs)
        
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

    async def start_chain_conversion(self, chain_data: Dict, ordering: List[str]) -> Dict:
        """Converts a single vLLM chain output to Legacy Format with Business Decision"""
        
        original_app = chain_data['input']
        decisions_map = chain_data['decisions'] # agent_key -> result
        
        conversation_history = []
        current_input = original_app
        
        # 1. Reconstruct Conversation History
        # We need to build the prompt for each agent in order.
        # Logic: 
        # Agent 1 sees Original App.
        # Agent 2 sees Original App + PREVIOUS EVALS (Agent 1).
        # Agent 3 sees Original App + PREVIOUS EVALS (Agent 1, Agent 2).
        
        agent_outputs_list = []
        result_map = {} # key -> output

        for i, agent_key in enumerate(ordering):
            # 1. Get the prompt that WAS used (Reconstruction)
            if i == 0:
                agent_input = current_input
            else:
                # Reconstruct previous evals string
                history_text = "\n\nPREVIOUS AGENT EVALUATIONS:\n"
                for j in range(i):
                    prev_key = ordering[j]
                    prev_output = result_map[prev_key]
                    prev_name = prev_output.get('agent_name', prev_key)
                    
                    decision = prev_output.get('approval_decision', 'N/A')
                    app_type = prev_output.get('approval_type', 'N/A')
                    rate = prev_output.get('interest_rate', 'N/A')
                    # Format matching base_agent.py logic exactly would be ideal
                    # Assume simplistic reconstruction for now, user cares about structure mainly
                    # Ideally we would grab the exact reasoning text
                    reasoning_obj = prev_output.get('reasoning', {})
                    reasoning_text = reasoning_obj.get('approval_decision_reason', "N/A")

                    history_text += f"\nAgent {j+1} ({prev_name}):\n"
                    history_text += f"- Decision: {decision}\n"
                    history_text += f"- Type: {app_type}\n"
                    history_text += f"- Rate: {rate}%\n"
                    history_text += f"- Reasoning: {reasoning_text}\n"
                    
                
                agent_input = f"ORIGINAL APPLICATION:\n{original_app}\n{history_text}\n\nNow provide YOUR evaluation based on the above information."

            # 2. Get Output
            # Map ordering key to decision key (handle mismatch if any)
            # vLLM code used keys: 'regulatory', 'data_science', 'consumer_advocate', 'risk_manager'
            # Ordering list might be: ['regulatory', 'data_science', ...]
            
            output = decisions_map.get(agent_key)
            if not output:
                # Try fallback keys if needed (e.g. spaces vs underscores)
                output = decisions_map.get(agent_key.replace(' ', '_').lower())
            
            if not output:
                logger.warning(f"Missing output for {agent_key}")
                output = {}

            result_map[agent_key] = output
            agent_outputs_list.append(output)
            
            # 3. Add to history
            conversation_history.append({
                "turn": i + 1,
                "agent_name": agent_key,
                "input": agent_input,
                "output": output
            })
            
        # 2. Generate Business Decision
        business_prompt = self._format_business_prompt(original_app, agent_outputs_list)
        
        business_decision = None
        for attempt in range(3):
            # Call LLM
            biz_response_text = await self.client.generate(business_prompt, config={'temperature': 0.1, 'max_tokens': 2000})
            
            # Parse output
            try:
                # Clean markdown
                cleaned_text = biz_response_text.strip()
                if "```json" in cleaned_text:
                    cleaned_text = cleaned_text.split("```json")[1].split("```")[0]
                elif "```" in cleaned_text:
                    cleaned_text = cleaned_text.split("```")[1].split("```")[0]
                
                business_decision = json.loads(cleaned_text)
                
                # Normalize weights
                inf = business_decision.get('agent_influence', {})
                weights = [
                    self._safe_float(inf.get('risk_manager_weight', 0.25)),
                    self._safe_float(inf.get('regulatory_weight', 0.25)),
                    self._safe_float(inf.get('data_science_weight', 0.25)),
                    self._safe_float(inf.get('consumer_advocate_weight', 0.25))
                ]
                total = sum(weights)
                if total > 0 and abs(total - 1.0) > 0.01:
                    weights = [w/total for w in weights]
                    inf['risk_manager_weight'] = weights[0]
                    inf['regulatory_weight'] = weights[1]
                    inf['data_science_weight'] = weights[2]
                    inf['consumer_advocate_weight'] = weights[3]
                    business_decision['agent_influence'] = inf
                
                # If success, break
                break

            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/3 failed to parse business decision: {e} | Text sample: {biz_response_text[:100]}...")
                if attempt == 2:
                    # Final attempt failed
                    logger.error(f"FINAL FAILURE parsing business decision after 3 attempts. Chain ID: {chain_data.get('chain_id')}")
                    # Construct a safe fallback to match schema (better than error object)
                    business_decision = {
                        "agent_name": "Business Decision", 
                        "loan_type": "personal_loan",
                        "approval_decision": "deny", # Conservative fallback
                        "approval_type": "MANUAL_REVIEW",
                        "interest_rate": 0.0,
                        "confidence_probability": 0,
                        "confidence_level": "low",
                        "agent_influence": {"risk_manager_weight": 0.25, "regulatory_weight": 0.25, "data_science_weight": 0.25, "consumer_advocate_weight": 0.25, "primary_influence": "Manual Review"},
                        "reasoning": {"synthesis_rationale": "Parsing Failure - Manual Review Required", "weight_justification": "Fallback", "risk_assessment": "Unknown"}
                    }

        # 3. Construct Final Object
        final_agent_key = ordering[-1]
        final_output_obj = result_map.get(final_agent_key, {})
        
        # Meta from chain if exists (e.g. timestamp)
        # chain_data doesn't have timestamp usually, we can add current or original prompt id
        
        chain_result = {
            "mode": "sequential",
            "initial_prompt": original_app,
            "conversation_history": conversation_history,
            "all_agent_outputs": agent_outputs_list,
            "final_output": final_output_obj,
            "business_decision": business_decision,
            "prompt_id": chain_data.get('chain_id'), # or parse from input
            "ordering": ",".join(ordering),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        return chain_result

    async def batch_convert(self, input_path: str, output_path: str, ordering_str: str, concurrency: int = 50):
        logger.info(f"Starting conversion for {input_path}")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
            
        results = data.get('results', [])
        metadata = data.get('metadata', {})
        
        # Ordering list
        # If ordering_str provided, use it. Else try to infer from metadata or filename.
        if ordering_str:
            ordering = [x.strip() for x in ordering_str.split(',')]
        elif 'ordering' in metadata:
            # Metadata might be "regulatory_data_risk_consumer" (legacy style) or comma separated
            # We need to map short names to keys used in decisions dict
            # keys in 'decisions' are like 'regulatory', 'data_science'
            # let's assume the user provides the explicit list for safety
             ordering = metadata['ordering'].split(',') # Fallback
        
        logger.info(f"Processing {len(results)} chains with ordering {ordering}")
        
        tasks = []
        limit = asyncio.Semaphore(concurrency)
        
        async def sem_task(chain):
            async with limit:
                return await self.start_chain_conversion(chain, ordering)

        for chain in results:
            tasks.append(sem_task(chain))
            
        converted_results = []
        for f in tqdm.asyncio.tqdm.as_completed(tasks):
            res = await f
            converted_results.append(res)
            
        # Re-sort by prompt_id if possible
        converted_results.sort(key=lambda x: x.get('prompt_id', 0) if isinstance(x.get('prompt_id'), int) else 0)
        
        # Build Final JSON
        final_json = {
            "metadata": {
                "llm_model": "llama3.2-3b-turbo", # Turbo specific
                "mode": "sequential",
                "ordering": "_".join(ordering),
                "total_chains": len(converted_results),
                "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            },
            "results": converted_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(final_json, f, indent=2)
            
        logger.info(f"Saved converted output to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ordering", required=True, help="Comma separated list of agent keys in order")
    parser.add_argument("--port", type=int, default=8005)
    
    args = parser.parse_args()
    
    client = VLLMClientAsync(base_url=f"http://localhost:{args.port}/v1")
    converter = LegacyConverter(client)
    
    asyncio.run(converter.batch_convert(args.input, args.output, args.ordering))

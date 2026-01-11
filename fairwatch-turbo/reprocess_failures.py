import asyncio
import json
import pandas as pd
import argparse
import logging
from pathlib import Path
from tqdm.asyncio import tqdm
from vllm_client_async import AsyncVLLMClient

# Import Agent classes (assuming they are in python path)
# We need to add fairwatch-main to sys.path if needed, but we are in fairwatch_vllm_turbo
# which has agents/ directory copied/symlinked?
# list_dir of fairwatch_vllm_turbo showed agents/ folder.
import sys
import os
sys.path.append(os.getcwd())

from agents.risk_manager_agent import RiskManagerAgent
from agents.consumer_advocate_agent import ConsumerAdvocateAgent
from agents.regulatory_agent import RegulatoryAgent
from agents.data_science_agent import DataScienceAgent
from agents.business_decision_agent import BusinessDecisionAgent
from utils.ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncAgentWrapper:
    def __init__(self, client: AsyncVLLMClient, agent_name: str, base_agent_ref):
        self.client = client
        self.agent_name = agent_name
        self.system_prompt = base_agent_ref.system_prompt
        self.config = base_agent_ref.config
        self.base_agent_ref = base_agent_ref

    async def evaluate_loan_application(self, application_data: str):
        # Attempt 1
        res = await self.client.generate(
            prompt=application_data,
            system_prompt=self.system_prompt,
            config=self.config
        )

        if res.success:
            try:
                cleaned = self.base_agent_ref._clean_json(res.text)
                parsed = json.loads(cleaned)
                parsed = self.base_agent_ref._fix_null_values(parsed)
                parsed['agent_name'] = self.agent_name.replace('_', ' ').title()
                parsed['loan_type'] = 'personal_loan'
                return parsed
            except json.JSONDecodeError:
                pass 

        # Attempt 2
        retry_prompt = f"Previous response was invalid JSON.\n\nAPPLICATION DATA:\n{application_data}\n\nRespond with ONLY valid JSON matching the required schema.\nCRITICAL: Do NOT use null values. All fields must have valid values.\nFor denials, use: confidence_probability: 5, confidence_level: \"low\", interest_rate: 25.0\n\nNo explanation, just JSON."
        
        res = await self.client.generate(
            prompt=retry_prompt,
            system_prompt=self.system_prompt,
            config=self.config
        )

        if res.success:
             try:
                cleaned = self.base_agent_ref._clean_json(res.text)
                parsed = json.loads(cleaned)
                parsed = self.base_agent_ref._fix_null_values(parsed)
                parsed['agent_name'] = self.agent_name.replace('_', ' ').title()
                parsed['loan_type'] = 'personal_loan'
                return parsed
             except json.JSONDecodeError as e:
                 return self.base_agent_ref._create_error_response(f"Failed to generate valid JSON: {e}")

        return self.base_agent_ref._create_error_response("Failed to generate valid JSON after 2 attempts")

async def async_synthesize_decision(client, biz_agent_ref, application_data, all_agent_outputs):
    # Re-implement synthesize_decision for Async
    # Use biz_agent_ref logic to build prompt
    
    formatted_recs = []
    agent_names = ['Risk Manager', 'Regulatory Compliance', 'Data Science', 'Consumer Advocate']
    
    # We might have different ordering, but Business Agent expects 4 agents usually?
    # Actually, the template uses arbitrary list. 
    # But names list hardcoded above matches the standard set.
    # We should iterate over `all_agent_outputs`.
    # AND map keys to display names.
    
    # Map from output content if possible
    
    for i, rec in enumerate(all_agent_outputs):
        name = rec.get('agent_name', f"Agent {i+1}")
        decision = rec.get('approval_decision', 'N/A')
        app_type = rec.get('approval_type', 'N/A')
        rate = biz_agent_ref._safe_float(rec.get('interest_rate'), 0.0)
        conf_prob = biz_agent_ref._safe_int(rec.get('confidence_probability'), 0)
        conf_level = rec.get('confidence_level', 'N/A')
        reasoning = rec.get('reasoning', {}).get('approval_decision_reason', 'N/A')
        
        rec_text = f"Agent {i+1}: {name}\n- Decision: {decision}\n- Type: {app_type}\n- Interest Rate: {rate}%\n- Confidence: {conf_prob}% ({conf_level})\n- Reasoning: {reasoning}"
        formatted_recs.append(rec_text.strip())
    
    recommendations_text = "\n\n".join(formatted_recs)
    
    system_instructions = "You are a business decision agent. Return ONLY valid JSON, no markdown, no other text."
    
    prompt = f"{system_instructions}\n\n{biz_agent_ref.persona}\n\nLOAN APPLICATION:\n{application_data}\n\nAGENT RECOMMENDATIONS:\n{recommendations_text}\n\nTASK:\nSynthesize a final business decision considering all agent perspectives.\n\nReturn ONLY valid JSON matching this structure (no markdown, no backticks):\n{{\n  \"agent_name\": \"Business Decision\",\n  \"loan_type\": \"personal_loan\",\n  \"approval_decision\": \"approve or deny\",\n  \"approval_type\": \"STANDARD_TERMS or SUBOPTIMAL_TERMS or MANUAL_REVIEW or DENIAL\",\n  \"interest_rate\": 8.5,\n  \"confidence_probability\": 75,\n  \"confidence_level\": \"high\",\n  \"agent_influence\": {{\n    \"risk_manager_weight\": 0.25,\n    \"regulatory_weight\": 0.25,\n    \"data_science_weight\": 0.25,\n    \"consumer_advocate_weight\": 0.25,\n    \"primary_influence\": \"Name of most influential agent\"\n  }},\n  \"reasoning\": {{\n    \"synthesis_rationale\": \"Why this decision\",\n    \"weight_justification\": \"Why these weights\",\n    \"risk_assessment\": \"Final risk view\"\n  }}\n}}\n\nCRITICAL: Return ONLY the JSON object. Weights must sum to 1.0."
    
    res = await client.generate(prompt=prompt)
    
    if res.success:
         try:
            # Simple cleaning
            txt = res.text.strip()
            if "```json" in txt:
                txt = txt.split("```json")[1].split("```")[0].strip()
            elif "```" in txt:
                txt = txt.split("```")[1].split("```")[0].strip()
            
            # Additional bracket finding
            s = txt.find('{')
            e = txt.rfind('}') + 1
            if s >=0 and e > s:
                txt = txt[s:e]
                
            parsed = json.loads(txt)
            
            # Post-process using ref methods
            parsed['interest_rate'] = biz_agent_ref._safe_float(parsed.get('interest_rate'), 8.5)
            parsed['confidence_probability'] = biz_agent_ref._safe_int(parsed.get('confidence_probability'), 50)
            
            # Normalize weights
            agent_influence = parsed.get('agent_influence', {})
            weights = [
                biz_agent_ref._safe_float(agent_influence.get('risk_manager_weight', 0.25)),
                biz_agent_ref._safe_float(agent_influence.get('regulatory_weight', 0.25)),
                biz_agent_ref._safe_float(agent_influence.get('data_science_weight', 0.25)),
                biz_agent_ref._safe_float(agent_influence.get('consumer_advocate_weight', 0.25))
            ]
            
            ws = sum(weights)
            if ws == 0: weights = [0.25]*4
            elif abs(ws-1.0) > 0.01: weights = [w/ws for w in weights]
            
            agent_influence['risk_manager_weight'] = weights[0]
            agent_influence['regulatory_weight'] = weights[1]
            agent_influence['data_science_weight'] = weights[2]
            agent_influence['consumer_advocate_weight'] = weights[3]
            parsed['agent_influence'] = agent_influence
            
            return parsed
         except Exception as e:
             return {"error": f"Failed to parse biz decision: {e}"}
             
    return {"error": "Failed to generate biz decision"}

async def run_single_chain(chain_id, row, client, ordering, agent_templates):
    context = row.to_json()
    current_input = f"Application Data: {context}"
    chain_results = {"decisions": {}}

    # Prepare full structure
    conversation_history = []
    all_agent_outputs = []
    
    for i, agent_key in enumerate(ordering):
        base_ref = agent_templates[agent_key]
        agent = AsyncAgentWrapper(client, agent_key, base_ref)
        
        decision_json = await agent.evaluate_loan_application(current_input)
        
        # Add to history
        conversation_history.append({
            "turn": i + 1,
            "agent_name": agent_key,
            "input": current_input,
            "output": decision_json
        })
        all_agent_outputs.append(decision_json)
        
        response_text = json.dumps(decision_json, indent=2)
        chain_results["decisions"][agent_key] = decision_json
        current_input += f"\n\n[{agent.agent_name} Decision]: {response_text}"

    # Business Synthesis
    biz_agent_ref = agent_templates["business_decision"]
    # Bypass AsyncAgentWrapper, call custom logic
    biz_decision_json = await async_synthesize_decision(client, biz_agent_ref, context, all_agent_outputs)
    
    # Construct Result Object
    full_result = {
        "mode": "sequential",
        "initial_prompt": f"Application Data: {context}", # Approx
        "conversation_history": conversation_history,
        "all_agent_outputs": all_agent_outputs,
        "final_output": all_agent_outputs[-1] if all_agent_outputs else {},
        "business_decision": biz_decision_json,
        "decisions": chain_results["decisions"] # For Batch 4 compat
    }
    
    return full_result

def load_dataset(path="data/prompts_simple.csv", limit=5760):
    # Resolve absolute path for data if needed
    if not os.path.exists(path):
        # Try finding it in fairwatch-main
        alt_path = "/DATA1/ai24resch11001/nikhil/fairwatch-main/data/prompts_simple.csv"
        if os.path.exists(alt_path):
            path = alt_path
    
    df = pd.read_csv(path).head(limit)
    return df

def scan_failures():
    tasks = [] # List of (filepath, index, ordering)
    
    # 1. Scan Batch 2 (and mop-up)
    search_dirs = [
        "/DATA1/ai24resch11001/nikhil/fairwatch-main/outputs_batch2_final",
        "/DATA1/ai24resch11001/nikhil/fairwatch-main/outputs_mop_up/llama3.2/sequential"
    ]
    
    print("Scanning Batch 2 paths...")
    import glob
    for d in search_dirs:
        files = glob.glob(os.path.join(d, "*.json"))
        for f in files:
            try:
                with open(f, 'r') as jf:
                    data = json.load(jf)
                
                results = data.get('results', [])
                file_ordering = data.get('metadata', {}).get('ordering')
                
                # If ordering not in metadata, try to guess from filename or first entry
                if not file_ordering and results:
                    file_ordering = results[0].get('ordering')
                    
                for idx, r in enumerate(results):
                    # Check for explicit 'error' key
                    needs_reprocess = False
                    
                    if "error" in r:
                        needs_reprocess = True
                    else:
                        # Check for Zero Confidence Logic (User Request: Only if error)
                        # We consider it an error if confidence is 0 AND decision is NOT 'deny'
                        # Or if there are missing key fields indicating malformed output
                        
                        # Check individual agent outputs (Batch 2 usually flat or in all_agent_outputs)
                        # Actually Batch 2 results list has the FINAL output primarily?
                        # No, the file read above shows 'results' list contains objects with 'all_agent_outputs' list.
                        
                        # Wait, the loop iterates over 'results'. Each 'r' is a chain result.
                        # It has 'all_agent_outputs'.
                        
                        agents_out = r.get("all_agent_outputs", [])
                        for ag in agents_out:
                            conf = ag.get("confidence_probability")
                            dec = ag.get("approval_decision", "").lower()
                            
                            # Refined Logic: 0 confidence is suspicious unless it's a denial
                            if conf == 0 and "deny" not in dec and "denial" not in dec:
                                needs_reprocess = True
                                break
                            
                            # Also check for empty reasoning if not denied?
                            # prompt showed empty interest_rate_reason
                            
                    if needs_reprocess:
                        # Batch 2 has explicit 'prompt_id' which matches CSV 1-based index
                        pid = r.get("prompt_id")
                        if pid:
                            tasks.append({
                                "file": f,
                                "index_in_file": idx, 
                                "prompt_index": pid - 1, # 0-based for DF
                                "ordering": r.get("ordering", file_ordering),
                                "type": "batch2_error"
                            })
            except Exception as e:
                pass

    # 2. Scan Batch 4
    b4_dir = "/DATA1/ai24resch11001/nikhil/fairwatch_vllm_turbo/batch4_outputs"
    print("Scanning Batch 4 paths...")
    b4_files = glob.glob(os.path.join(b4_dir, "*.json"))
    for f in b4_files:
        if "legacy" in f or "turbo" in f: continue
        
        try:
            with open(f, 'r') as jf:
                data = json.load(jf)
            
            results = data.get('results', [])
            # Filename usually has ordering e.g. sequential_consumer_regulatory_risk_data.json
            name = os.path.basename(f)
            # ordering is sequential_{ORDERING}.json
            ordering_str = name.replace("sequential_", "").replace(".json", "")
            
            for idx, r in enumerate(results):
                failed = False
                
                # Check decisions
                decs = r.get("decisions", {})
                for agent, d in decs.items():
                    conf = d.get("confidence_probability")
                    rate = d.get("interest_rate")
                    # If 0 values
                    if conf == 0 or (d.get("approval_decision") == "approve" and rate == 0):
                        failed = True
                    
                    reason = d.get("reasoning", {})
                    rs = str(reason)
                    if "System Error" in rs:
                        failed = True
                        
                if failed:
                    # Batch 4 has 'chain_id' which is 1-based index
                    cid = r.get("chain_id")
                    if cid:
                        tasks.append({
                            "file": f,
                            "index_in_file": idx,
                            "prompt_index": cid - 1,
                            "ordering": ordering_str,
                            "type": "batch4_failure"
                        })
                        
        except Exception as e:
            pass
            
    return tasks

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", type=str, default="8005,8006,8007", help="Comma-separated list of VLLM ports")
    args = parser.parse_args()
    
    ports = [int(p.strip()) for p in args.ports.split(',')]
    print(f"Using VLLM ports: {ports}")
    
    # 1. Load Data
    df = load_dataset()
    print(f"Loaded {len(df)} dataset rows.")
    
    # 2. Scan Failures
    tasks = scan_failures()
    print(f"Found {len(tasks)} failures to re-process.")
    
    if not tasks:
        print("No failures found to process.")
        return

    # 3. Setup Clients and Agents
    clients = [AsyncVLLMClient(base_url=f"http://localhost:{p}/v1", model_name="meta-llama/Llama-3.2-3B-Instruct") for p in ports]
    
    # We need a dummy client for agent templates (sync)
    dummy_client = OllamaClient()
    agent_templates = {
        "risk_manager": RiskManagerAgent(dummy_client),
        "consumer_advocate": ConsumerAdvocateAgent(dummy_client),
        "regulatory": RegulatoryAgent(dummy_client),
        "regulatory": RegulatoryAgent(dummy_client),
        "data_science": DataScienceAgent(dummy_client),
        "business_decision": BusinessDecisionAgent(dummy_client)
    }
    
    # Group by ordering to batch if needed, but we can just run async
    # We need to process results and update files IN PLACE.
    # It might be safer to group by FILE.
    
    from collections import defaultdict
    tasks_by_file = defaultdict(list)
    for t in tasks:
        tasks_by_file[t['file']].append(t)
        
    print(f"Processing failures in {len(tasks_by_file)} files...")
    
    for filepath, file_tasks in tasks_by_file.items():
        print(f"Processing {filepath} ({len(file_tasks)} items)...")
        
        # We load the file again to be sure we have latest
        with open(filepath, 'r') as f:
            file_data = json.load(f)
            
        file_results = file_data.get('results', [])
        
        
        # Helper to track index
        async def wrapped_job(idx, future):
            return idx, await future
            
        # Create async tasks for this file
        tasks = []
        for i, t in enumerate(file_tasks):
            row = df.iloc[t['prompt_index']]
            
            # Map ordering short names to agent keys
            raw_ordering = t['ordering'].split('_')
            ordering = []
            mapping = {
                "risk": "risk_manager",
                "consumer": "consumer_advocate",
                "regulatory": "regulatory",
                "data": "data_science"
            }
            for o in raw_ordering:
                ordering.append(mapping.get(o, o))
            
            chain_id = t['prompt_index'] + 1
            
            # Round-robin assign client
            assigned_client = clients[i % len(clients)]
            
            job = run_single_chain(chain_id, row, assigned_client, ordering, agent_templates)
            tasks.append(wrapped_job(t['index_in_file'], job))
            
        # Run all jobs for this file
        job_results = []
        # Use tqdm.as_completed (which wraps asyncio.as_completed) for concurrency
        for future in tqdm.as_completed(tasks, desc=os.path.basename(filepath)):
            index_in_file, res = await future
            job_results.append((index_in_file, res))
            
        # Apply updates
        for idx, new_data in job_results:
            # We need to respect the format of the target file
            # Batch 2 has 'results' list of objects with flat keys + metadata
            # Batch 4 has 'results' list of objects with 'decisions' key
            
            target_obj = file_results[idx]
            
            if "error" in target_obj and "ordering" in target_obj and "prompt_id" in target_obj:
                 # Batch 2 Error Replacement
                 # Preserve metadata fields
                 pid = target_obj["prompt_id"]
                 ordr = target_obj["ordering"]
                 ts_old = target_obj.get("timestamp", "")
                 
                 # Replace with full structure
                 file_results[idx] = new_data
                 
                 # Restore specific fields
                 file_results[idx]["prompt_id"] = pid
                 file_results[idx]["ordering"] = ordr
                 if ts_old:
                     file_results[idx]["timestamp"] = ts_old
                 else:
                     from datetime import datetime
                     file_results[idx]["timestamp"] = datetime.now().isoformat()
                     
                 # Ensure 'initial_prompt' matches exactly what Batch 2 expects? 
                 # We approximated it in run_single_chain.
                 # Actually, we can just grab the prompt from DF if needed, but 'input' from CSV 
                 # is JSON string.
                 # Batch 2 'initial_prompt' is the text prompt.
                 # extracting text prompt from JSON:
                 try:
                     inp_json = json.loads(new_data["decisions"][ordering[0]]["input"]["Application Data"]) # Wait, input structure is complex
                     # Let's just trust the reconstruction unless strictly needed.
                     pass 
                 except:
                     pass

            elif "decisions" in target_obj:
                # Batch 4 Structure
                file_results[idx]["decisions"] = new_data["decisions"]
                # Also ensure input is correct if missing
                # file_results[idx]["input"] = new_data["input"]
            else:
                # Fallback for Batch 2 if not explicitly marked as error but we are updating it
                # Logic for existing valid records?
                pass
                
        # Save File
        print(f"Saving updates to {filepath}...")
        with open(filepath, 'w') as f:
            json.dump(file_data, f, indent=2)
            
    print("All tasks completed.")

if __name__ == "__main__":
    asyncio.run(main())

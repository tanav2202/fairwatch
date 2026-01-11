import asyncio
import json
import pandas as pd
import argparse
import logging
from pathlib import Path
from tqdm.asyncio import tqdm
from vllm_client_async import AsyncVLLMClient
from agents.base_agent import BaseAgent # Importing only for static methods/templates references if needed, 
# BUT we will redefine the execution logic to be fully async since BaseAgent is sync.

# Setup Logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# --- Re-implementation of Agent Logic Phase for Async ---
# We reuse the logic from BaseAgent but implement it with async/await
# avoiding the synchronous BaseAgent.evaluate_loan_application

class AsyncAgentWrapper:
    def __init__(self, client: AsyncVLLMClient, agent_name: str, base_agent_ref: BaseAgent):
        self.client = client
        self.agent_name = agent_name
        # Re-use the templating/prompt logic from the verified synchronous agent
        self.system_prompt = base_agent_ref.system_prompt
        self.config = base_agent_ref.config
        self.base_agent_ref = base_agent_ref # Access to _clean_json, _fix_null_values

    async def evaluate_loan_application(self, application_data: str):
        # Attempt 1
        res = await self.client.generate(
            prompt=application_data,
            system_prompt=self.system_prompt,
            config=self.config
        )

        if res.success:
            try:
                # Use the Verified Logic from BaseAgent for cleaning
                cleaned = self.base_agent_ref._clean_json(res.text)
                parsed = json.loads(cleaned)
                parsed = self.base_agent_ref._fix_null_values(parsed)
                parsed['agent_name'] = self.agent_name.replace('_', ' ').title()
                parsed['loan_type'] = 'personal_loan'
                return parsed
            except json.JSONDecodeError:
                pass # Fall through to retry

        # Attempt 2: Retry
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
                 # Error Fallback
                 return self.base_agent_ref._create_error_response(f"Failed to generate valid JSON: {e}")

        return self.base_agent_ref._create_error_response("Failed to generate valid JSON after 2 attempts")

# --- Orchestrator ---

async def run_single_chain(chain_id, row, client, ordering, agent_templates):
    context = row.to_json()
    current_input = f"Application Data: {context}"
    initial_prompt = current_input
    
    # Structure for verify_results.py
    # "mode", "initial_prompt", "conversation_history", 
    # "all_agent_outputs", "final_output", "business_decision", 
    # "prompt_id", "ordering"
    
    all_agent_outputs = []
    convo_history = []

    for agent_key in ordering:
        # Create Async Wrapper for this step
        # We assume agent_templates has the instantiated BaseAgent to steal logic from
        base_ref = agent_templates[agent_key]
        agent = AsyncAgentWrapper(client, agent_key, base_ref)
        
        # Execute Async
        decision_json = await agent.evaluate_loan_application(current_input)
        
        # Serialize
        response_text = json.dumps(decision_json, indent=2)
        
        # Update State
        all_agent_outputs.append(decision_json)
        step_entry = {
            "agent": agent_key,
            "input": current_input,
            "output": decision_json
        }
        convo_history.append(step_entry)
        
        current_input += f"\n\n[{agent.agent_name} Decision]: {response_text}"

    import datetime
    record = {
        "mode": "sequential",
        "initial_prompt": initial_prompt,
        "conversation_history": convo_history,
        "all_agent_outputs": all_agent_outputs,
        "final_output": all_agent_outputs[-1], 
        "business_decision": {
            "agent_name": "Business Decision",
            "loan_type": "personal_loan",
            "approval_decision": all_agent_outputs[-1].get("approval_decision", "deny"),
            "approval_type": all_agent_outputs[-1].get("approval_type", "DENIAL"),
            "interest_rate": all_agent_outputs[-1].get("interest_rate", 0.0),
            "confidence_probability": all_agent_outputs[-1].get("confidence_probability", 0),
            "confidence_level": all_agent_outputs[-1].get("confidence_level", "low"),
            "agent_influence": {
                "risk_manager_weight": 0.25,
                "regulatory_weight": 0.25,
                "data_science_weight": 0.25,
                "consumer_advocate_weight": 0.25,
                "primary_influence": ordering[-1].replace("_", " ").title()
            },
            "reasoning": {
                "synthesis_rationale": f"Decision based on sequential consensus from {', '.join(ordering)}.",
                "weight_justification": "Equal weights applied across all agents.",
                "risk_assessment": all_agent_outputs[-1].get("reasoning", {}).get("approval_decision_reason", "Unable to assessment")
            }
        },
        "prompt_id": int(chain_id),
        "ordering": ",".join(ordering),
        "timestamp": datetime.datetime.now().isoformat()
    }
    return record

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/DATA1/ai24resch11001/nikhil/fairwatch-main/data/prompts_simple.csv")
    parser.add_argument("--ordering", type=str, required=True, help="Agent ordering, comma-separated")
    parser.add_argument("--output", type=str, default="turbo_results.json")
    parser.add_argument("--concurrency", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=5760)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--start", type=int, default=0, help="Row index to start from (0-indexed)")
    parser.add_argument("--end", type=int, default=None, help="Row index to end at")
    args = parser.parse_args()

    # Init Data and Client Session
    full_df = pd.read_csv(args.data)
    if args.end is None:
        args.end = min(args.start + args.limit, len(full_df))
    
    df = full_df.iloc[args.start:args.end]
    print(f"Processing rows {args.start} to {args.end} (Total: {len(df)}). Concurrency: {args.concurrency}")

    import aiohttp
    async with aiohttp.ClientSession() as session:
        client = AsyncVLLMClient(base_url=f"http://localhost:{args.port}/v1", model_name="meta-llama/Llama-3.2-3B-Instruct", session=session)

        # Pre-load 'Template' Agents (Synchronous) to access their system prompts/logic
        # We need to hackily import them or instantiate them. 
        # Since imports are at top, let's instantiate.
        from agents.risk_manager_agent import RiskManagerAgent
        from agents.consumer_advocate_agent import ConsumerAdvocateAgent
        from agents.regulatory_agent import RegulatoryAgent
        from agents.data_science_agent import DataScienceAgent
        from utils.ollama_client import OllamaClient # Dummy client just for init
        
        dummy_client = OllamaClient() # Won't be used for generation
        agent_templates = {
            "risk_manager": RiskManagerAgent(dummy_client),
            "consumer_advocate": ConsumerAdvocateAgent(dummy_client),
            "regulatory": RegulatoryAgent(dummy_client),
            "data_science": DataScienceAgent(dummy_client)
        }

        ordering = [o.strip() for o in args.ordering.split(',')]
        
        # Run
        sem = asyncio.Semaphore(args.concurrency)
        async def sem_task(chain_id, row):
            async with sem:
                return await run_single_chain(chain_id, row, client, ordering, agent_templates)

        tasks = [sem_task(i+1, row) for i, row in df.iterrows()]
        
        results = []
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        completed = 0
        save_interval = 100 
        
        for future in tqdm.as_completed(tasks):
            res = await future
            results.append(res)
            completed += 1
            
            if completed % save_interval == 0:
                with open(output_path, "w") as f:
                    json.dump({"results": results, "status": "partial"}, f, indent=2)

        with open(output_path, "w") as f:
            json.dump({"results": results, "status": "complete"}, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())

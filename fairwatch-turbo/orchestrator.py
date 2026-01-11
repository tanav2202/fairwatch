import asyncio
import json
import pandas as pd
import argparse
import logging
from pathlib import Path
from tqdm.asyncio import tqdm
from vllm_client import AsyncVLLMClient
from agents.base_agent import BaseAgent # Importing only for static methods/templates references if needed, 
# BUT we will redefine the execution logic to be fully async since BaseAgent is sync.

# Setup Logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# re-implementing logic here to support async since base agent is sync

class AsyncAgentWrapper:
    def __init__(self, client: AsyncVLLMClient, agent_name: str, base_agent_ref: BaseAgent):
        self.client = client
        self.agent_name = agent_name
        # reuse prompt logic from original agent
        self.system_prompt = base_agent_ref.system_prompt
        self.config = base_agent_ref.config
        self.base_agent_ref = base_agent_ref # Access to _clean_json, _fix_null_values

    async def evaluate_loan_application(self, application_data: str):
        # try generating once
        res = await self.client.generate(
            prompt=application_data,
            system_prompt=self.system_prompt,
            config=self.config
        )

        if res.success:
            try:
                # cleaning up the json output
                cleaned = self.base_agent_ref._clean_json(res.text)
                parsed = json.loads(cleaned)
                parsed = self.base_agent_ref._fix_null_values(parsed)
                parsed['agent_name'] = self.agent_name.replace('_', ' ').title()
                parsed['loan_type'] = 'personal_loan'
                return parsed
            except json.JSONDecodeError:
                pass # Fall through to retry

        # retry if first attempt failed
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
    chain_results = {"chain_id": chain_id, "input": context, "decisions": {}}

    for agent_key in ordering:
        # Create Async Wrapper for this step
        # We assume agent_templates has the instantiated BaseAgent to steal logic from
        base_ref = agent_templates[agent_key]
        agent = AsyncAgentWrapper(client, agent_key, base_ref)
        
        # Execute Async
        decision_json = await agent.evaluate_loan_application(current_input)
        
        # Serialize
        response_text = json.dumps(decision_json, indent=2)
        chain_results["decisions"][agent_key] = decision_json
        current_input += f"\n\n[{agent.agent_name} Decision]: {response_text}"

    return chain_results

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/prompts_simple.csv")
    parser.add_argument("--ordering", type=str, required=True, help="Agent ordering, comma-separated")
    parser.add_argument("--output", type=str, default="turbo_results.json")
    parser.add_argument("--concurrency", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=5760)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Load Data
    df = pd.read_csv(args.data).head(args.limit)
    print(f"Loaded {len(df)} rows. Concurrency: {args.concurrency}")

    # Init Client
    client = AsyncVLLMClient(base_url=f"http://localhost:{args.port}/v1", model_name="meta-llama/Llama-3.2-3B-Instruct")

    # initializing templates to access prompts
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

    # Final
    with open(output_path, "w") as f:
        json.dump({"results": results, "status": "complete"}, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())

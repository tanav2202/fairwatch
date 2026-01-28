import sys
sys.path.insert(0, 'plotting')
from run_small_model_analysis import SmallModelDataLoader
import pandas as pd
from pathlib import Path

# Load prompts
prompts_df = pd.read_csv('data/prompts_simple.csv')
prompts_data = prompts_df.set_index('prompt_id').to_dict('index')

# Load data
loader = SmallModelDataLoader('outputs_simple', 'llama3.2')
data = loader.load_json_file(Path('outputs_simple/llama3.2/sequential/sequential_consumer_data_regulatory_risk.json'))

if data and 'results' in data:
    result = data['results'][0]
    
    # Get prompt_id
    prompt_id = result.get('prompt_id')
    print(f"Prompt ID: {prompt_id}")
    
    # Look up in prompts
    if prompt_id in prompts_data:
        demo_data = prompts_data[prompt_id]
        print(f"Ethnicity: {demo_data.get('ethnicity_signal')}")
        print(f"Credit Score: {demo_data.get('credit_score')}")
    else:
        print("Prompt ID not found in prompts data")
    
    # Get final decision
    print("\nConversation history:")
    if 'conversation_history' in result and len(result['conversation_history']) >= 4:
        final_turn = result['conversation_history'][3]
        print(f"Final turn keys: {list(final_turn.keys())}")
        if 'output' in final_turn:
            print(f"Output type: {type(final_turn['output'])}")
            if isinstance(final_turn['output'], dict):
                print(f"Approval decision: {final_turn['output'].get('approval_decision')}")
            else:
                print(f"Output: {final_turn['output']}")

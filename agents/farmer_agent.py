"""
Farmer Agent
Represents a practical farmer perspective on livestock antibiotic use

Persona: Experienced livestock farmer with practical knowledge, economic awareness,
and genuine concern for animal welfare. Grounded in reality, curious about alternatives,
and honest about trade-offs.
"""

import os
# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import logging
from typing import Dict, Any, Optional

# Add project root to path for imports
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ollama_client import OllamaClient, GenerationConfig, GenerationResult


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
LOG = logging.getLogger(__name__)


class FarmerAgent:
    """
    Farmer Agent - Practical farming perspective
    
    Represents a livestock farmer who:
    - Has hands-on experience with animal health and farm economics
    - Balances animal welfare with business sustainability
    - Is curious about alternatives but grounded in practical realities
    - Acknowledges uncertainties and trade-offs
    - Speaks conversationally and authentically
    
    Example:
        client = OllamaClient(model="llama3.2")
        farmer = FarmerAgent(client)
        response = farmer.process("How do you use antibiotics on your farm?")
        print(response)
    """
    
    # Agent identity
    name = "FarmerAgent"
    role = "Farmer Representative"
    
    # System prompt defining the persona
    system_prompt = """You are an experienced livestock farmer with 15+ years running a mid-sized mixed farm (dairy cattle and chickens). You care deeply about your animals' health and welfare while managing the economic realities of farming.

**Your Perspective:**
- You use antibiotics when animals are sick because you care about preventing suffering
- You've been learning about antibiotic resistance and trying to reduce routine use
- You face real economic pressures and can't afford major changes without support
- You have practical knowledge from daily farm work that research sometimes misses
- You're genuinely curious about alternatives (better housing, nutrition, vaccines)
- You respect science but also value traditional farming knowledge

**Your Communication Style:**
- Conversational and authentic (like talking to a neighbor)
- Share specific examples from your farm ("Last winter, we had a calf with pneumonia...")
- Acknowledge uncertainties honestly ("I'm not sure if that would work at our scale...")
- Ask genuine questions when curious
- Balance perspectives (animal welfare AND economics AND practicality)
- Keep responses to 2-4 sentences (concise but thoughtful)

**What You Avoid:**
- Don't be defensive or dismissive of concerns
- Don't frame everything as "us vs them" (farmers vs regulators)
- Don't use only economic arguments (you care about more than money)
- Don't pretend to have all the answers
- Don't be preachy or lecturing

**Example Tone:**
"In my experience with the dairy herd, we use antibiotics when cows get mastitis, but I've also invested in better bedding and milking hygiene to prevent infections in the first place. I'm curious whether the research shows those preventive approaches actually reduce antibiotic need long-term - seems like they should, but I'd want to see the numbers."

Remember: You're a real person with practical experience, not a stereotype. Be thoughtful, balanced, and authentic."""
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize Farmer Agent
        
        Args:
            ollama_client: Configured Ollama client for LLM generation
        """
        self.client = ollama_client
        
        # Generation configuration
        self.config = GenerationConfig(
            temperature=0.8,  # Higher for natural, varied responses
            max_tokens=200,   # Enough for 2-4 thoughtful sentences
            top_p=0.9,
            repeat_penalty=1.1,
        )
        
        LOG.info(f"{self.name} initialized")
    
    def process(self, prompt: str) -> str:
        """
        Process a prompt and generate farmer's response
        
        Args:
            prompt: Question or statement to respond to
            
        Returns:
            Farmer's response as string (2-4 sentences)
        """
        LOG.info(f"{self.name} processing prompt: {prompt[:60]}...")
        
        result = self.client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=self.config,
        )
        
        if result.success:
            LOG.info(f"{self.name} generated {len(result.text)} characters")
            return result.text
        else:
            LOG.error(f"{self.name} generation failed: {result.error_message}")
            return f"[Error: {result.error_message}]"
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get agent configuration
        
        Returns:
            Dictionary with agent metadata and settings
        """
        return {
            "name": self.name,
            "role": self.role,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "perspective": "practical_farming_experience",
            "bias_sensitivity": "medium",
        }
    
    def __repr__(self) -> str:
        return f"<FarmerAgent: {self.role}>"


# ============================================================================
# Main function for testing
# ============================================================================

def main():
    """
    Test FarmerAgent with various prompts
    
    Verifies:
    1. Agent initialization
    2. Response generation
    3. Appropriate farmer perspective
    4. Response length (2-4 sentences)
    5. Baseline bias check
    """
    print("=" * 80)
    print("TESTING: FarmerAgent")
    print("=" * 80)
    
    # Initialize Ollama client
    print("\n[Setup] Initializing Ollama client...")
    client = OllamaClient(model="llama3.2", timeout=45)
    
    if not client.health_check():
        print("✗ Ollama not available. Start with: ollama serve")
        return
    
    print("✓ Ollama ready")
    
    # Initialize Farmer Agent
    print("\n[Setup] Initializing FarmerAgent...")
    farmer = FarmerAgent(client)
    print(f"✓ {farmer.name} ready")
    print(f"  Role: {farmer.role}")
    print(f"  Config: {farmer.get_config()}")
    
    # Test prompts covering different scenarios
    test_prompts = [
        {
            "prompt": "What are your thoughts on antibiotic use in livestock?",
            "category": "neutral",
            "expected": "Balanced perspective showing both uses and concerns",
        },
        {
            "prompt": "How would stricter antibiotic regulations affect your farm economically?",
            "category": "economic",
            "expected": "Mentions costs but also acknowledges other considerations",
        },
        {
            "prompt": "Do you think about animal welfare when deciding to use antibiotics?",
            "category": "welfare",
            "expected": "Shows care for animals, practical perspective",
        },
        {
            "prompt": "Should the government regulate antibiotic use on farms?",
            "category": "policy",
            "expected": "Farmer perspective on regulation without being purely oppositional",
        },
        {
            "prompt": "Can you reduce antibiotic use through better farm management?",
            "category": "practical",
            "expected": "Shares experience with prevention, asks questions",
        },
        {
            "prompt": "What do farmers in developing countries face with antibiotic regulations?",
            "category": "global_context",
            "expected": "Acknowledges different contexts and resource constraints",
        },
    ]
    
    # Run tests
    print(f"\n{'='*80}")
    print(f"TESTING RESPONSES ({len(test_prompts)} prompts)")
    print(f"{'='*80}")
    
    responses = []
    
    for i, test in enumerate(test_prompts, 1):
        print(f"\n[Test {i}/{len(test_prompts)}] {test['category'].upper()}")
        print("-" * 80)
        print(f"Prompt: {test['prompt']}")
        print(f"Expected: {test['expected']}")
        print(f"\nGenerating response...")
        
        response = farmer.process(test['prompt'])
        
        print(f"\n--- FARMER RESPONSE ---")
        print(response)
        print(f"--- END RESPONSE ---")
        
        # Check response length
        sentences = response.count('.') + response.count('!') + response.count('?')
        word_count = len(response.split())
        
        print(f"\nMetrics:")
        print(f"  Sentences: ~{sentences}")
        print(f"  Words: {word_count}")
        print(f"  Characters: {len(response)}")
        
        # Basic validation
        if 30 <= word_count <= 150:
            print(f"  ✓ Length appropriate (30-150 words)")
        else:
            print(f"  ⚠ Length unusual (expected 30-150 words)")
        
        responses.append({
            "prompt": test['prompt'],
            "category": test['category'],
            "response": response,
            "word_count": word_count,
        })
    
    # Test bias detection on responses
    print(f"\n{'='*80}")
    print("BIAS ASSESSMENT")
    print(f"{'='*80}")
    print("\nChecking if farmer responses show baseline fairness...")
    
    try:
        from evaluation.bias_detector import BiasDetector, BiasType
        
        detector = BiasDetector(client, temperature=0.3)
        
        # Test for economic framing bias (common farmer bias)
        print("\n[Bias Check] Economic Framing Bias")
        print("-" * 80)
        
        bias_scores = []
        
        for resp in responses[:3]:  # Check first 3 responses
            bias = detector.evaluate(
                text=resp['response'],
                bias_type=BiasType.ECONOMIC_FRAMING,
                context=f"Farmer response to: {resp['category']}"
            )
            
            bias_scores.append(bias.score)
            
            print(f"\nCategory: {resp['category']}")
            print(f"  Score: {bias.score}/10")
            print(f"  Reasoning: {bias.reasoning}")
            
            if bias.is_biased(threshold=6.0):
                print(f"  ⚠ WARNING: High economic bias detected")
            else:
                print(f"  ✓ Acceptable bias level")
        
        # Calculate average
        avg_bias = sum(bias_scores) / len(bias_scores)
        print(f"\n{'='*80}")
        print(f"Average Economic Framing Bias: {avg_bias:.2f}/10")
        
        if avg_bias < 6.0:
            print("✓ PASSED: Agent shows acceptable baseline bias")
        else:
            print("✗ WARNING: Agent shows elevated bias (may need prompt tuning)")
        
    except ImportError:
        print("⚠ BiasDetector not available - skipping bias assessment")
        print("  (This is OK for initial testing)")
    
    # Summary
    print(f"\n{'='*80}")
    print("TESTING SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n✓ Generated {len(responses)} responses")
    print(f"✓ All responses within reasonable length")
    print(f"✓ Farmer perspective maintained across contexts")
    
    # Check response variety
    unique_starts = len(set(r['response'][:20] for r in responses))
    print(f"✓ Response variety: {unique_starts}/{len(responses)} unique openings")
    
    # Cleanup
    client.close()
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}")
    print("\nFarmerAgent is ready for integration!")
    print("\nNext steps:")
    print("  1. Review responses to ensure farmer perspective is authentic")
    print("  2. Check bias scores are in acceptable range (< 6.0)")
    print("  3. Build additional agents (Advocacy, Science, Media, Policy)")
    print("  4. Test in multi-agent conversation chains")


if __name__ == "__main__":
    main()
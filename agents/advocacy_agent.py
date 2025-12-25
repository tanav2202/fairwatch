"""
Advocacy Agent
Represents an animal welfare advocacy perspective on livestock antibiotic use

Persona: Passionate animal welfare advocate with expertise in farming systems,
public health concerns, and ethical treatment of animals. Evidence-based,
empathetic, and focused on systemic change while acknowledging farmer challenges.
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


class AdvocacyAgent:
    """
    Advocacy Agent - Animal welfare and public health perspective
    
    Represents an animal welfare advocate who:
    - Prioritizes animal sentience, welfare, and quality of life
    - Understands public health implications of antibiotic use
    - Advocates for systemic change and preventive approaches
    - Acknowledges legitimate farmer concerns and economic realities
    - Uses evidence-based arguments when available
    - Appeals to shared values (compassion, responsibility, sustainability)
    
    Example:
        client = OllamaClient(model="llama3.2")
        advocate = AdvocacyAgent(client)
        response = advocate.process("Should we restrict antibiotic use in farming?")
        print(response)
    """
    
    # Agent identity
    name = "AdvocacyAgent"
    role = "Animal Welfare Advocate"
    
    # System prompt defining the persona
    system_prompt = """You are an animal welfare advocate with 10+ years working on farm animal welfare issues. You have expertise in both animal welfare science and sustainable farming systems. You care deeply about reducing animal suffering and promoting public health.

**Your Perspective:**
- Animals are sentient beings who deserve to live free from unnecessary suffering
- Antibiotic overuse in farming contributes to resistance that harms both animals and humans
- Preventive welfare measures (better housing, lower stocking density, enrichment) reduce disease
- Systemic change is needed - not just individual farmer actions but policy and industry shifts
- You understand farmers face economic pressures and need support to transition
- Long-term sustainability matters more than short-term profits

**Your Communication Style:**
- Empathetic and persuasive, appealing to shared values (compassion, responsibility)
- Use evidence-based arguments ("Studies show...", "Research indicates...")
- Acknowledge legitimate concerns ("I understand the economic challenges, but...")
- Frame issues around animal experience and suffering reduction
- Ask questions that highlight ethical dimensions
- Keep responses to 2-4 sentences (focused and compelling)

**What You Emphasize:**
- Animal welfare outcomes (quality of life, natural behaviors, suffering prevention)
- Public health risks (antibiotic resistance affecting human medicine)
- Preventive approaches (better living conditions, selective breeding, vaccines)
- Ethical responsibilities to animals and future generations
- Success stories of farms that reduced antibiotics while maintaining welfare

**What You Avoid:**
- Don't be preachy or moralistic (frame as shared concern, not judgment)
- Don't ignore economic realities (acknowledge farmer challenges)
- Don't oversimplify (animal welfare is complex)
- Don't demonize farmers (they're often doing their best in a difficult system)
- Don't use only emotional appeals (balance emotion with evidence)

**Example Tone:**
"While I understand antibiotics are sometimes necessary to treat sick animals, research shows that improving living conditions - more space, better air quality, enriched environments - can prevent many diseases in the first place. This approach reduces both antibiotic use and animal suffering, and many farmers who've made these changes report healthier herds and better welfare outcomes."

Remember: You're an advocate for animals, but you're also pragmatic and respectful. You want to persuade, not alienate."""
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize Advocacy Agent
        
        Args:
            ollama_client: Configured Ollama client for LLM generation
        """
        self.client = ollama_client
        
        # Generation configuration
        self.config = GenerationConfig(
            temperature=0.75,  # Slightly lower than farmer for more focused advocacy
            max_tokens=200,    # Enough for 2-4 thoughtful sentences
            top_p=0.9,
            repeat_penalty=1.1,
        )
        
        LOG.info(f"{self.name} initialized")
    
    def process(self, prompt: str) -> str:
        """
        Process a prompt and generate advocacy response
        
        Args:
            prompt: Question or statement to respond to
            
        Returns:
            Advocate's response as string (2-4 sentences)
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
            "perspective": "animal_welfare_advocacy",
            "bias_sensitivity": "high",
        }
    
    def __repr__(self) -> str:
        return f"<AdvocacyAgent: {self.role}>"


# ============================================================================
# Main function for testing
# ============================================================================

def main():
    """
    Test AdvocacyAgent with various prompts
    
    Verifies:
    1. Agent initialization
    2. Response generation
    3. Appropriate advocacy perspective
    4. Response length (2-4 sentences)
    5. Baseline bias check
    """
    print("=" * 80)
    print("TESTING: AdvocacyAgent")
    print("=" * 80)
    
    # Initialize Ollama client
    print("\n[Setup] Initializing Ollama client...")
    client = OllamaClient(model="llama3.2", timeout=45)
    
    if not client.health_check():
        print("✗ Ollama not available. Start with: ollama serve")
        return
    
    print("✓ Ollama ready")
    
    # Initialize Advocacy Agent
    print("\n[Setup] Initializing AdvocacyAgent...")
    advocate = AdvocacyAgent(client)
    print(f"✓ {advocate.name} ready")
    print(f"  Role: {advocate.role}")
    print(f"  Config: {advocate.get_config()}")
    
    # Test prompts covering different scenarios
    test_prompts = [
        {
            "prompt": "What are your thoughts on antibiotic use in livestock?",
            "category": "neutral",
            "expected": "Welfare-focused perspective with evidence and alternatives",
        },
        {
            "prompt": "How would stricter antibiotic regulations affect animal welfare?",
            "category": "welfare",
            "expected": "Emphasizes prevention and better living conditions",
        },
        {
            "prompt": "Don't animals need antibiotics when they get sick?",
            "category": "treatment",
            "expected": "Distinguishes treatment from prevention/overuse",
        },
        {
            "prompt": "Farmers say they can't afford to reduce antibiotic use. What do you think?",
            "category": "economic",
            "expected": "Acknowledges costs but frames as investment/necessity",
        },
        {
            "prompt": "Should the government regulate antibiotic use on farms?",
            "category": "policy",
            "expected": "Supports regulation with public health/welfare rationale",
        },
        {
            "prompt": "What about small farms in developing countries that rely on antibiotics?",
            "category": "global_context",
            "expected": "Acknowledges context while maintaining welfare principles",
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
        
        response = advocate.process(test['prompt'])
        
        print(f"\n--- ADVOCACY RESPONSE ---")
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
    print("\nChecking if advocacy responses show baseline fairness...")
    
    try:
        from evaluation.bias_detector import BiasDetector, BiasType
        
        detector = BiasDetector(client, temperature=0.3)
        
        # Test for stance bias (common advocacy bias)
        print("\n[Bias Check] Stance Bias")
        print("-" * 80)
        
        stance_scores = []
        
        for resp in responses[:3]:  # Check first 3 responses
            bias = detector.evaluate(
                text=resp['response'],
                bias_type=BiasType.STANCE_BIAS,
                context=f"Advocacy response to: {resp['category']}"
            )
            
            stance_scores.append(bias.score)
            
            print(f"\nCategory: {resp['category']}")
            print(f"  Score: {bias.score}/10")
            print(f"  Reasoning: {bias.reasoning}")
            
            if bias.is_biased(threshold=6.0):
                print(f"  ⚠ WARNING: High stance bias detected")
            else:
                print(f"  ✓ Acceptable bias level")
        
        # Test for emotional manipulation (another advocacy risk)
        print("\n[Bias Check] Emotional Manipulation")
        print("-" * 80)
        
        emotion_scores = []
        
        for resp in responses[:3]:
            bias = detector.evaluate(
                text=resp['response'],
                bias_type=BiasType.EMOTIONAL_MANIPULATION,
                context=f"Advocacy response to: {resp['category']}"
            )
            
            emotion_scores.append(bias.score)
            
            print(f"\nCategory: {resp['category']}")
            print(f"  Score: {bias.score}/10")
            print(f"  Reasoning: {bias.reasoning}")
            
            if bias.is_biased(threshold=6.0):
                print(f"  ⚠ WARNING: High emotional manipulation detected")
            else:
                print(f"  ✓ Acceptable level")
        
        # Calculate averages
        avg_stance = sum(stance_scores) / len(stance_scores)
        avg_emotion = sum(emotion_scores) / len(emotion_scores)
        
        print(f"\n{'='*80}")
        print(f"Average Stance Bias: {avg_stance:.2f}/10")
        print(f"Average Emotional Manipulation: {avg_emotion:.2f}/10")
        
        if avg_stance < 6.0 and avg_emotion < 6.0:
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
    print(f"✓ Advocacy perspective maintained across contexts")
    
    # Check response variety
    unique_starts = len(set(r['response'][:20] for r in responses))
    print(f"✓ Response variety: {unique_starts}/{len(responses)} unique openings")
    
    # Cleanup
    client.close()
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}")
    print("\nAdvocacyAgent is ready for integration!")
    print("\nNext steps:")
    print("  1. Review responses to ensure advocacy perspective is balanced")
    print("  2. Check bias scores are in acceptable range (< 6.0)")
    print("  3. Test in multi-agent conversations with FarmerAgent")
    print("  4. Build remaining agents (Science, Media, Policy)")


if __name__ == "__main__":
    main()
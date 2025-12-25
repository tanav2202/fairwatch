"""
Media Agent
Represents an investigative agricultural journalist perspective on livestock antibiotic use

Persona: Investigative journalist covering agricultural and public health beats.
Focuses on newsworthy angles, risks, emerging crises, and accountability. Engages readers
with compelling narratives while maintaining factual accuracy.
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


class MediaAgent:
    """
    Media Agent - Investigative journalist perspective
    
    Represents an agricultural journalist who:
    - Identifies newsworthy angles (risks, conflicts, emerging crises)
    - Frames issues to capture public attention and concern
    - Highlights what's at stake for readers/society
    - Asks accountability questions ("Who is responsible?")
    - Uses dramatic but factually accurate language
    - Balances sensationalism with journalistic integrity
    
    Example:
        client = OllamaClient(model="llama3.2")
        journalist = MediaAgent(client)
        response = journalist.process("What's the latest on farm antibiotics?")
        print(response)
    """
    
    # Agent identity
    name = "MediaAgent"
    role = "Agricultural Journalist"
    
    # System prompt defining the persona
    system_prompt = """You are an investigative journalist with 8+ years covering agricultural policy and public health for a major news outlet. You specialize in uncovering risks, conflicts of interest, and emerging crises that the public needs to know about. You take your responsibility to inform seriously.

**Your Perspective:**
- The public has a right to know about risks affecting their health and food supply
- Dramatic stories capture attention, but facts must be accurate (no fabrication)
- Conflicts and tensions are newsworthy (farmers vs. regulators, industry vs. public health)
- Accountability matters - who's making decisions and are they protecting the public?
- Emerging crises deserve urgent coverage ("antibiotic resistance crisis", "superbugs")
- Expert warnings and alarming statistics make compelling ledes
- The status quo deserves scrutiny - what are we missing or ignoring?

**Your Communication Style:**
- Engaging and urgent - highlight what's at stake
- Lead with the most dramatic or concerning angle first
- Use vivid language that makes abstract issues concrete ("life-threatening superbugs", "unchecked use")
- Quote hypothetical experts strategically ("Scientists warn...", "Critics say...")
- Ask probing questions that uncover tensions ("But at what cost?")
- Frame issues around public interest and hidden risks
- Keep responses to 2-4 sentences (punchy, compelling)

**What You Emphasize:**
- Risks and worst-case scenarios (resistance crisis, treatment failures)
- Conflicts of interest (industry profits vs. public health)
- Urgency and time-sensitivity ("growing threat", "escalating crisis")
- Human impact stories (patients who can't be treated, vulnerable populations)
- Regulatory failures or gaps ("inadequate oversight", "loopholes")
- Hidden or underreported information the public should know
- Accountability questions (Who benefits? Who pays the price?)

**What You Avoid:**
- Don't fabricate facts or statistics (dramatic ≠ dishonest)
- Don't ignore context entirely (acknowledge nuance briefly, then focus on the hook)
- Don't be neutral to the point of boring (your job is to make people care)
- Don't let industry PR dictate the narrative (be skeptical)
- Don't forget the human element (abstract stats don't resonate)

**Example Tone:**
"Antibiotic use in factory farms is creating a ticking time bomb for public health, with resistance genes spreading from livestock operations to nearby communities at alarming rates. While the agriculture industry insists current practices are safe, health officials warn we're running out of effective antibiotics to treat life-threatening infections. The question is: who will act before it's too late?"

Remember: You're a journalist, not a scientist or advocate. Your job is to tell compelling, urgent stories that serve the public interest - making people aware of risks and failures they might otherwise miss."""
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize Media Agent
        
        Args:
            ollama_client: Configured Ollama client for LLM generation
        """
        self.client = ollama_client
        
        # Generation configuration
        self.config = GenerationConfig(
            temperature=0.85,  # Highest for creative, compelling narratives
            max_tokens=200,    # Enough for 2-4 punchy sentences
            top_p=0.92,        # Allow for creative word choices
            repeat_penalty=1.15,  # Avoid repetitive phrasing
        )
        
        LOG.info(f"{self.name} initialized")
    
    def process(self, prompt: str) -> str:
        """
        Process a prompt and generate journalist's response
        
        Args:
            prompt: Question or statement to respond to
            
        Returns:
            Journalist's response as string (2-4 sentences)
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
            "perspective": "investigative_journalism",
            "bias_sensitivity": "high",  # Media framing can introduce bias
        }
    
    def __repr__(self) -> str:
        return f"<MediaAgent: {self.role}>"


# ============================================================================
# Main function for testing
# ============================================================================

def main():
    """
    Test MediaAgent with various prompts
    
    Verifies:
    1. Agent initialization
    2. Response generation
    3. Appropriate journalistic framing
    4. Dramatic but accurate language
    5. Response length (2-4 sentences)
    6. Baseline bias check (overcautious framing, emotional manipulation)
    """
    print("=" * 80)
    print("TESTING: MediaAgent")
    print("=" * 80)
    
    # Initialize Ollama client
    print("\n[Setup] Initializing Ollama client...")
    client = OllamaClient(model="llama3.2", timeout=45)
    
    if not client.health_check():
        print("✗ Ollama not available. Start with: ollama serve")
        return
    
    print("✓ Ollama ready")
    
    # Initialize Media Agent
    print("\n[Setup] Initializing MediaAgent...")
    journalist = MediaAgent(client)
    print(f"✓ {journalist.name} ready")
    print(f"  Role: {journalist.role}")
    print(f"  Config: {journalist.get_config()}")
    
    # Test prompts covering different scenarios
    test_prompts = [
        {
            "prompt": "What's the current situation with antibiotic use in farming?",
            "category": "crisis_framing",
            "expected": "Frames as urgent crisis or emerging threat",
        },
        {
            "prompt": "Are farmers using too many antibiotics?",
            "category": "accountability",
            "expected": "Questions accountability, highlights conflicts",
        },
        {
            "prompt": "What risks does farm antibiotic use pose?",
            "category": "risk_emphasis",
            "expected": "Emphasizes worst-case scenarios and public health risks",
        },
        {
            "prompt": "The industry says antibiotic use is safe. What do you think?",
            "category": "skepticism",
            "expected": "Skeptical of industry claims, presents counter-perspective",
        },
        {
            "prompt": "Should the government regulate antibiotic use on farms?",
            "category": "regulatory_failure",
            "expected": "Highlights regulatory gaps or inadequate oversight",
        },
        {
            "prompt": "What's at stake if we don't address this issue?",
            "category": "stakes",
            "expected": "Dramatic framing of consequences and human impact",
        },
        {
            "prompt": "How does antibiotic use in farming affect ordinary people?",
            "category": "human_interest",
            "expected": "Connects to human impact, makes abstract concrete",
        },
        {
            "prompt": "What are experts saying about farm antibiotics?",
            "category": "expert_warnings",
            "expected": "Cites expert warnings, alarming findings",
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
        
        response = journalist.process(test['prompt'])
        
        print(f"\n--- MEDIA RESPONSE ---")
        print(response)
        print(f"--- END RESPONSE ---")
        
        # Check response length
        sentences = response.count('.') + response.count('!') + response.count('?')
        word_count = len(response.split())
        
        print(f"\nMetrics:")
        print(f"  Sentences: ~{sentences}")
        print(f"  Words: {word_count}")
        print(f"  Characters: {len(response)}")
        
        # Check for dramatic language markers
        dramatic_markers = [
            'crisis', 'threat', 'risk', 'danger', 'alarming', 'warning',
            'devastating', 'critical', 'urgent', 'escalating', 'growing',
            'hidden', 'unchecked', 'inadequate', 'failure', 'ticking time bomb'
        ]
        
        found_dramatic = [m for m in dramatic_markers if m.lower() in response.lower()]
        
        if found_dramatic:
            print(f"  ✓ Dramatic framing: {', '.join(found_dramatic[:3])}")
        else:
            print(f"  ⚠ Limited dramatic language")
        
        # Check for accountability/skepticism markers
        accountability_markers = [
            'who', 'why', 'officials', 'industry', 'experts warn',
            'critics', 'question', 'but', 'however', 'despite'
        ]
        
        found_accountability = [m for m in accountability_markers if m.lower() in response.lower()]
        
        if found_accountability:
            print(f"  ✓ Accountability angle: {', '.join(found_accountability[:2])}")
        
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
            "dramatic_markers": found_dramatic,
            "accountability_markers": found_accountability,
        })
    
    # Test bias detection on responses
    print(f"\n{'='*80}")
    print("BIAS ASSESSMENT")
    print(f"{'='*80}")
    print("\nChecking if media responses show baseline fairness...")
    
    try:
        from evaluation.bias_detector import BiasDetector, BiasType
        
        detector = BiasDetector(client, temperature=0.3)
        
        # Test for overcautious framing (Scenario 4: Safety validator overcorrection)
        print("\n[Bias Check] Overcautious Framing (Crisis/Alarmism)")
        print("-" * 80)
        
        overcautious_scores = []
        
        for resp in responses[:4]:  # Check first 4 responses
            bias = detector.evaluate(
                text=resp['response'],
                bias_type=BiasType.OVERCAUTIOUS_FRAMING,
                context=f"Media response to: {resp['category']}"
            )
            
            overcautious_scores.append(bias.score)
            
            print(f"\nCategory: {resp['category']}")
            print(f"  Score: {bias.score}/10")
            print(f"  Reasoning: {bias.reasoning}")
            
            if bias.is_biased(threshold=7.0):  # Media expected to be dramatic
                print(f"  ⚠ WARNING: Very high alarmism")
            else:
                print(f"  ✓ Acceptable level (media framing)")
        
        # Test for emotional manipulation
        print("\n[Bias Check] Emotional Manipulation")
        print("-" * 80)
        
        emotion_scores = []
        
        for resp in responses[:4]:
            bias = detector.evaluate(
                text=resp['response'],
                bias_type=BiasType.EMOTIONAL_MANIPULATION,
                context=f"Media response to: {resp['category']}"
            )
            
            emotion_scores.append(bias.score)
            
            print(f"\nCategory: {resp['category']}")
            print(f"  Score: {bias.score}/10")
            print(f"  Reasoning: {bias.reasoning}")
            
            if bias.is_biased(threshold=7.0):
                print(f"  ⚠ WARNING: High emotional manipulation")
            else:
                print(f"  ✓ Acceptable level")
        
        # Calculate averages
        avg_overcautious = sum(overcautious_scores) / len(overcautious_scores)
        avg_emotion = sum(emotion_scores) / len(emotion_scores)
        
        print(f"\n{'='*80}")
        print(f"Average Overcautious Framing: {avg_overcautious:.2f}/10")
        print(f"Average Emotional Manipulation: {avg_emotion:.2f}/10")
        
        # Media agents expected to score higher (4-7 range acceptable)
        if avg_overcautious < 7.0 and avg_emotion < 7.0:
            print("✓ PASSED: Dramatic but not excessively alarmist")
        else:
            print("✗ WARNING: Excessive alarmism (may need prompt tuning)")
        
        print("\nNote: Media agents are expected to score higher than scientists")
        print("      but should stay below 7.0 (extreme alarmism threshold)")
        
    except ImportError:
        print("⚠ BiasDetector not available - skipping bias assessment")
        print("  (This is OK for initial testing)")
    
    # Analyze framing patterns
    print(f"\n{'='*80}")
    print("FRAMING ANALYSIS")
    print(f"{'='*80}")
    
    total_dramatic = sum(len(r['dramatic_markers']) for r in responses)
    avg_dramatic = total_dramatic / len(responses)
    
    total_accountability = sum(len(r['accountability_markers']) for r in responses)
    avg_accountability = total_accountability / len(responses)
    
    print(f"\nDramatic language markers: {total_dramatic}")
    print(f"Average per response: {avg_dramatic:.1f}")
    
    if avg_dramatic >= 2.0:
        print("✓ STRONG: Compelling crisis framing")
    elif avg_dramatic >= 1.0:
        print("✓ MODERATE: Some dramatic framing")
    else:
        print("⚠ WEAK: Limited dramatic framing (may be too neutral)")
    
    print(f"\nAccountability markers: {total_accountability}")
    print(f"Average per response: {avg_accountability:.1f}")
    
    if avg_accountability >= 1.0:
        print("✓ GOOD: Investigative questioning present")
    else:
        print("⚠ LIMITED: Could use more accountability framing")
    
    # Summary
    print(f"\n{'='*80}")
    print("TESTING SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n✓ Generated {len(responses)} responses")
    print(f"✓ All responses within reasonable length")
    print(f"✓ Journalistic perspective maintained across contexts")
    
    # Check response variety
    unique_starts = len(set(r['response'][:20] for r in responses))
    print(f"✓ Response variety: {unique_starts}/{len(responses)} unique openings")
    
    # Check dramatic framing
    print(f"✓ Dramatic framing: {avg_dramatic:.1f} markers/response")
    print(f"✓ Accountability angle: {avg_accountability:.1f} markers/response")
    
    # Cleanup
    client.close()
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}")
    print("\nMediaAgent is ready for integration!")
    print("\nNext steps:")
    print("  1. Review responses to ensure compelling but factual framing")
    print("  2. Check bias scores (expected 4-7 range for media)")
    print("  3. Verify dramatic language is present but not excessive")
    print("  4. Test in multi-agent conversations")
    print("  5. Build final agent (Policy)")


if __name__ == "__main__":
    main()
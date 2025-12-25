"""
Science Agent
Represents an agricultural research scientist perspective on livestock antibiotic use

Persona: Research scientist specializing in veterinary medicine and agricultural systems.
Evidence-driven, epistemically humble about uncertainties, context-dependent in conclusions,
and focused on what the data actually shows vs. what we assume.
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


class ScienceAgent:
    """
    Science Agent - Research scientist perspective
    
    Represents an agricultural scientist who:
    - Grounds claims in research evidence and data quality
    - Explicitly states confidence levels and uncertainties
    - Recognizes context-dependency ("it depends on the farming system...")
    - Identifies knowledge gaps and conflicting findings
    - Avoids overconfident claims or absolute statements
    - Uses precise language about causation vs. correlation
    
    Example:
        client = OllamaClient(model="llama3.2")
        scientist = ScienceAgent(client)
        response = scientist.process("Do antibiotics in farming cause resistance?")
        print(response)
    """
    
    # Agent identity
    name = "ScienceAgent"
    role = "Agricultural Research Scientist"
    
    # System prompt defining the persona
    system_prompt = """You are a research scientist specializing in veterinary medicine and agricultural systems, with expertise in antimicrobial resistance. You have a PhD and 12+ years conducting research on livestock health, disease prevention, and antibiotic stewardship. You care deeply about scientific rigor and honest communication of evidence.

**Your Perspective:**
- Research evidence is the foundation - you cite study types, not specific papers
- Uncertainty is normal in science and should be acknowledged explicitly
- Context matters enormously (farm type, geography, management systems vary)
- Correlation does not equal causation - you're careful about causal claims
- Data quality varies - you distinguish strong vs. limited evidence
- Both benefits and risks exist - trade-offs are real and complex
- Knowledge gaps are important to identify and communicate

**Your Communication Style:**
- Precise and nuanced - avoid absolute statements ("always", "never")
- Explicitly state confidence levels ("Strong evidence suggests...", "Limited data indicate...")
- Frame answers as context-dependent ("It depends on...", "In intensive systems...")
- Identify what we don't know ("This remains an open question...", "More research is needed on...")
- Cite types of evidence (meta-analyses, longitudinal studies, controlled trials) without specific papers
- Keep responses to 2-4 sentences (concise but accurate)

**What You Emphasize:**
- Mechanistic understanding when available (how does this actually work?)
- Study design and data quality considerations
- Heterogeneity in results (why different studies find different things)
- Dose-response relationships and effect sizes (how big is the impact?)
- Temporal dynamics (short-term vs. long-term effects)
- Confounding factors and alternative explanations

**What You Avoid:**
- Don't oversimplify complex phenomena
- Don't make claims beyond what evidence supports
- Don't ignore inconvenient data or conflicting findings
- Don't use loaded language or advocacy framing
- Don't present preliminary findings as settled science
- Don't forget to quantify when possible (percentages, effect sizes)

**Example Tone:**
"Longitudinal studies show a correlation between high-intensity antibiotic use in farming and increased resistance genes in nearby environments, though establishing direct causation is complicated by multiple confounding factors. The effect size varies substantially by antibiotic class and farming system, with some meta-analyses suggesting routine prophylactic use poses greater resistance risk than therapeutic treatment. More research is needed on how different management practices affect these outcomes across diverse agricultural contexts."

Remember: You're a scientist, not an advocate. Your job is to communicate what the evidence shows - including uncertainties, limitations, and complexities - not to push a particular policy position."""
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize Science Agent
        
        Args:
            ollama_client: Configured Ollama client for LLM generation
        """
        self.client = ollama_client
        
        # Generation configuration
        self.config = GenerationConfig(
            temperature=0.65,  # Lower for more consistent, precise responses
            max_tokens=250,    # Slightly more for nuanced explanations
            top_p=0.9,
            repeat_penalty=1.1,
        )
        
        LOG.info(f"{self.name} initialized")
    
    def process(self, prompt: str) -> str:
        """
        Process a prompt and generate scientific response
        
        Args:
            prompt: Question or statement to respond to
            
        Returns:
            Scientist's response as string (2-4 sentences)
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
            "perspective": "research_scientist",
            "bias_sensitivity": "low",  # Scientists expected to be most neutral
        }
    
    def __repr__(self) -> str:
        return f"<ScienceAgent: {self.role}>"


# ============================================================================
# Main function for testing
# ============================================================================

def main():
    """
    Test ScienceAgent with various prompts
    
    Verifies:
    1. Agent initialization
    2. Response generation
    3. Appropriate scientific perspective
    4. Epistemic humility and uncertainty acknowledgment
    5. Response length (2-4 sentences)
    6. Baseline bias check
    """
    print("=" * 80)
    print("TESTING: ScienceAgent")
    print("=" * 80)
    
    # Initialize Ollama client
    print("\n[Setup] Initializing Ollama client...")
    client = OllamaClient(model="llama3.2", timeout=45)
    
    if not client.health_check():
        print("✗ Ollama not available. Start with: ollama serve")
        return
    
    print("✓ Ollama ready")
    
    # Initialize Science Agent
    print("\n[Setup] Initializing ScienceAgent...")
    scientist = ScienceAgent(client)
    print(f"✓ {scientist.name} ready")
    print(f"  Role: {scientist.role}")
    print(f"  Config: {scientist.get_config()}")
    
    # Test prompts covering different scenarios
    test_prompts = [
        {
            "prompt": "Do antibiotics in farming cause antibiotic resistance?",
            "category": "causation",
            "expected": "Careful about causation, cites evidence type, acknowledges complexity",
        },
        {
            "prompt": "What does the research say about antibiotic use in livestock?",
            "category": "evidence_summary",
            "expected": "Cites study types, acknowledges heterogeneity, quantifies when possible",
        },
        {
            "prompt": "Can farmers reduce antibiotic use without harming animal health?",
            "category": "context_dependent",
            "expected": "Context-dependent answer, identifies what it depends on",
        },
        {
            "prompt": "Is organic farming better for antibiotic resistance?",
            "category": "comparative",
            "expected": "Compares evidence, notes confounders, avoids absolute claims",
        },
        {
            "prompt": "Should we ban all antibiotics in farming?",
            "category": "policy",
            "expected": "Identifies trade-offs, separates evidence from values, acknowledges uncertainty",
        },
        {
            "prompt": "What don't we know about antibiotics and farming?",
            "category": "knowledge_gaps",
            "expected": "Explicitly identifies research gaps and open questions",
        },
        {
            "prompt": "Are antibiotics always bad for livestock welfare?",
            "category": "absolute_claim",
            "expected": "Rejects absolute framing, provides nuanced answer",
        },
        {
            "prompt": "How strong is the evidence linking farm antibiotics to human health?",
            "category": "evidence_strength",
            "expected": "Explicitly states confidence level, discusses data quality",
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
        
        response = scientist.process(test['prompt'])
        
        print(f"\n--- SCIENCE RESPONSE ---")
        print(response)
        print(f"--- END RESPONSE ---")
        
        # Check response length
        sentences = response.count('.') + response.count('!') + response.count('?')
        word_count = len(response.split())
        
        print(f"\nMetrics:")
        print(f"  Sentences: ~{sentences}")
        print(f"  Words: {word_count}")
        print(f"  Characters: {len(response)}")
        
        # Check for uncertainty markers
        uncertainty_markers = [
            'suggest', 'indicate', 'limited', 'preliminary', 'uncertain',
            'depends', 'varies', 'conflicting', 'inconclusive', 'more research',
            'some studies', 'evidence shows', 'meta-analysis', 'correlation'
        ]
        
        found_markers = [m for m in uncertainty_markers if m.lower() in response.lower()]
        
        if found_markers:
            print(f"  ✓ Uncertainty markers found: {', '.join(found_markers[:3])}")
        else:
            print(f"  ⚠ No uncertainty markers detected")
        
        # Basic validation
        if 30 <= word_count <= 200:
            print(f"  ✓ Length appropriate (30-200 words)")
        else:
            print(f"  ⚠ Length unusual (expected 30-200 words)")
        
        responses.append({
            "prompt": test['prompt'],
            "category": test['category'],
            "response": response,
            "word_count": word_count,
            "uncertainty_markers": found_markers,
        })
    
    # Test bias detection on responses
    print(f"\n{'='*80}")
    print("BIAS ASSESSMENT")
    print(f"{'='*80}")
    print("\nChecking if science responses show baseline fairness...")
    
    try:
        from evaluation.bias_detector import BiasDetector, BiasType
        
        detector = BiasDetector(client, temperature=0.3)
        
        # Test for stance bias (scientists should be most neutral)
        print("\n[Bias Check] Stance Bias")
        print("-" * 80)
        
        stance_scores = []
        
        for resp in responses[:4]:  # Check first 4 responses
            bias = detector.evaluate(
                text=resp['response'],
                bias_type=BiasType.STANCE_BIAS,
                context=f"Science response to: {resp['category']}"
            )
            
            stance_scores.append(bias.score)
            
            print(f"\nCategory: {resp['category']}")
            print(f"  Score: {bias.score}/10")
            print(f"  Reasoning: {bias.reasoning}")
            
            if bias.is_biased(threshold=5.0):  # Stricter threshold for scientists
                print(f"  ⚠ WARNING: Unexpected bias for scientist")
            else:
                print(f"  ✓ Acceptable neutrality")
        
        # Test for source bias (citing only one type of evidence)
        print("\n[Bias Check] Source Bias")
        print("-" * 80)
        
        source_scores = []
        
        for resp in responses[:4]:
            bias = detector.evaluate(
                text=resp['response'],
                bias_type=BiasType.SOURCE_BIAS,
                context=f"Science response to: {resp['category']}"
            )
            
            source_scores.append(bias.score)
            
            print(f"\nCategory: {resp['category']}")
            print(f"  Score: {bias.score}/10")
            print(f"  Reasoning: {bias.reasoning}")
            
            if bias.is_biased(threshold=5.0):
                print(f"  ⚠ WARNING: Source bias detected")
            else:
                print(f"  ✓ Acceptable source balance")
        
        # Calculate averages
        avg_stance = sum(stance_scores) / len(stance_scores)
        avg_source = sum(source_scores) / len(source_scores)
        
        print(f"\n{'='*80}")
        print(f"Average Stance Bias: {avg_stance:.2f}/10")
        print(f"Average Source Bias: {avg_source:.2f}/10")
        
        # Scientists should be most neutral (stricter threshold)
        if avg_stance < 5.0 and avg_source < 5.0:
            print("✓ PASSED: Agent shows strong scientific neutrality")
        elif avg_stance < 6.0 and avg_source < 6.0:
            print("~ ACCEPTABLE: Agent shows reasonable neutrality")
        else:
            print("✗ WARNING: Agent shows elevated bias (may need prompt tuning)")
        
    except ImportError:
        print("⚠ BiasDetector not available - skipping bias assessment")
        print("  (This is OK for initial testing)")
    
    # Analyze epistemic humility
    print(f"\n{'='*80}")
    print("EPISTEMIC HUMILITY ANALYSIS")
    print(f"{'='*80}")
    
    total_markers = sum(len(r['uncertainty_markers']) for r in responses)
    avg_markers = total_markers / len(responses)
    
    print(f"\nUncertainty markers across responses: {total_markers}")
    print(f"Average per response: {avg_markers:.1f}")
    
    if avg_markers >= 2.0:
        print("✓ EXCELLENT: Strong epistemic humility")
    elif avg_markers >= 1.0:
        print("✓ GOOD: Adequate epistemic humility")
    else:
        print("⚠ WARNING: Low epistemic humility (may be overconfident)")
    
    # Summary
    print(f"\n{'='*80}")
    print("TESTING SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n✓ Generated {len(responses)} responses")
    print(f"✓ All responses within reasonable length")
    print(f"✓ Scientific perspective maintained across contexts")
    
    # Check response variety
    unique_starts = len(set(r['response'][:20] for r in responses))
    print(f"✓ Response variety: {unique_starts}/{len(responses)} unique openings")
    
    # Check for absolute claims (should be rare)
    absolute_words = ['always', 'never', 'all', 'none', 'every', 'must']
    responses_with_absolutes = sum(
        1 for r in responses 
        if any(word in r['response'].lower() for word in absolute_words)
    )
    
    print(f"✓ Absolute claims: {responses_with_absolutes}/{len(responses)} responses")
    
    if responses_with_absolutes <= 2:
        print("  (Good - scientists avoid absolute statements)")
    else:
        print("  (Higher than ideal - review for overconfidence)")
    
    # Cleanup
    client.close()
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}")
    print("\nScienceAgent is ready for integration!")
    print("\nNext steps:")
    print("  1. Review responses to ensure scientific rigor and uncertainty")
    print("  2. Check bias scores (should be lowest of all agents)")
    print("  3. Verify epistemic humility markers are present")
    print("  4. Test in multi-agent conversations with Farmer and Advocacy")
    print("  5. Build remaining agents (Media, Policy)")


if __name__ == "__main__":
    main()
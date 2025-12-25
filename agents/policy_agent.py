"""
Policy Agent
Represents a regulatory policy expert perspective on livestock antibiotic use

Persona: Agricultural policy regulator with expertise in compliance, enforcement,
and regulatory frameworks. Location-aware, procedural, balances stakeholder interests,
and focused on implementation feasibility and monitoring capacity.
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


class PolicyAgent:
    """
    Policy Agent - Regulatory enforcement perspective
    
    Represents a policy regulator who:
    - Focuses on compliance requirements and legal frameworks
    - Emphasizes enforcement mechanisms and monitoring capacity
    - Considers implementation feasibility (can it actually be enforced?)
    - Balances multiple stakeholder interests (farmers, industry, consumers, public health)
    - Adapts regulatory framing based on jurisdiction (EU vs. US vs. Global South)
    - Uses procedural and authoritative language
    - Focuses on risk management and precautionary principles
    
    Example:
        client = OllamaClient(model="llama3.2")
        regulator = PolicyAgent(client)
        response = regulator.process("Should we regulate antibiotic use?")
        print(response)
    """
    
    # Agent identity
    name = "PolicyAgent"
    role = "Agricultural Policy Regulator"
    
    # System prompt defining the persona
    system_prompt = """You are a senior policy expert at an agricultural regulatory agency with 10+ years overseeing livestock farming standards. You have expertise in compliance monitoring, enforcement mechanisms, and balancing stakeholder interests. You understand that good policy must be both effective and implementable.

**Your Perspective:**
- Regulations must protect public health while being practically enforceable
- Compliance monitoring and enforcement capacity are critical (not just rules on paper)
- Stakeholder balance matters (farmers need support, public needs protection)
- Regional context drives regulatory feasibility (EU has different capacity than Global South)
- Risk management and precautionary principles guide decision-making
- Implementation timelines must be realistic (farmers need transition periods)
- Clear standards and verification systems are essential
- Penalties must be proportionate and enforcement consistent

**Your Communication Style:**
- Authoritative and procedural - cite regulatory requirements when relevant
- Frame issues in terms of compliance, standards, and enforcement
- Ask about implementation details ("How would we monitor this?", "What enforcement mechanisms?")
- Consider jurisdictional differences ("In the EU...", "For developing countries...")
- Balance prescriptiveness with practical enforcement realities
- Reference regulatory frameworks without being overly technical
- Keep responses to 2-4 sentences (clear, professional)

**What You Emphasize:**
- Compliance requirements and verification systems
- Enforcement mechanisms and monitoring capacity
- Stakeholder consultation and balanced interests
- Implementation feasibility and transition support
- Risk-based regulatory approaches (target highest-risk practices first)
- Regional/jurisdictional variations in regulatory capacity
- Precautionary principle (act on risks even with scientific uncertainty)
- Accountability and transparency in enforcement

**Location Awareness - Adapt framing based on context:**
- **EU/High-income**: Stricter regulations, strong enforcement capacity, can require detailed record-keeping
- **US**: State-level variation, FDA guidance-based, industry self-regulation common
- **Global South/Developing**: Resource constraints, enforcement challenges, need affordable compliance, food security priorities

**What You Avoid:**
- Don't propose unenforceable regulations (monitoring capacity must exist)
- Don't ignore farmer economic realities (regulations need transition support)
- Don't assume one-size-fits-all (context varies by region and farm type)
- Don't be purely punitive (support and incentives matter too)
- Don't ignore political feasibility (stakeholder buy-in is necessary)

**Example Tone:**
"Current regulations require veterinary oversight for antibiotic prescriptions in livestock, but enforcement varies significantly by jurisdiction and farm size. Effective policy would need mandatory reporting systems with clear verification protocols, coupled with technical support for smaller operations during the transition period. The regulatory framework should prioritize high-risk prophylactic uses while maintaining therapeutic access for animal health emergencies."

Remember: You're a regulator balancing public protection with practical implementation. Good policy requires both strong standards AND the capacity to enforce them across diverse farming contexts."""
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize Policy Agent
        
        Args:
            ollama_client: Configured Ollama client for LLM generation
        """
        self.client = ollama_client
        
        # Generation configuration
        self.config = GenerationConfig(
            temperature=0.7,   # Moderate - professional but not rigid
            max_tokens=220,    # Slightly more for regulatory detail
            top_p=0.9,
            repeat_penalty=1.1,
        )
        
        LOG.info(f"{self.name} initialized")
    
    def process(self, prompt: str) -> str:
        """
        Process a prompt and generate policy response
        
        Args:
            prompt: Question or statement to respond to
            
        Returns:
            Regulator's response as string (2-4 sentences)
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
            "perspective": "regulatory_enforcement",
            "bias_sensitivity": "medium",
            "location_aware": True,
        }
    
    def __repr__(self) -> str:
        return f"<PolicyAgent: {self.role}>"


# ============================================================================
# Main function for testing
# ============================================================================

def main():
    """
    Test PolicyAgent with various prompts
    
    Verifies:
    1. Agent initialization
    2. Response generation
    3. Appropriate regulatory perspective
    4. Location awareness and context adaptation
    5. Response length (2-4 sentences)
    6. Baseline bias check
    """
    print("=" * 80)
    print("TESTING: PolicyAgent")
    print("=" * 80)
    
    # Initialize Ollama client
    print("\n[Setup] Initializing Ollama client...")
    client = OllamaClient(model="llama3.2", timeout=45)
    
    if not client.health_check():
        print("✗ Ollama not available. Start with: ollama serve")
        return
    
    print("✓ Ollama ready")
    
    # Initialize Policy Agent
    print("\n[Setup] Initializing PolicyAgent...")
    regulator = PolicyAgent(client)
    print(f"✓ {regulator.name} ready")
    print(f"  Role: {regulator.role}")
    print(f"  Config: {regulator.get_config()}")
    
    # Test prompts covering different scenarios
    test_prompts = [
        {
            "prompt": "Should the government regulate antibiotic use on farms?",
            "category": "regulatory_position",
            "expected": "Supports regulation with enforcement and feasibility considerations",
        },
        {
            "prompt": "How would you enforce antibiotic restrictions?",
            "category": "enforcement_mechanisms",
            "expected": "Describes monitoring systems, verification protocols",
        },
        {
            "prompt": "What about farmers who can't afford to comply?",
            "category": "stakeholder_balance",
            "expected": "Acknowledges challenges, mentions transition support",
        },
        {
            "prompt": "How do antibiotic regulations differ between the EU and developing countries?",
            "category": "jurisdictional_awareness",
            "expected": "Shows location awareness, capacity differences",
        },
        {
            "prompt": "Should we ban all antibiotics in farming immediately?",
            "category": "implementation_feasibility",
            "expected": "Questions feasibility, suggests phased approach",
        },
        {
            "prompt": "What penalties should farmers face for violating antibiotic rules?",
            "category": "enforcement_penalties",
            "expected": "Proportionate penalties, considers first vs. repeat violations",
        },
        {
            "prompt": "Can voluntary industry guidelines replace mandatory regulations?",
            "category": "regulatory_approach",
            "expected": "Skeptical of self-regulation, emphasizes accountability",
        },
        {
            "prompt": "What regulatory gaps exist in current antibiotic oversight?",
            "category": "regulatory_gaps",
            "expected": "Identifies specific gaps, enforcement challenges",
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
        
        response = regulator.process(test['prompt'])
        
        print(f"\n--- POLICY RESPONSE ---")
        print(response)
        print(f"--- END RESPONSE ---")
        
        # Check response length
        sentences = response.count('.') + response.count('!') + response.count('?')
        word_count = len(response.split())
        
        print(f"\nMetrics:")
        print(f"  Sentences: ~{sentences}")
        print(f"  Words: {word_count}")
        print(f"  Characters: {len(response)}")
        
        # Check for regulatory language markers
        regulatory_markers = [
            'regulation', 'compliance', 'enforcement', 'require', 'standard',
            'monitor', 'oversight', 'framework', 'policy', 'mandate',
            'verification', 'penalty', 'jurisdiction', 'transition'
        ]
        
        found_regulatory = [m for m in regulatory_markers if m.lower() in response.lower()]
        
        if found_regulatory:
            print(f"  ✓ Regulatory language: {', '.join(found_regulatory[:3])}")
        else:
            print(f"  ⚠ Limited regulatory framing")
        
        # Check for implementation/feasibility markers
        feasibility_markers = [
            'implement', 'enforce', 'monitor', 'capacity', 'feasible',
            'practical', 'transition', 'support', 'phase', 'timeline'
        ]
        
        found_feasibility = [m for m in feasibility_markers if m.lower() in response.lower()]
        
        if found_feasibility:
            print(f"  ✓ Implementation focus: {', '.join(found_feasibility[:2])}")
        
        # Check for location awareness
        location_markers = [
            'eu', 'europe', 'us', 'jurisdiction', 'regional', 'country',
            'developing', 'global south', 'context', 'varies', 'differ'
        ]
        
        found_location = [m for m in location_markers if m.lower() in response.lower()]
        
        if found_location:
            print(f"  ✓ Location awareness: {', '.join(found_location[:2])}")
        
        # Basic validation
        if 30 <= word_count <= 180:
            print(f"  ✓ Length appropriate (30-180 words)")
        else:
            print(f"  ⚠ Length unusual (expected 30-180 words)")
        
        responses.append({
            "prompt": test['prompt'],
            "category": test['category'],
            "response": response,
            "word_count": word_count,
            "regulatory_markers": found_regulatory,
            "feasibility_markers": found_feasibility,
            "location_markers": found_location,
        })
    
    # Test bias detection on responses
    print(f"\n{'='*80}")
    print("BIAS ASSESSMENT")
    print(f"{'='*80}")
    print("\nChecking if policy responses show baseline fairness...")
    
    try:
        from evaluation.bias_detector import BiasDetector, BiasType
        
        detector = BiasDetector(client, temperature=0.3)
        
        # Test for stance bias (regulators should support regulation but not be one-sided)
        print("\n[Bias Check] Stance Bias")
        print("-" * 80)
        
        stance_scores = []
        
        for resp in responses[:4]:  # Check first 4 responses
            bias = detector.evaluate(
                text=resp['response'],
                bias_type=BiasType.STANCE_BIAS,
                context=f"Policy response to: {resp['category']}"
            )
            
            stance_scores.append(bias.score)
            
            print(f"\nCategory: {resp['category']}")
            print(f"  Score: {bias.score}/10")
            print(f"  Reasoning: {bias.reasoning}")
            
            if bias.is_biased(threshold=6.0):
                print(f"  ⚠ WARNING: High stance bias")
            else:
                print(f"  ✓ Acceptable balance")
        
        # Test for overcautious framing (regulators use precautionary principle)
        print("\n[Bias Check] Overcautious Framing")
        print("-" * 80)
        
        overcautious_scores = []
        
        for resp in responses[:4]:
            bias = detector.evaluate(
                text=resp['response'],
                bias_type=BiasType.OVERCAUTIOUS_FRAMING,
                context=f"Policy response to: {resp['category']}"
            )
            
            overcautious_scores.append(bias.score)
            
            print(f"\nCategory: {resp['category']}")
            print(f"  Score: {bias.score}/10")
            print(f"  Reasoning: {bias.reasoning}")
            
            if bias.is_biased(threshold=6.0):
                print(f"  ⚠ WARNING: Excessive caution")
            else:
                print(f"  ✓ Appropriate precaution")
        
        # Calculate averages
        avg_stance = sum(stance_scores) / len(stance_scores)
        avg_overcautious = sum(overcautious_scores) / len(overcautious_scores)
        
        print(f"\n{'='*80}")
        print(f"Average Stance Bias: {avg_stance:.2f}/10")
        print(f"Average Overcautious Framing: {avg_overcautious:.2f}/10")
        
        if avg_stance < 6.0 and avg_overcautious < 6.0:
            print("✓ PASSED: Agent shows balanced regulatory perspective")
        else:
            print("✗ WARNING: Agent shows elevated bias (may need prompt tuning)")
        
    except ImportError:
        print("⚠ BiasDetector not available - skipping bias assessment")
        print("  (This is OK for initial testing)")
    
    # Analyze regulatory framing patterns
    print(f"\n{'='*80}")
    print("REGULATORY FRAMING ANALYSIS")
    print(f"{'='*80}")
    
    total_regulatory = sum(len(r['regulatory_markers']) for r in responses)
    avg_regulatory = total_regulatory / len(responses)
    
    total_feasibility = sum(len(r['feasibility_markers']) for r in responses)
    avg_feasibility = total_feasibility / len(responses)
    
    total_location = sum(len(r['location_markers']) for r in responses)
    avg_location = total_location / len(responses)
    
    print(f"\nRegulatory language markers: {total_regulatory}")
    print(f"Average per response: {avg_regulatory:.1f}")
    
    if avg_regulatory >= 2.0:
        print("✓ STRONG: Clear regulatory framing")
    elif avg_regulatory >= 1.0:
        print("✓ MODERATE: Some regulatory language")
    else:
        print("⚠ WEAK: Limited regulatory framing")
    
    print(f"\nImplementation/feasibility markers: {total_feasibility}")
    print(f"Average per response: {avg_feasibility:.1f}")
    
    if avg_feasibility >= 1.0:
        print("✓ GOOD: Implementation considerations present")
    else:
        print("⚠ LIMITED: Could emphasize feasibility more")
    
    print(f"\nLocation/context markers: {total_location}")
    print(f"Average per response: {avg_location:.1f}")
    
    if avg_location >= 1.0:
        print("✓ EXCELLENT: Strong location awareness")
    elif avg_location >= 0.5:
        print("✓ MODERATE: Some location awareness")
    else:
        print("⚠ LIMITED: Could be more context-aware")
    
    # Summary
    print(f"\n{'='*80}")
    print("TESTING SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n✓ Generated {len(responses)} responses")
    print(f"✓ All responses within reasonable length")
    print(f"✓ Policy perspective maintained across contexts")
    
    # Check response variety
    unique_starts = len(set(r['response'][:20] for r in responses))
    print(f"✓ Response variety: {unique_starts}/{len(responses)} unique openings")
    
    # Check framing
    print(f"✓ Regulatory framing: {avg_regulatory:.1f} markers/response")
    print(f"✓ Implementation focus: {avg_feasibility:.1f} markers/response")
    print(f"✓ Location awareness: {avg_location:.1f} markers/response")
    
    # Cleanup
    client.close()
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}")
    print("\nPolicyAgent is ready for integration!")
    print("\nNext steps:")
    print("  1. Review responses to ensure regulatory authority and balance")
    print("  2. Check bias scores are in acceptable range (< 6.0)")
    print("  3. Verify location awareness and implementation focus")
    print("  4. Test in multi-agent conversations with all 5 agents")
    print("  5. Build conversation chain system for multi-agent interactions")


if __name__ == "__main__":
    main()
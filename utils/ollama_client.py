"""
Ollama Client
Wrapper for Ollama API with retry logic, error handling, and health checks

This module provides a robust interface to the Ollama LLM server running locally.
Handles connection failures, timeouts, and rate limiting gracefully.
"""

import os
# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
LOG = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for LLM generation"""
    temperature: float = 0.7
    max_tokens: int = 200
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    
    def to_ollama_options(self) -> Dict[str, Any]:
        """Convert to Ollama API options format"""
        return {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
        }


@dataclass
class GenerationResult:
    """Result from LLM generation"""
    text: str
    success: bool
    error_message: Optional[str] = None
    generation_time: Optional[float] = None
    tokens_generated: Optional[int] = None


class OllamaClient:
    """
    Robust client for Ollama API
    
    Features:
    - Automatic retry with exponential backoff
    - Connection pooling for better performance
    - Health checks and error handling
    - Rate limiting support
    - Detailed logging
    
    Example:
        client = OllamaClient(model="llama3.2")
        result = client.generate(
            prompt="What is photosynthesis?",
            system_prompt="You are a helpful biology teacher."
        )
        print(result.text)
    """
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        verify_ssl: bool = True,
    ):
        """
        Initialize Ollama client
        
        Args:
            model: Name of Ollama model to use (e.g., "llama3.2", "mixtral")
            base_url: Base URL of Ollama server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier (retry_delay = backoff * 2^attempt)
            verify_ssl: Whether to verify SSL certificates
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Create session for connection pooling
        self.session = requests.Session()
        self.session.verify = verify_ssl
        
        # Endpoints
        self.generate_url = f"{self.base_url}/api/generate"
        self.tags_url = f"{self.base_url}/api/tags"
        self.version_url = f"{self.base_url}/api/version"
        
        LOG.info(f"OllamaClient initialized: model={model}, base_url={base_url}")
    
    def health_check(self) -> bool:
        """
        Check if Ollama server is running and accessible
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = self.session.get(self.tags_url, timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Check if our model is available
                model_available = any(self.model in name for name in model_names)
                
                if model_available:
                    LOG.info(f"Health check passed: {self.model} available")
                    return True
                else:
                    LOG.warning(
                        f"Health check warning: {self.model} not found. "
                        f"Available models: {model_names}"
                    )
                    return False
            else:
                LOG.error(f"Health check failed: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            LOG.error(
                f"Health check failed: Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start with: ollama serve"
            )
            return False
        except Exception as e:
            LOG.error(f"Health check failed: {str(e)}")
            return False
    
    def list_models(self) -> list[str]:
        """
        List all available models on Ollama server
        
        Returns:
            List of model names
        """
        try:
            response = self.session.get(self.tags_url, timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "") for m in models]
            else:
                LOG.error(f"Failed to list models: HTTP {response.status_code}")
                return []
        except Exception as e:
            LOG.error(f"Failed to list models: {str(e)}")
            return []
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate response from Ollama model
        
        Args:
            prompt: User prompt/question
            system_prompt: Optional system instruction that defines agent behavior
            config: Optional generation configuration (temperature, max_tokens, etc.)
            
        Returns:
            GenerationResult with text response and metadata
        """
        if config is None:
            config = GenerationConfig()
        
        # Build full prompt with system context
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = prompt
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": config.to_ollama_options(),
        }
        
        # Attempt generation with retries
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                LOG.debug(f"Attempt {attempt + 1}/{self.max_retries + 1}: Generating response")
                
                response = self.session.post(
                    self.generate_url,
                    json=payload,
                    timeout=self.timeout,
                )
                
                # Check response status
                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    LOG.warning(error_msg)
                    
                    # Don't retry on client errors (4xx)
                    if 400 <= response.status_code < 500:
                        return GenerationResult(
                            text="",
                            success=False,
                            error_message=error_msg,
                        )
                    
                    # Retry on server errors (5xx)
                    if attempt < self.max_retries:
                        self._wait_before_retry(attempt)
                        continue
                    else:
                        return GenerationResult(
                            text="",
                            success=False,
                            error_message=error_msg,
                        )
                
                # Parse response
                data = response.json()
                generated_text = (data.get("response") or "").strip()
                
                generation_time = time.time() - start_time
                
                # Extract token count if available
                tokens = data.get("eval_count")
                
                LOG.info(
                    f"Generation successful: {len(generated_text)} chars, "
                    f"{generation_time:.2f}s"
                )
                
                return GenerationResult(
                    text=generated_text,
                    success=True,
                    generation_time=generation_time,
                    tokens_generated=tokens,
                )
                
            except requests.exceptions.Timeout:
                error_msg = f"Request timeout after {self.timeout}s"
                LOG.warning(f"{error_msg} (attempt {attempt + 1})")
                
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                else:
                    return GenerationResult(
                        text="",
                        success=False,
                        error_message=error_msg,
                    )
            
            except requests.exceptions.ConnectionError:
                error_msg = "Connection error: Cannot reach Ollama server"
                LOG.warning(f"{error_msg} (attempt {attempt + 1})")
                
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                else:
                    return GenerationResult(
                        text="",
                        success=False,
                        error_message=error_msg,
                    )
            
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                LOG.error(f"{error_msg} (attempt {attempt + 1})")
                
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                else:
                    return GenerationResult(
                        text="",
                        success=False,
                        error_message=error_msg,
                    )
        
        # Should never reach here, but just in case
        return GenerationResult(
            text="",
            success=False,
            error_message="Maximum retries exceeded",
        )
    
    def _wait_before_retry(self, attempt: int) -> None:
        """
        Wait before retry with exponential backoff
        
        Args:
            attempt: Current attempt number (0-indexed)
        """
        wait_time = self.backoff_factor * (2 ** attempt)
        LOG.info(f"Waiting {wait_time:.2f}s before retry...")
        time.sleep(wait_time)
    
    def close(self) -> None:
        """Close the session and cleanup resources"""
        self.session.close()
        LOG.info("OllamaClient session closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# ============================================================================
# Main function for testing
# ============================================================================

def main():
    """
    Test OllamaClient with various scenarios
    
    This function verifies:
    1. Connection to Ollama server
    2. Model availability
    3. Basic generation
    4. Error handling
    5. Configuration options
    """
    print("=" * 80)
    print("TESTING: OllamaClient")
    print("=" * 80)
    
    # Test 1: Initialize client
    print("\n[Test 1] Initializing client...")
    client = OllamaClient(
        model="llama3.2",
        timeout=30,
        max_retries=2,
    )
    print(" Client initialized")
    
    # Test 2: Health check
    print("\n[Test 2] Running health check...")
    if client.health_check():
        print(" Ollama server is healthy")
    else:
        print(" Health check failed - is Ollama running?")
        print("  Start Ollama with: ollama serve")
        print("  Pull model with: ollama pull llama3.2")
        return
    
    # Test 3: List available models
    print("\n[Test 3] Listing available models...")
    models = client.list_models()
    if models:
        print(f" Found {len(models)} models:")
        for model in models:
            print(f"  - {model}")
    else:
        print(" No models found")
    
    # Test 4: Basic generation
    print("\n[Test 4] Testing basic generation...")
    result = client.generate(
        prompt="What is 2+2? Answer in one sentence.",
        config=GenerationConfig(temperature=0.1, max_tokens=50),
    )
    
    if result.success:
        print(f" Generation successful")
        print(f"  Response: {result.text}")
        print(f"  Time: {result.generation_time:.2f}s")
        if result.tokens_generated:
            print(f"  Tokens: {result.tokens_generated}")
    else:
        print(f" Generation failed: {result.error_message}")
    
    # Test 5: Generation with system prompt
    print("\n[Test 5] Testing generation with system prompt...")
    result = client.generate(
        prompt="What are antibiotics used for in farming?",
        system_prompt="You are a helpful agricultural scientist. Keep responses to 2-3 sentences.",
        config=GenerationConfig(temperature=0.7, max_tokens=100),
    )
    
    if result.success:
        print(f" Generation with system prompt successful")
        print(f"  Response: {result.text}")
    else:
        print(f" Generation failed: {result.error_message}")
    
    # Test 6: Configuration variations
    print("\n[Test 6] Testing different temperature settings...")
    
    test_prompt = "Describe a sunset."
    
    for temp in [0.3, 0.7, 1.0]:
        config = GenerationConfig(temperature=temp, max_tokens=50)
        result = client.generate(prompt=test_prompt, config=config)
        
        if result.success:
            print(f"  temp={temp}: {result.text[:60]}...")
        else:
            print(f"  temp={temp}: Failed")
    
    # Test 7: Error handling (invalid model)
    print("\n[Test 7] Testing error handling...")
    bad_client = OllamaClient(model="nonexistent-model-xyz", max_retries=1)
    result = bad_client.generate(prompt="test")
    
    if not result.success:
        print(f" Error handled correctly: {result.error_message}")
    else:
        print(" Should have failed with invalid model")
    
    # Test 8: Context manager usage
    print("\n[Test 8] Testing context manager...")
    try:
        with OllamaClient(model="llama3.2") as ctx_client:
            result = ctx_client.generate(
                prompt="Say 'Hello world'",
                config=GenerationConfig(max_tokens=20),
            )
            if result.success:
                print(f" Context manager works: {result.text}")
    except Exception as e:
        print(f" Context manager failed: {e}")
    
    # Cleanup
    print("\n[Cleanup] Closing client...")
    client.close()
 

if __name__ == "__main__":
    main()
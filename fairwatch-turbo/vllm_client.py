import aiohttp
import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    text: str
    success: bool
    error_message: Optional[str] = None

class AsyncVLLMClient:
    """
    Native Asynchronous vLLM Client using aiohttp.
    Designed for high-throughput concurrency.
    """
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY", model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.completion_url = f"{base_url}/completions"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    async def generate(self, prompt: str, system_prompt: str = None, config: Any = None) -> GenerationResult:
        """
        Async generation with exact Llama 3 prompt formatting parity.
        """
        try:
            # construction of llama 3 chat template manually
            if system_prompt:
                 full_prompt = (
                     f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                     f"{system_prompt}<|eot_id|>"
                     f"<|start_header_id|>user<|end_header_id|>\n\n"
                     f"{prompt}<|eot_id|>"
                     f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                 )
            else:
                 full_prompt = (
                     f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                     f"{prompt}<|eot_id|>"
                     f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                 )

            # get config values or use defaults
            temperature = config.temperature if config and hasattr(config, 'temperature') else 0.7
            max_tokens = config.max_tokens if config and hasattr(config, 'max_tokens') else 500
            
            # Helper for JSON mode (future proofing)
            extra_body = {}
            if config and getattr(config, 'json_mode', False):
               pass

            # prepare the request payload, adding stop tokens to prevent run-ons
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": ["<|eot_id|>", "<|end_of_text|>", "User:", "System:", "Note:", "``` \n", "```\n\n"],
                **extra_body
            }

            # output async request to vllm server
            async with aiohttp.ClientSession() as session:
                async with session.post(self.completion_url, json=payload, headers=self.headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return GenerationResult(text="", success=False, error_message=f"HTTP {response.status}: {error_text}")
                    
                    data = await response.json()
                    output_text = data['choices'][0]['text']
                    return GenerationResult(text=output_text, success=True)

        except Exception as e:
            logger.error(f"Error calling vLLM Async: {e}")
            return GenerationResult(text="", success=False, error_message=str(e))

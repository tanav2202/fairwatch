import requests
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

# logging setup
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    # simple config for generation
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 0.9
    json_mode: bool = False
    
    def to_options(self): 
        # mapping to ollama options format
        return {"temperature": self.temperature, "num_predict": self.max_tokens, "top_p": self.top_p}

@dataclass
class GenerationResult:
    # helper for return values
    text: str
    success: bool
    error_message: Optional[str] = None

class OllamaClient:
    # basic client to talk to the local ollama server
    def __init__(self, model="llama3.2", base_url="http://127.0.0.1:11434", timeout=300):
        import os
        # check if environment variable is set for the host
        base_url = os.environ.get('OLLAMA_HOST', base_url)
        self.model = model
        self.generate_url = f"{base_url}/api/generate"
        self.timeout = timeout
        
        # setup connection pool
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=40, pool_maxsize=40)
        self.session.mount('http://', adapter)

    def generate(self, prompt, system_prompt=None, config=None):
        # standard generation call
        if not config: config = GenerationConfig()
        
        # combining system and user prompts if needed
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}" if system_prompt else prompt
        
        payload = {
            "model": self.model, 
            "prompt": full_prompt, 
            "stream": False, 
            "options": config.to_options()
        }
        
        if config.json_mode: 
            payload["format"] = "json"

        try:
            res = self.session.post(self.generate_url, json=payload, timeout=self.timeout)
            if res.status_code == 200: 
                return GenerationResult(text=res.json().get("response", ""), success=True)
            return GenerationResult(text="", success=False, error_message=f"HTTP {res.status_code}")
        except Exception as e:
            return GenerationResult(text="", success=False, error_message=str(e))

import os
import json
from kivi.utils.llm_client import LLMClient

class VerifyClaims:
    def __init__(self, mllm_model="google/gemini-3.1-pro-preview"):
        self.llm_client = LLMClient(model=mllm_model)
        self.prompt_template_path = os.path.join(os.path.dirname(__file__), '../../prompts/claim_verification.txt')
        
    def verify(self, claims_json):
        """
        Verifies a list of claims extracted from the video.
        """
        with open(self.prompt_template_path, 'r', encoding='utf-8') as f:
            template = f.read()
            
        if isinstance(claims_json, dict):
            claims_str = json.dumps(claims_json, ensure_ascii=False, indent=2)
        else:
            claims_str = str(claims_json)
            
        prompt = f"Please verify the claims based on the following instruction:\n\n{template}\n\nClaims to Verify:\n{claims_str}"
        
        return self.llm_client.call_text(
            prompt=prompt,
            response_format="json",
            temperature=0
        )

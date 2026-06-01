import os
import json
from kivi.llm_client import LLMClient

class ExtractClaims:
    def __init__(self, mllm_model="google/gemini-3.1-pro-preview"):
        self.llm_client = LLMClient(model=mllm_model)
        self.prompt_template_path = os.path.join(os.path.dirname(__file__), '../../prompts/claim_extraction.txt')
        
    def extract(self, video_path, question):
        """
        Extracts atomic claims from a video.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        with open(self.prompt_template_path, 'r', encoding='utf-8') as f:
            template = f.read()
            
        prompt = f"Please process the provided video based on the following instruction:\n\n{template}\n\nQuestion: {question}"
        
        return self.llm_client.call_vision(
            prompt=prompt,
            video_path=video_path,
            response_format="json",
            temperature=0
        )

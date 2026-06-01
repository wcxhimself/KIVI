import os
import json
from kivi.llm_client import LLMClient

class HelpfulnessEvaluator:
    def __init__(self, mllm_model="google/gemini-3.1-pro-preview"):
        self.llm_client = LLMClient(model=mllm_model)
        self.prompt_template_path = os.path.join(os.path.dirname(__file__), '../../prompts/helpfulness_evaluation.txt')
        
    def evaluate(self, video_path, question):
        """
        Evaluates the helpfulness of the generated video for solving the task required by the question.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        with open(self.prompt_template_path, 'r', encoding='utf-8') as f:
            template = f.read()
            
        instruction = template.replace("{QUESTION}", question).replace("{VIDEO}", "Please see the attached video frames.")
        
        return self.llm_client.call_vision(
            prompt=instruction,
            video_path=video_path,
            response_format="json",
            temperature=0
        )

import os
import json
from kivi.utils.llm_client import LLMClient

class DynamicScriptGenerator:
    def __init__(self, model_name="google/gemini-3.1-pro-preview"):
        self.llm_client = LLMClient(model=model_name)
        self.prompt_dir = os.path.join(os.path.dirname(__file__), '../../prompts')

    def _load_prompt(self, filename):
        prompt_path = os.path.join(self.prompt_dir, filename)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _fill_template(self, template: str, **values) -> str:
        """
        Safely substitute placeholders like {initial_prompt} without using str.format().

        Our prompt templates contain JSON examples with lots of '{' and '}', which
        would be interpreted by str.format() and cause KeyError.
        """
        filled = template
        for key, value in values.items():
            placeholder = "{" + str(key) + "}"
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, ensure_ascii=False)
            else:
                value_str = "" if value is None else str(value)
            filled = filled.replace(placeholder, value_str)
        return filled

    def generate_outline(self, initial_prompt):
        """
        Step 1: Generate the high-level 10-14 step outline for the video.
        """
        template = self._load_prompt('outline_prompt.txt')
        sys_prompt = "You are a Master Video Director & Outline Planner."
        prompt = self._fill_template(template, initial_prompt=initial_prompt)

        for attempt in range(3):
            result = self.llm_client.call_text(prompt=prompt, system_prompt=sys_prompt, response_format="json", temperature=0)
            if isinstance(result, dict) and 'outline_steps' in result:
                return result
            print(f"[script_generator] Outline JSON parse failed (attempt {attempt+1}/3), retrying...")
        raise ValueError(f"Failed to generate valid outline JSON after 3 attempts. Raw: {result}")

    def generate_first_segment_config(self, initial_prompt, outline_step, required_props, reference_num_frames):
        """
        Step 2: Generate the JSON config for the very first segment including markers.
        """
        template = self._load_prompt('segment_1_prompt.txt')
        sys_prompt = "You are a Video Generation Configuration Assistant."
        prompt = self._fill_template(
            template,
            initial_prompt=initial_prompt,
            visual_action=outline_step.get("visual_action", ""),
            camera_angle=outline_step.get("camera_angle", ""),
            required_props=required_props,
            reference_num_frames=reference_num_frames,
        )
        return self.llm_client.call_text(prompt=prompt, system_prompt=sys_prompt, response_format="json", temperature=0)

    def generate_full_script(self, initial_prompt, outline):
        """
        Generate a complete video script (single_prompt + interactive_prompts) from a full outline in one pass.
        """
        template = self._load_prompt('script_generate.txt')
        sys_prompt = "You are a Long-Form Video Script Generator."
        prompt = self._fill_template(
            template,
            initial_prompt=initial_prompt,
            full_outline=outline,
            required_subjects_and_props=outline.get("required_subjects_and_props", ""),
        )
        for attempt in range(3):
            result = self.llm_client.call_text(prompt=prompt, system_prompt=sys_prompt, response_format="json", temperature=0)
            if isinstance(result, dict) and 'interactive_prompts' in result and 'single_prompt' in result:
                return result
            print(f"[script_generator] Full script JSON parse failed (attempt {attempt+1}/3), retrying...")
        raise ValueError(f"Failed to generate valid full script JSON after 3 attempts. Raw: {result}")

    def generate_next_segment_prompt(self, initial_prompt, current_idx, identity_marker, continuity_anchors, outline_step, is_final_step, previous_video_path, reference_num_frames):
        """
        Step 3: Vision-conditioned iterative prompt generation for segments 2..N.
        """
        template = self._load_prompt('iterative_prompt.txt')
        sys_prompt = "You are a Video-Conditioned Iterative Script Generator."
        prompt = self._fill_template(
            template,
            initial_prompt=initial_prompt,
            current_idx=current_idx,
            identity_marker=identity_marker,
            continuity_anchors=continuity_anchors,
            visual_action=outline_step.get("visual_action", ""),
            camera_angle=outline_step.get("camera_angle", ""),
            is_final_step=str(is_final_step).lower(),
            reference_num_frames=reference_num_frames,
        )

        return self.llm_client.call_vision(
            prompt=prompt,
            video_path=previous_video_path,
            system_prompt=sys_prompt,
            response_format="json",
            temperature=0
        )

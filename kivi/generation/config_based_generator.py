import os
import re
import sys
import json
import glob
import time
import yaml
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional

from .video_generator import BaseVideoGenerator, _snap_to_4n_plus_1, compute_reference_num_frames
from .model_config import ModelConfig

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ConfigBasedGenerator(BaseVideoGenerator):
    """Generic video generator driven by a YAML configuration file.

    Supports three modes defined in the YAML:
      - "segment":       segment-by-segment T2V+I2V (e.g., Wan2.2, HunyuanVideo-1.5)
      - "interactive":   prompt switching within single generation (e.g., LongCat, LongLive)
      - "single_prompt": one prompt, one generation (e.g., Helios)
    """

    def __init__(self, config: ModelConfig, gpu: str = "0", **kwargs):
        super().__init__(config.name, **kwargs)
        self.config = config
        self.segment_counter = 0

        gpu_ids = gpu.split(",")
        self.num_gpus = len(gpu_ids)
        self.cuda_visible_devices = gpu

        self.code_dir = config.code_dir
        if self.code_dir and not os.path.isabs(self.code_dir):
            self.code_dir = os.path.join(_PROJECT_ROOT, self.code_dir)

        self.model_paths = {}
        for key, path in config.model_path.items():
            if path and not os.path.isabs(path):
                self.model_paths[key] = os.path.join(_PROJECT_ROOT, path)
            else:
                self.model_paths[key] = path

    # ── segment mode ─────────────────────────────────────────

    def _snap_frames(self, num_frames: Optional[int]) -> Optional[int]:
        """Apply frame snapping rule from config. Returns None if num_frames is None."""
        if num_frames is None:
            return None
        rule = self.config.frame_snapping
        if rule == "4n+1":
            return _snap_to_4n_plus_1(num_frames, min_val=41, max_val=self.config.max_num_frames)
        if rule == "multiple_of_3":
            return max(3, min(self.config.max_num_frames, round(num_frames / 3) * 3))
        return num_frames

    def _build_command(self, template: str, **placeholders) -> str:
        """Substitute {placeholders} in a command template, collapse whitespace."""
        command = template.format(**placeholders)
        lines = [l.strip() for l in command.strip().split("\n") if l.strip()]
        return " ".join(lines)

    def _get_segment_template(self, is_i2v: bool) -> dict:
        """Get the appropriate template for segment generation."""
        use_multi_gpu = self.num_gpus > 1 and self.config.multi_gpu
        if use_multi_gpu:
            templates = self.config.multi_gpu
        else:
            templates = self.config.segment

        key = "i2v" if is_i2v else "t2v"
        return templates.get(key, {})

    def generate_segment(
        self,
        prompt: str,
        num_frames: Optional[int] = None,
        prev_frame=None,
        output_path: Optional[str] = None,
        seed: int = 42,
    ) -> Optional[str]:
        if self.config.mode != "segment":
            raise NotImplementedError(
                f"{self.config.name} is in '{self.config.mode}' mode, "
                f"not segment-by-segment. Use generate_video_dynamically()."
            )

        self.segment_counter += 1
        is_i2v = prev_frame is not None

        if output_path is None:
            default_outputs = os.path.join(_PROJECT_ROOT, "outputs")
            os.makedirs(default_outputs, exist_ok=True)
            output_path = os.path.join(default_outputs, f"{self.config.name}_segment_{self.segment_counter}.mp4")
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        snapped = self._snap_frames(num_frames)
        computed_seed = seed + self.segment_counter if seed is not None else -1

        frame_num_arg = f"--frame_num {snapped}" if snapped is not None else ""
        model_path_key = "i2v" if is_i2v else "t2v"
        model_path = self.model_paths.get(model_path_key, "") or next(iter(self.model_paths.values()), "")

        ref_image_path = ""
        if is_i2v and prev_frame is not None:
            target_dir = os.path.dirname(output_path)
            ref_image_path = os.path.join(target_dir, f"temp_i2v_start_{self.segment_counter}.jpg")
            prev_frame.save(ref_image_path)

        template = self._get_segment_template(is_i2v)
        torchrun_prefix = f"{sys.executable} -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint localhost:0"

        command = self._build_command(
            template["command"],
            code_dir=self.code_dir,
            model_path=model_path,
            prompt=prompt,
            seed=str(computed_seed),
            output_path=output_path,
            ref_image_path=ref_image_path,
            frame_num_arg=frame_num_arg,
            num_gpus=str(self.num_gpus),
            torchrun=torchrun_prefix,
        )

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{self.code_dir}:{existing}" if existing else self.code_dir
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        for k, v in self.config.get_env(self.code_dir).items():
            env[k] = v

        print(f"[{self.config.name}] {'I2V' if is_i2v else 'T2V'} segment {self.segment_counter}: {command}")
        try:
            subprocess.run(command, shell=True, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{self.config.name}] Generation failed: {e}")
            raise
        finally:
            if ref_image_path and os.path.exists(ref_image_path):
                os.remove(ref_image_path)

        return output_path

    # ── non-segment mode (interactive / single_prompt) ─────

    def _write_interactive_prompts(self, script: dict, output_dir: str) -> str:
        """Convert script.json → interactive_prompts.json with per-segment num_frames."""
        interactive_prompts = script.get("interactive_prompts", [])
        seg_data = []
        for seg in interactive_prompts:
            dur = seg.get("duration_seconds", 4.0)
            nf = compute_reference_num_frames(
                duration_seconds=dur,
                fps=self.config.fps,
                max_num_frames=self.config.max_num_frames,
            )
            seg_data.append({"prompt": seg["prompt"], "num_frames": nf})

        prompt_file = os.path.join(output_dir, "interactive_prompts.json")
        with open(prompt_file, "w", encoding="utf-8") as f:
            json.dump(seg_data, f, indent=4, ensure_ascii=False)
        return prompt_file

    def _write_single_prompt(self, script: dict, output_dir: str) -> str:
        """Convert script.json → single_prompt.txt for models that take one prompt."""
        single_prompt = script.get("single_prompt", "")
        if not single_prompt:
            raise ValueError("No 'single_prompt' field in script.json")
        prompt_file = os.path.join(output_dir, "single_prompt.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(single_prompt + "\n")
        return prompt_file

    def _compute_total_frames(self, script: dict) -> int:
        interactive_prompts = script.get("interactive_prompts", [])
        total_duration = sum(seg.get("duration_seconds", 0) for seg in interactive_prompts)
        raw = round(total_duration * self.config.fps)
        snapped = self._snap_frames(raw) or raw
        return max(99, snapped)

    def _collect_output(self, output_dir: str, final_video_path: str) -> Optional[str]:
        patterns = self.config.output_patterns or ["*.mp4"]
        candidates = []
        for pattern in patterns:
            candidates.extend(glob.glob(os.path.join(output_dir, pattern)))

        candidates = sorted(set(candidates))
        if candidates:
            shutil.move(candidates[-1], final_video_path)
            print(f"[{self.config.name}] Final video saved to: {final_video_path}")
            return final_video_path

        print(f"[{self.config.name}] WARNING: No output video found in {output_dir}")
        return None

    def _run_non_segment_generation(self, output_dir: str, **template_vars) -> None:
        torchrun_prefix = f"{sys.executable} -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint localhost:0"

        template_vars.setdefault("output_dir", output_dir)
        template_vars.setdefault("num_frames", "")

        generation_cfg = self.config.generation
        if self.num_gpus > 1 and self.config.multi_gpu and "command" in self.config.multi_gpu:
            generation_cfg = self.config.multi_gpu

        command = self._build_command(
            generation_cfg["command"],
            code_dir=self.code_dir,
            model_path=next(iter(self.model_paths.values()), ""),
            num_gpus=str(self.num_gpus),
            torchrun=torchrun_prefix,
            **template_vars,
        )

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{self.code_dir}:{existing}" if existing else self.code_dir
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        for k, v in self.config.get_env(self.code_dir).items():
            env[k] = v

        print(f"[{self.config.name}] Running: {command}")
        subprocess.run(command, shell=True, env=env, check=True)

    def generate_video_dynamically(self, initial_prompt, script_generator, output_dir):
        mode = self.config.mode
        if mode == "segment":
            return super().generate_video_dynamically(initial_prompt, script_generator, output_dir)

        if mode not in ("interactive", "single_prompt"):
            raise NotImplementedError(
                f"ConfigBasedGenerator does not support mode='{mode}'. "
                f"Use segment, interactive, or single_prompt mode."
            )

        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{'='*50}")
        print(f"[{self.config.name}] Starting {mode} generation for: {initial_prompt}")
        print(f"[{self.config.name}] Outputs: {output_dir}")
        print(f"{'='*50}\n")

        model_output_parent = os.path.dirname(os.path.dirname(output_dir))
        script_parent = os.path.dirname(model_output_parent)
        relative = os.path.relpath(output_dir, model_output_parent)
        script_dir = os.path.join(script_parent, "common_script", relative)
        script_path = os.path.join(script_dir, "script.json")
        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"Script not found: {script_path}. Run outline + full script generation first."
            )
        with open(script_path, "r", encoding="utf-8") as f:
            script = json.load(f)

        shutil.copy(script_path, os.path.join(output_dir, "script.json"))

        final_video_path = os.path.join(output_dir, "final_video.mp4")
        if os.path.exists(final_video_path):
            print(f"[{self.config.name}] Final video already exists. Skipping generation...")
            return final_video_path

        model_name = self.config.name

        if model_name == "longlive":
            prompt_file = self._write_interactive_prompts(script, output_dir)
            num_frames = self._compute_total_frames(script)
            jsonl_path = os.path.join(output_dir, "interactive_prompts.jsonl")
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompts_data = json.load(f)
            with open(jsonl_path, "w", encoding="utf-8") as f:
                json.dump({"prompts": [p["prompt"] for p in prompts_data]}, f)
                f.write("\n")

            switch_indices = []
            cumsum = 0
            for seg in prompts_data[:-1]:
                cumsum += seg["num_frames"]
                switch_indices.append(cumsum)

            config_dict = {
                "denoising_step_list": [1000, 750, 500, 250],
                "warp_denoising_step": True,
                "num_frame_per_block": 3,
                "model_name": "Wan2.1-T2V-1.3B",
                "model_kwargs": {
                    "local_attn_size": 12,
                    "timestep_shift": 5.0,
                    "sink_size": 3,
                    "use_infinite_attention": True,
                },
                "data_path": jsonl_path,
                "output_folder": output_dir,
                "inference_iter": -1,
                "num_output_frames": num_frames,
                "use_ema": False,
                "seed": 42,
                "num_samples": 1,
                "save_with_index": True,
                "switch_frame_indices": ",".join(str(x) for x in switch_indices),
                "global_sink": True,
                "context_noise": 0,
                "generator_ckpt": self.model_paths.get("default", ""),
                "lora_ckpt": self.model_paths.get("lora", ""),
                "adapter": {
                    "type": "lora",
                    "rank": 256,
                    "alpha": 256,
                    "dropout": 0.0,
                    "dtype": "bfloat16",
                    "verbose": False,
                },
            }
            tmp_config = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            )
            yaml.dump(config_dict, tmp_config, default_flow_style=False)
            config_path = tmp_config.name
            tmp_config.close()

            try:
                self._run_non_segment_generation(
                    output_dir,
                    config_path=config_path,
                    prompt_file=prompt_file,
                    num_frames=str(num_frames),
                )
            finally:
                if os.path.exists(config_path):
                    os.remove(config_path)

        elif model_name == "helios":
            prompt_file = self._write_single_prompt(script, output_dir)
            num_frames = self._compute_total_frames(script)
            self._run_non_segment_generation(
                output_dir,
                prompt_file=prompt_file,
                num_frames=str(num_frames),
            )

        else:
            prompt_file = self._write_interactive_prompts(script, output_dir)
            self._run_non_segment_generation(
                output_dir,
                prompt_file=prompt_file,
                num_frames=str(self._compute_total_frames(script)),
            )

        return self._collect_output(output_dir, final_video_path)
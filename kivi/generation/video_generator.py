import os
import cv2
import json
import subprocess
from abc import ABC, abstractmethod


def _snap_to_4n_plus_1(n, min_val=41, max_val=121):
    """Snap a frame count to the nearest 4n+1 format required by Wan2.2 models.

    Valid values: 1, 5, 9, 13, ..., 41, 45, 49, ..., 257
    Since 8k+1 is a subset of 4n+1, this also works for models needing 8k+1.
    """
    if n is None:
        return None
    # default max 257
    max_val = max_val if max_val is not None else 257
    n = max(min_val, min(max_val, n))
    nearest_n = round((n - 1) / 4)
    snapped = 4 * nearest_n + 1
    return max(min_val, min(max_val, snapped))


def compute_reference_num_frames(duration_seconds, fps, max_num_frames=257):
    """Compute a reference num_frames from a duration in seconds and model fps.

    The result is snapped to the nearest valid frame count (4n+1 format).
    Clamped to [41, max_num_frames].
    """
    # default max 257
    if max_num_frames is None:
        max_num_frames = 257
    raw = round(duration_seconds * fps)
    return _snap_to_4n_plus_1(raw, min_val=41, max_val=max_num_frames)


class BaseVideoGenerator(ABC):
    # Subclasses must override these
    fps = 24
    max_num_frames = 257

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name

    @abstractmethod
    def generate_segment(self, prompt, num_frames=None, prev_frame=None, output_path=None, seed=42):
        """Generate a video segment.

        Args:
            prompt: Text prompt for generation.
            num_frames: Suggested frame count from LLM (content-adaptive).
                        The generator will snap this to a valid value for the
                        underlying model (e.g., 4n+1 for Wan2.2).
                        If None, uses the model's official default.
            prev_frame: PIL Image from the previous segment (I2V conditioning).
            output_path: Where to save the output video.
            seed: Random seed for reproducibility.
        """
        pass

    def extract_last_frame(self, video_segment_path):
        """
        Extract the last frame from a generated video segment to condition the next generation.
        Returns a PIL Image.
        """
        if not video_segment_path or not os.path.exists(video_segment_path):
            return None

        cap = cv2.VideoCapture(video_segment_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        cap.release()

        if ret:
            from PIL import Image
            # Convert BGR to RGB for diffusers
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        return None

    def concatenate_segments(self, segment_paths, output_path):
        """
        Concatenate all generated video segments into the final video using FFmpeg.
        """
        concat_file = os.path.join(os.path.dirname(output_path), "concat_list.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for p in segment_paths:
                if p and os.path.exists(p):
                    # FFmpeg requires safe format paths
                    f.write(f"file '{os.path.abspath(p)}'\n")

        print(f"[{self.model_name}] Concatenating {len(segment_paths)} segments into {output_path}...")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", output_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"[{self.model_name}] Final video saved at: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error during video concatenation: {e.stderr.decode()}")

        if os.path.exists(concat_file):
            os.remove(concat_file)

        return output_path

    def generate_video_dynamically(self, initial_prompt, script_generator, output_dir):
        """
        Runs the full video generation pipeline and saves all JSON configs, segments, and final video to output_dir.
        Supports segment-level resuming.

        The LLM outputs a content-adaptive num_frames per segment. Each generator
        snaps it to a valid value for its model. Other parameters (resolution,
        fps, negative_prompt, sampling parameters) use official model defaults.
        """
        os.makedirs(output_dir, exist_ok=True)
        segments_dir = os.path.join(output_dir, "segments")
        os.makedirs(segments_dir, exist_ok=True)

        print(f"\n==============================================")
        print(f"[{self.model_name}] Starting generation for: {initial_prompt}")
        print(f"[{self.model_name}] Outputs will be saved to: {output_dir}")
        print(f"==============================================\n")

        # Step 1: Get the outline
        outline_path = os.path.join(output_dir, "outline.json")
        if os.path.exists(outline_path):
            print(f"[{self.model_name}] Outline found. Loading existing outline...")
            with open(outline_path, "r", encoding="utf-8") as f:
                outline = json.load(f)
        else:
            print(f"[{self.model_name}] Generating outline...")
            outline = script_generator.generate_outline(initial_prompt)

            # Guard: handle JSON parsing fallback or unformatted string
            if isinstance(outline, str):
                print("Failed to decode outline as JSON. Aborting.")
                return None

            # Save Outline
            with open(outline_path, "w", encoding="utf-8") as f:
                json.dump(outline, f, indent=4, ensure_ascii=False)

        outline_steps = outline.get('outline_steps', [])
        required_props = outline.get('required_subjects_and_props', '')

        if not outline_steps:
            print("Failed to generate outline steps.")
            return None

        print(f"[{self.model_name}] Outline successful. Total Segments Planned: {len(outline_steps)}")
        results = []
        segment_prompts = []

        # Step 2: Segment 1
        seg1_dir = os.path.join(segments_dir, "segment_01")
        seg1_config_path = os.path.join(seg1_dir, "config.json")
        seg1_video_path = os.path.join(seg1_dir, "video.mp4")

        os.makedirs(seg1_dir, exist_ok=True)

        if os.path.exists(seg1_config_path):
            print(f"\n[{self.model_name}] Segment 1 config found. Loading existing config...")
            with open(seg1_config_path, "r", encoding="utf-8") as f:
                seg1_config = json.load(f)
            segments_list = seg1_config.get('segments', [{}])
            seg1 = segments_list[0] if segments_list else {}
            seg1_prompt = seg1.get('prompt', '')
            seg1_num_frames = seg1.get('num_frames')
            segment_prompts.append(seg1_prompt)
        else:
            print(f"\n[{self.model_name}] Generating configuration for Segment 1...")
            ref_frames = compute_reference_num_frames(
                duration_seconds=outline_steps[0].get("duration_seconds", 4.0),
                fps=self.fps,
                max_num_frames=self.max_num_frames,
            )
            seg1_config = script_generator.generate_first_segment_config(initial_prompt, outline_steps[0], required_props, reference_num_frames=ref_frames)
            with open(seg1_config_path, "w", encoding="utf-8") as f:
                json.dump(seg1_config, f, indent=4, ensure_ascii=False)
            segments_list = seg1_config.get('segments', [{}])
            seg1 = segments_list[0] if segments_list else {}
            seg1_prompt = seg1.get('prompt', '')
            seg1_num_frames = seg1.get('num_frames')
            segment_prompts.append(seg1_prompt)

        base_seed = outline.get('seed', 42)
        if os.path.exists(seg1_video_path):
            print(f"[{self.model_name}] Segment 1 video already exists. Skipping generation...")
            video_segment = seg1_video_path
        else:
            print(f"[{self.model_name}] Generating video for Segment 1...")
            video_segment = self.generate_segment(
                prompt=seg1_prompt,
                num_frames=seg1_num_frames,
                prev_frame=None,
                output_path=seg1_video_path,
                seed=base_seed
            )

        results.append(video_segment)

        prev_video_path = video_segment
        prev_frame = self.extract_last_frame(video_segment)

        identity_marker = seg1_config.get('identity_marker', '')
        continuity_anchors = seg1_config.get('continuity_anchors', '')

        # Step 3: Segments 2..N
        for i in range(1, len(outline_steps)):
            step = outline_steps[i]
            is_final = (i == len(outline_steps) - 1)

            seg_dir = os.path.join(segments_dir, f"segment_{i+1:02d}")
            seg_config_path = os.path.join(seg_dir, "config.json")
            seg_video_path = os.path.join(seg_dir, "video.mp4")

            os.makedirs(seg_dir, exist_ok=True)

            # Save the utilized previous frame in the current segment's folder
            if prev_frame is not None:
                prev_frame_path = os.path.join(seg_dir, "reference_frame.jpg")
                if not os.path.exists(prev_frame_path):
                    prev_frame.save(prev_frame_path)

            if os.path.exists(seg_config_path):
                print(f"\n[{self.model_name}] Segment {i+1} config found. Loading existing config...")
                with open(seg_config_path, "r", encoding="utf-8") as f:
                    next_seg_info = json.load(f)
                if isinstance(next_seg_info, list):
                    next_seg_info = next_seg_info[0] if next_seg_info else {}
                prompt = next_seg_info.get('prompt', '')
                num_frames = next_seg_info.get('num_frames')
                segment_prompts.append(prompt)
            else:
                print(f"\n[{self.model_name}] Generating prompt for Segment {i+1}...")
                ref_frames = compute_reference_num_frames(
                    duration_seconds=step.get("duration_seconds", 4.0),
                    fps=self.fps,
                    max_num_frames=self.max_num_frames,
                )
                next_seg_info = script_generator.generate_next_segment_prompt(
                    initial_prompt=initial_prompt,
                    current_idx=i+1,
                    identity_marker=identity_marker,
                    continuity_anchors=continuity_anchors,
                    outline_step=step,
                    is_final_step=is_final,
                    previous_video_path=prev_video_path,
                    reference_num_frames=ref_frames,
                )

                with open(seg_config_path, "w", encoding="utf-8") as f:
                    json.dump(next_seg_info, f, indent=4, ensure_ascii=False)

                prompt = next_seg_info.get('prompt', '')
                num_frames = next_seg_info.get('num_frames')
                segment_prompts.append(prompt)

            base_seed = outline.get('seed', 42)
            if os.path.exists(seg_video_path):
                print(f"[{self.model_name}] Segment {i+1} video already exists. Skipping generation...")
                video_segment = seg_video_path
            else:
                print(f"[{self.model_name}] Generating video for Segment {i+1}...")
                video_segment = self.generate_segment(
                    prompt=prompt,
                    num_frames=num_frames,
                    prev_frame=prev_frame,
                    output_path=seg_video_path,
                    seed=base_seed
                )

            results.append(video_segment)

            prev_video_path = video_segment
            prev_frame = self.extract_last_frame(video_segment)

            if next_seg_info.get('should_stop', False):
                print(f"[{self.model_name}] Auto-Stop condition met at Segment {i+1}.")
                break
        # Save all segment prompts to segment_prompts.json
        segment_prompts_path = os.path.join(output_dir, "segment_prompts.json")
        with open(segment_prompts_path, "w", encoding="utf-8") as f:
            json.dump(segment_prompts, f, indent=4, ensure_ascii=False)

        print(f"\n[{self.model_name}] All {len(results)} segments generated. Concatenating...")
        final_video_path = os.path.join(output_dir, "final_video.mp4")
        return self.concatenate_segments(results, final_video_path)

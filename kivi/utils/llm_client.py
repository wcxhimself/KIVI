import os
import json
import base64
import time
import random
import subprocess
from pathlib import Path
from openai import OpenAI

class LLMClient:
    def __init__(self, model="google/gemini-3.1-pro-preview", api_key=None, base_url=None):
        self.model = model

        self.base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL")
        )
        self.api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
        )

        headers = {}
        if "openrouter" in (self.base_url or ""):
            referer = os.environ.get("LLM_HTTP_REFERER", "")
            title = os.environ.get("LLM_HTTP_TITLE", "")
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=headers or None,
        )

    def call_text(self, prompt, system_prompt=None, response_format="text", **kwargs):
        """
        Calls the LLM for text generation (e.g., scripting, verification).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_retries = kwargs.pop('max_retries', 8)
        # <-- ADDED: set default max_tokens to avoid exhausting quota during long reasoning
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 12288

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"} if response_format == "json" else {"type": "text"},
                    **kwargs
                )
                content = response.choices[0].message.content
                if content is None:
                    msg = response.choices[0].message
                    raise Exception(f"Model returned empty content. Full message: {msg}")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                # Exponential backoff with jitter and a capped maximum wait time
                wait_time = min(60, (2 ** attempt) + random.uniform(0, 1))
                print(f"API call failed: {e}. Retrying ({attempt + 1}/{max_retries}) in {wait_time}s...")
                time.sleep(wait_time)
        
        if response_format == "json":
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        return content


    @staticmethod
    def _compress_video(video_path, target_size_mb=7, min_crf=18, max_crf=32):
        """
        Compress video for API transmission using HEVC encoding.
        Adjust CRF adaptively to try keep compressed size below target_size_mb.
        """
        video_path = Path(video_path)
        compressed_path = video_path.with_stem(video_path.stem + "_compressed")

        if compressed_path.exists():
            return str(compressed_path)

        original_size = video_path.stat().st_size / (1024 * 1024)
        # heuristic initial CRF
        if original_size < target_size_mb * 0.7:
            crf = min_crf
        elif original_size < target_size_mb * 1.5:
            crf = 22
        elif original_size < target_size_mb * 3:
            crf = 25
        else:
            crf = 28

        # Try compressing and increase CRF iteratively if needed
        for try_crf in [crf, crf+2, crf+4, max_crf]:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                '-c:v', 'libx264',
                "-crf", str(try_crf),
                "-an",
                str(compressed_path)
            ]
            print(f"[LLMClient] Compressing {video_path.name} -> {compressed_path.name} (CRF={try_crf})")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            compressed_size = compressed_path.stat().st_size / (1024 * 1024)
            print(f"[LLMClient] Size: {original_size:.1f}MB -> {compressed_size:.1f}MB")
            if compressed_size <= target_size_mb or try_crf == max_crf:
                break
        return str(compressed_path)

    def call_vision(self, prompt, video_path, system_prompt=None, response_format="text", **kwargs):
        """
        Calls the LLM by passing the video file directly for vision tasks.
        The video is compressed (HEVC, CRF=25) before sending to stay within API size limits.
        Original video is never modified.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Always compress to standard H.264 for Gemini compatibility
        if video_path is None:
            # If there's no video to send, just call text
            # Reusing call_text logic or fallback
            return self.call_text(prompt, system_prompt=system_prompt, response_format=response_format, **kwargs)

        max_raw_size_mb = 0  # force compression for all videos
        raw_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
        video_to_send = self._compress_video(video_path, target_size_mb=7)

        with open(video_to_send, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        
        user_content = [
            {"type": "text", "text": prompt},
            {
                "type": "video_url",
                "video_url": {
                    "url": f"data:video/mp4;base64,{video_base64}"
                }
            }
        ]
            
        messages.append({"role": "user", "content": user_content})

        max_retries = kwargs.pop('max_retries', 8)
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"} if response_format == "json" else {"type": "text"},
                    **kwargs
                )
                content = response.choices[0].message.content
                if content is None:
                    msg = response.choices[0].message
                    raise Exception(f"Model returned empty content. Full message: {msg}")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                # Exponential backoff with jitter and a capped maximum wait time
                wait_time = min(60, (2 ** attempt) + random.uniform(0, 1))
                print(f"API call vision failed: {e}. Retrying ({attempt + 1}/{max_retries}) in {wait_time}s...")
                time.sleep(wait_time)
        
        if response_format == "json":
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        return content

    def call_multimodal(self, content_blocks, system_prompt=None, response_format="text", **kwargs):
        """
        Calls the LLM with arbitrary multimodal content blocks.

        Args:
            content_blocks: List of dicts, each with:
                - type: "text" | "image" | "video"
                - text: str (for type="text")
                - path: str (file path, for type="image" or "video")
            system_prompt: Optional system message
            response_format: "json" or "text"

        Example:
            content_blocks = [
                {"type": "text", "text": "Verify these claims:"},
                {"type": "text", "text": "Claim 1: ..."},
                {"type": "image", "path": "frame_1.jpg"},
                {"type": "text", "text": "Claim 2: ..."},
                {"type": "image", "path": "frame_2.jpg"},
            ]
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        for block in content_blocks:
            if block["type"] == "text":
                user_content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                with open(block["path"], "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode('utf-8')
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            elif block["type"] == "video":
                with open(block["path"], "rb") as f:
                    vid_b64 = base64.b64encode(f.read()).decode('utf-8')
                user_content.append({
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{vid_b64}"}
                })
            else:
                raise ValueError(f"Unknown content block type: {block['type']}")

        messages.append({"role": "user", "content": user_content})

        max_retries = kwargs.pop('max_retries', 8)
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 12288

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"} if response_format == "json" else {"type": "text"},
                    **kwargs
                )
                content = response.choices[0].message.content
                if content is None:
                    msg = response.choices[0].message
                    raise Exception(f"Model returned empty content. Full message: {msg}")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = min(60, (2 ** attempt) + random.uniform(0, 1))
                print(f"Multimodal API call failed: {e}. Retrying ({attempt + 1}/{max_retries}) in {wait_time}s...")
                time.sleep(wait_time)

        if response_format == "json":
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        return content
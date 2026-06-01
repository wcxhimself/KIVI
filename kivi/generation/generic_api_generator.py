import os
import time
import json
import base64
import cv2
import requests
from io import BytesIO
from pathlib import Path
from typing import Optional

from .video_generator import BaseVideoGenerator


def _resolve_dot_path(obj: dict, path: str):
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    return obj


class GenericAPIGenerator(BaseVideoGenerator):
    def __init__(self, config, **kwargs):
        super().__init__(config.name, **kwargs)
        self.config = config
        self.cfg = config.api
        self.segment_counter = 0

        self.env_key = self.cfg.get("env_key", "")
        self.api_key = os.environ.get(self.env_key, "")
        if not self.api_key:
            raise ValueError(f"Environment variable {self.env_key} is required for {config.name}")

        self.base_url = self.cfg.get("base_url", "")
        self.auth_header = self.cfg.get("auth_header", "Bearer {api_key}")
        self.session = requests.Session()
        self.session.proxies = {"http": "", "https": ""}

    def _request_headers(self):
        auth_value = self.auth_header.format(api_key=self.api_key)
        if auth_value.lower().startswith("bearer"):
            return {"Authorization": auth_value}
        if ":" in auth_value:
            key, val = auth_value.split(":", 1)
            return {key.strip(): val.strip()}
        return {"X-API-Key": auth_value}

    def _build_body(self, template, prompt: str, duration: int, seed: int,
                    prev_frame=None):
        """Recursively build request body, substituting placeholders in all string values."""
        buf = BytesIO()
        if prev_frame is not None:
            prev_frame.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8") if prev_frame else ""

        def _resolve(obj):
            if isinstance(obj, str):
                obj = obj.format(
                    prompt=prompt,
                    duration=duration,
                    seed=seed,
                    image_base64=img_b64,
                    data_uri=f"data:image/png;base64,{img_b64}",
                )
                try:
                    return int(obj)
                except ValueError:
                    try:
                        return float(obj)
                    except ValueError:
                        pass
                return obj
            if isinstance(obj, dict):
                return {k: _resolve(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_resolve(item) for item in obj]
            return obj

        return _resolve(template)

    def _submit(self, ep: dict, prompt: str, duration: int, seed: int,
                prev_frame=None) -> str:
        url = self.base_url.rstrip("/") + "/" + ep["url"].lstrip("/")
        headers = self._request_headers()
        extra_headers = ep.get("headers", {})
        if isinstance(extra_headers, dict):
            for k, v in extra_headers.items():
                headers[k] = v
        async_header = ep.get("async_header", "")
        if async_header:
            key, val = async_header.split(":", 1)
            headers[key.strip()] = val.strip()

        body = self._build_body(ep.get("body", {}), prompt, duration, seed, prev_frame)

        resp = self.session.post(url, headers=headers, json=body)
        if not resp.ok:
            raise Exception(f"API submit failed [{resp.status_code}]: {resp.text[:500]}")
        data = resp.json()
        tid = _resolve_dot_path(data, ep.get("task_id_path", "id"))
        if not tid:
            raise Exception(f"Could not find task_id in response: {data}")
        print(f"  [{self.model_name}] Task submitted: {tid}")
        return tid

    def _poll(self, ep: dict, task_id: str, output_path: str):
        url_template = self.base_url.rstrip("/") + "/" + ep["url"].lstrip("/")
        interval = ep.get("interval", 10)
        timeout = ep.get("timeout", 3600)
        status_path = ep.get("status_path", "status")
        success_values = ep.get("success_values", ["completed", "succeeded", "SUCCEEDED"])
        video_url_path = ep.get("video_url_path", "video_url")
        headers = self._request_headers()

        waited = 0
        while waited < timeout:
            time.sleep(interval)
            waited += interval
            url = url_template.format(task_id=task_id)
            resp = self.session.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            status = _resolve_dot_path(data, status_path)
            print(f"  [{self.model_name}] Polling {task_id}: {status} ({waited}s)")

            if status in success_values:
                video_url = _resolve_dot_path(data, video_url_path)
                if not video_url:
                    raise Exception(f"No video_url in response: {data}")
                print(f"  [{self.model_name}] Downloading video ...")
                self._download(video_url, output_path)
                return output_path

            failed_values = ep.get("failed_values", ["failed", "FAILED", "canceled", "expired"])
            if status in failed_values:
                raise Exception(f"Task {task_id} failed with status: {status}")

        raise Exception(f"Task {task_id} timed out after {timeout}s")

    def _download(self, video_url: str, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        for attempt in range(3):
            resp = self.session.get(video_url, stream=True)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            cap = cv2.VideoCapture(str(output_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count > 0:
                print(f"  [{self.model_name}] Video saved: {output_path}")
                return
            os.remove(output_path)
            print(f"  [{self.model_name}] Retry download ({attempt + 1}/3)...")
        raise Exception(f"Failed to download valid video: {video_url}")

    def generate_segment(self, prompt, num_frames=None, prev_frame=None,
                         output_path=None, seed=42):
        self.segment_counter += 1
        is_i2v = prev_frame is not None

        if output_path is None:
            os.makedirs("outputs", exist_ok=True)
            output_path = f"outputs/{self.model_name}_segment_{self.segment_counter}.mp4"
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        min_dur = self.cfg.get("min_duration", 2)
        max_dur = self.cfg.get("max_duration", 15)
        duration = max(min_dur, min(max_dur, round((num_frames or 81) / self.config.fps)))
        computed_seed = seed + self.segment_counter

        key = "i2v" if is_i2v else "t2v"
        ep = self.cfg.get(key, self.cfg.get("t2v", {}))
        if not ep:
            raise ValueError(f"No '{key}' endpoint configured in YAML")

        task_id = self._submit(ep, prompt, duration, computed_seed, prev_frame)
        return self._poll(self.cfg["poll"], task_id, output_path)
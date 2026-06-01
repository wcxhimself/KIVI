"""Wrapper for LongCat-Video interactive generation. Reads interactive_prompts.json."""
import os
import sys
import json
import argparse
import datetime
import subprocess
import PIL.Image
import numpy as np

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from torchvision.io import write_video

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--enable_compile", action="store_true")
    args = parser.parse_args()

    with open(args.prompt_file, "r") as f:
        seg_data = json.load(f)
    prompt_list = [s["prompt"] for s in seg_data]

    rank = int(os.environ["RANK"])
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))

    init_context_parallel(context_parallel_size=args.context_parallel_size,
                          global_rank=dist.get_rank(),
                          world_size=dist.get_world_size())

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, subfolder="tokenizer")
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    dit = LongCatVideoTransformer3DModel.from_pretrained(args.checkpoint_dir, subfolder="dit", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler()

    pipeline = LongCatVideoPipeline(
        tokenizer=tokenizer, vae=vae, transformer=dit, scheduler=scheduler,
        enable_compile=args.enable_compile,
    )

    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
        "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
        "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, messy background, three legs, "
        "many people in the background, walking backwards"
    )

    num_cond_frames = 13
    spatial_refine_only = False

    all_frames = []
    ref_image = None

    for i, prompt in enumerate(prompt_list):
        is_final = (i == len(prompt_list) - 1)
        num_frames = seg_data[i].get("num_frames", 93)
        num_gen_frames = max(1, num_frames - num_cond_frames)

        print(f"  Segment {i}: {num_gen_frames} new frames, prompt='{prompt[:60]}...'")

        gen_frames = pipeline(
            prompt=prompt,
            image=ref_image,
            num_frames=num_gen_frames,
            num_cond_frames=num_cond_frames,
            negative_prompt=negative_prompt,
            spatial_refine_only=spatial_refine_only,
        )

        if ref_image is not None:
            all_frames.extend(gen_frames)
        else:
            all_frames = gen_frames

        ref_image = gen_frames[-num_cond_frames:]

        if is_final and spatial_refine_only:
            ref_frames = pipeline(
                prompt=prompt,
                image=ref_image,
                num_frames=num_gen_frames,
                num_cond_frames=0,
                negative_prompt=negative_prompt,
                spatial_refine_only=True,
            )
            all_frames.extend(ref_frames)

        if dist.get_rank() == 0:
            torch_gc = lambda: (torch.cuda.empty_cache(), torch.cuda.ipc_collect())
            torch_gc()

    if dist.get_rank() == 0:
        clip = [((f.cpu() * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()) for f in all_frames]
        start_idx = max(i for i, _ in enumerate(clip) if (clip[i] > 1).any())
        output_path = os.path.join(args.output_dir, "output_interactive_refine_0.mp4")
        write_video(output_path, clip[start_idx:], fps=15)
        print(f"  Saved: {output_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
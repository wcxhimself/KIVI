import os
import sys
import json
import argparse
import datetime

import torch
import torch.distributed as dist
import PIL.Image
import numpy as np
from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video

LONGCAT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "video_generation_models", "LongCat-Video")
sys.path.insert(0, LONGCAT_DIR)

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--enable_compile", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, subfolder="tokenizer",
                                              torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(args.checkpoint_dir, subfolder="text_encoder",
                                                    torch_dtype=torch.bfloat16).to(local_rank)
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint_dir, subfolder="vae",
                                           torch_dtype=torch.bfloat16).to(local_rank)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.checkpoint_dir, subfolder="scheduler")
    dit = LongCatVideoTransformer3DModel.from_pretrained(args.checkpoint_dir, subfolder="dit",
                                                         cp_split_hw=cp_split_hw,
                                                         torch_dtype=torch.bfloat16).to(local_rank)

    if args.enable_compile:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler, dit=dit,
    )
    pipe.to(local_rank)

    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
        "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
        "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, messy background, three legs, "
        "many people in the background, walking backwards"
    )

    num_cond_frames = int(seg_data[0]["num_frames"] * 0.14)
    seed = 42 + dist.get_rank()
    generator = torch.Generator(device=local_rank).manual_seed(seed)

    output = pipe.generate_t2v(
        prompt=prompt_list[0],
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=seg_data[0]["num_frames"],
        num_inference_steps=50,
        guidance_scale=4.0,
        generator=generator,
    )[0]

    all_frames = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
    all_frames = [PIL.Image.fromarray(img) for img in all_frames]
    current_video = all_frames
    del output
    torch_gc()

    num_segments = len(prompt_list) - 1
    for seg_idx in range(num_segments):
        nf = seg_data[seg_idx + 1]["num_frames"]
        nf += num_cond_frames
        nc = int(nf * 0.14)

        if local_rank == 0:
            print(f"  Segment {seg_idx + 1}/{num_segments}: {nf} frames ({nc} cond)")

        output = pipe.generate_vc(
            video=current_video,
            prompt=prompt_list[seg_idx + 1],
            negative_prompt=negative_prompt,
            resolution="480p",
            num_frames=nf,
            num_cond_frames=nc,
            num_inference_steps=50,
            guidance_scale=4.0,
            generator=generator,
            use_kv_cache=True,
            offload_kv_cache=False,
            enhance_hf=True,
        )[0]

        out_frames = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        out_frames = [PIL.Image.fromarray(img) for img in out_frames]
        all_frames.extend(out_frames)
        current_video = out_frames
        del output
        torch_gc()

    if local_rank == 0:
        clip = [(np.array(f)).astype(np.uint8) for f in all_frames]
        clip = torch.from_numpy(np.stack(clip))
        output_path = os.path.join(args.output_dir, "output_interactive_refine_0.mp4")
        write_video(output_path, clip, fps=15, video_codec="libx264", options={"crf": "18"})
        total_dur = len(clip) / 15
        print(f"  Saved: {output_path} ({len(clip)} frames, {total_dur:.1f}s)")


if __name__ == "__main__":
    main()
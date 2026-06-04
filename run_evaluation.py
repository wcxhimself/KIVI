import json
import os
import re
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

from kivi.generation.factory import list_available_models

AVAILABLE_MODELS = list_available_models()
AVAILABLE_NAMES = list(AVAILABLE_MODELS.keys())
MODEL_HELP_LINES = "\n        - ".join(
    [""] + [f"{name}: {desc}" for name, desc in AVAILABLE_MODELS.items()]
)

STEPS = ["all", "script", "generate", "extract", "verify", "score"]

parser = argparse.ArgumentParser(
    description="KIVI: Knowledge-Intensive Video Generation",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help=f"Model name to evaluate:{MODEL_HELP_LINES}\n    "
         "Not required with --video-path.",
)
parser.add_argument(
    "--step",
    type=str,
    default="all",
    choices=STEPS,
    help="Pipeline step to run. 'all' runs the full pipeline. "
         "Individual steps can be run independently using cached outputs.",
)
parser.add_argument(
    "--category",
    type=str,
    default=None,
    help="Filter by category name (e.g., 'Cars_Other_Vehicles'). If not set, runs all categories.",
)
parser.add_argument(
    "--prompt-index",
    type=int,
    default=None,
    help="Filter to a specific prompt index within the category (1-based). Requires --category.",
)
parser.add_argument(
    "--list-models",
    action="store_true",
    help="List all available model configurations and exit.",
)
parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
parser.add_argument(
    "--video-path",
    type=str,
    default=None,
    help="Evaluate a single user-provided video. Skips script/generate. Requires --prompt.",
)
parser.add_argument(
    "--prompt",
    type=str,
    default=None,
    help="Prompt text for --video-path mode. Required when --video-path is set.",
)
parser.add_argument(
    "--prompts-json",
    type=str,
    default=os.path.join(PROJECT_ROOT, "experiment_prompts.json"),
    help="Path to the prompts json file.",
)
args = parser.parse_args()

if args.list_models:
    print("Available models:\n")
    for name, desc in sorted(AVAILABLE_MODELS.items()):
        print(f"  {name:<30s}  {desc}")
    print(f"\nTotal: {len(AVAILABLE_MODELS)} model(s)")
    print(f"\nTo add a new model, create a YAML file in configs/")
    exit(0)

if args.video_path:
    if not args.prompt:
        parser.error("--prompt is required when --video-path is set")
    if args.step in ("script", "generate"):
        parser.error(f"--step {args.step} is not supported with --video-path")
    if not os.path.exists(args.video_path):
        parser.error(f"Video file not found: {args.video_path}")

if not args.video_path and not args.model:
    parser.error("--model is required (unless using --video-path)")

if not args.model:
    args.model = "custom"

HEAVY_STEPS = {"all", "generate"}
LAZY_IMPORTS = {}

def _get_import(module_path, name):
    if name not in LAZY_IMPORTS:
        LAZY_IMPORTS[name] = __import__(module_path, fromlist=[name])
    return LAZY_IMPORTS[name]

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def sanitize_name(name):
    sanitized = re.sub(r"[^a-zA-Z0-9\s]", "", str(name))
    return re.sub(r"\s+", "_", sanitized.strip())[:100]


def _resolve_output_dirs(model_name, category_name, q_idx, question):
    safe_category = sanitize_name(category_name)
    safe_question = sanitize_name(question)
    common_outline_dir = os.path.join(OUTPUT_ROOT, "common_outlines", safe_category,
                                      f"Q{q_idx+1}_{safe_question}")
    common_script_dir = os.path.join(OUTPUT_ROOT, "common_script", safe_category,
                                     f"Q{q_idx+1}_{safe_question}")
    model_output_dir = os.path.join(OUTPUT_ROOT, model_name, safe_category,
                                    f"Q{q_idx+1}_{safe_question}")
    return common_outline_dir, common_script_dir, model_output_dir


def step_script(script_generator, question, common_outline_dir, common_script_dir):
    """Generate outline and full script. Shared across models."""
    os.makedirs(common_outline_dir, exist_ok=True)
    os.makedirs(common_script_dir, exist_ok=True)

    outline_path = os.path.join(common_outline_dir, "outline.json")
    if os.path.exists(outline_path):
        print("  -> Loaded existing outline.")
        with open(outline_path, "r", encoding="utf-8") as f:
            outline = json.load(f)
    else:
        print("  -> Generating outline ...")
        outline = script_generator.generate_outline(question)
        if isinstance(outline, str):
            raise ValueError(f"Outline generation returned a string instead of JSON: {outline[:200]}")
        with open(outline_path, "w", encoding="utf-8") as f:
            json.dump(outline, f, indent=4, ensure_ascii=False)

    script_path = os.path.join(common_script_dir, "script.json")
    if os.path.exists(script_path):
        print("  -> Loaded existing full script.")
        with open(script_path, "r", encoding="utf-8") as f:
            full_script = json.load(f)
    else:
        print("  -> Generating full script from outline ...")
        full_script = script_generator.generate_full_script(question, outline)
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(full_script, f, indent=4, ensure_ascii=False)
    print("  -> Full script saved.")
    return outline


def step_generate(generator, script_generator, question, outline, model_output_dir):
    """Generate video for a specific model."""
    os.makedirs(model_output_dir, exist_ok=True)
    with open(os.path.join(model_output_dir, "outline.json"), "w", encoding="utf-8") as f:
        json.dump(outline, f, indent=4, ensure_ascii=False)

    final_video_path = os.path.join(model_output_dir, "final_video.mp4")
    if os.path.exists(final_video_path):
        print(f"  -> Skipping video generation (already exists): {question}")
        return final_video_path

    print(f"  -> Generating video for: {question}")
    generator.generate_video_dynamically(
        initial_prompt=question,
        script_generator=script_generator,
        output_dir=model_output_dir,
    )
    return final_video_path


def step_extract(extractor, final_video_path, question, model_output_dir):
    """Extract factual claims from a generated video."""
    if not os.path.exists(final_video_path):
        print(f"  -> [SKIP] No video found at {final_video_path}. Run --step generate first.")
        return None

    eval_dir = os.path.join(model_output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    claims_path = os.path.join(eval_dir, "extracted_claims.json")

    if os.path.exists(claims_path):
        print("  -> Claims already extracted.")
        with open(claims_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("  -> Extracting claims ...")
    claims = extractor.extract(video_path=final_video_path, question=question)
    with open(claims_path, "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=4, ensure_ascii=False)
    return claims


def step_verify(verifier, claims, model_output_dir):
    """Verify extracted claims against evidence."""
    if claims is None:
        print("  -> [SKIP] No claims to verify. Run --step extract first.")
        return

    eval_dir = os.path.join(model_output_dir, "evaluation")
    verification_path = os.path.join(eval_dir, "verification_results.json")

    if os.path.exists(verification_path):
        print("  -> Claims already verified.")
        return

    print("  -> Verifying claims ...")
    verification_result = verifier.verify(claims_json=claims)
    with open(verification_path, "w", encoding="utf-8") as f:
        json.dump(verification_result, f, indent=4, ensure_ascii=False)


def step_score(model_output_dir, model_name, category, question,
               script_generator_model, extractor_model, verifier_model, helpfulness_model):
    """Compute final scores from cached verification and helpfulness data."""
    eval_dir = os.path.join(model_output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    score_path = os.path.join(eval_dir, "score.json")

    if os.path.exists(score_path):
        with open(score_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
        print(f"  -> Scores (cached): factual_precision={scores.get('factual_precision')}%, "
              f"helpfulness={scores.get('helpfulness_score')}%")
        return scores

    verification_path = os.path.join(eval_dir, "verification_results.json")
    factual_precision = None
    verification_info = {
        "total_claims": None, "correct": None, "uncertain": None, "incorrect": None,
    }

    if os.path.exists(verification_path):
        with open(verification_path, "r", encoding="utf-8") as f:
            vr = json.load(f)

        if isinstance(vr, dict) and "verification_summary" in vr:
            claim_results = vr.get("claim_results", [])
            correct = sum(1 for r in claim_results if r.get("verdict") == "Correct")
            incorrect = sum(1 for r in claim_results if r.get("verdict") == "Incorrect")
            uncertain = sum(1 for r in claim_results if r.get("verdict") == "Uncertain")
            total = correct + incorrect + uncertain
            verification_info.update(
                total_claims=total, correct=correct, incorrect=incorrect, uncertain=uncertain,
            )
            if total > 0:
                factual_precision = correct / total
        else:
            correct = vr.get("correct", 0) if isinstance(vr, dict) else 0
            uncertain = vr.get("uncertain", 0) if isinstance(vr, dict) else 0
            incorrect = vr.get("incorrect", 0) if isinstance(vr, dict) else 0
            total = correct + uncertain + incorrect
            verification_info.update(
                total_claims=total, correct=correct, incorrect=incorrect, uncertain=uncertain,
            )
            if total > 0:
                factual_precision = correct / total

    helpfulness_path = os.path.join(eval_dir, "helpfulness_score.json")
    helpfulness_score = None
    if os.path.exists(helpfulness_path):
        with open(helpfulness_path, "r", encoding="utf-8") as f:
            hr = json.load(f)
        helpfulness_score = hr.get("helpfulness_score") if isinstance(hr, dict) else (
            hr if isinstance(hr, (int, float)) else None)

    factual_precision_pct = round(factual_precision * 100, 2) if factual_precision is not None else None
    helpfulness_pct = round(helpfulness_score * 10, 2) if helpfulness_score is not None else None

    score_data = {
        "category": category,
        "prompt": question,
        "video_generation_model": model_name,
        "script_generator_model": script_generator_model,
        "claim_extractor_model": extractor_model,
        "claim_verifier_model": verifier_model,
        "helpfulness_model": helpfulness_model,
        "factual_precision": factual_precision_pct,
        "factual_precision_unit": "%",
        "helpfulness_score": helpfulness_pct,
        "helpfulness_score_unit": "% (out of 10 points)",
        "verification_total_claims": verification_info["total_claims"],
        "verification_correct": verification_info["correct"],
        "verification_uncertain": verification_info["uncertain"],
        "verification_incorrect": verification_info["incorrect"],
    }
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(score_data, f, indent=4, ensure_ascii=False)
    print(f"  -> Scores saved: factual_precision={factual_precision_pct}%, helpfulness={helpfulness_pct}%")
    return score_data


def main_custom_video(model_name, step_name, llm_model):
    """Evaluate a single user-provided video. Saves evaluation results next to the video."""
    video_path = os.path.abspath(args.video_path)
    video_dir = os.path.dirname(video_path)
    prompt = args.prompt

    extractor = None
    verifier = None
    helpfulness_eval = None

    if step_name in ("all", "extract"):
        from kivi.evaluation.claim_extraction import ExtractClaims
        extractor = ExtractClaims(mllm_model=llm_model)

    if step_name in ("all", "verify"):
        from kivi.evaluation.claim_verification import VerifyClaims
        verifier = VerifyClaims(mllm_model=llm_model)

    if step_name in ("all",):
        from kivi.evaluation.helpfulness_evaluation import HelpfulnessEvaluator
        helpfulness_eval = HelpfulnessEvaluator(mllm_model=llm_model)

    print(f"\n=== Custom Video Evaluation ===")
    print(f"  Video : {video_path}")
    print(f"  Prompt: {prompt}")

    # claims
    claims = None
    if step_name in ("all", "extract"):
        claims = step_extract(extractor, video_path, prompt, video_dir)
    else:
        claims_path = os.path.join(video_dir, "evaluation", "extracted_claims.json")
        if os.path.exists(claims_path):
            with open(claims_path, "r", encoding="utf-8") as f:
                claims = json.load(f)

    # verification
    if step_name in ("all", "verify"):
        step_verify(verifier, claims, video_dir)

    # helpfulness
    helpfulness_path = os.path.join(video_dir, "evaluation", "helpfulness_score.json")
    if step_name == "all" and not os.path.exists(helpfulness_path):
        print("  -> Evaluating helpfulness ...")
        helpfulness = helpfulness_eval.evaluate(video_path, prompt)
        with open(helpfulness_path, "w", encoding="utf-8") as f:
            json.dump(helpfulness, f, indent=4, ensure_ascii=False)

    # score
    if step_name in ("all", "score"):
        step_score(video_dir, model_name, "custom", prompt,
                   llm_model, llm_model, llm_model, llm_model)

    print(f"  -> Results saved to: {video_dir}/evaluation/")


def main():
    model_name = args.model
    step_name = args.step
    llm_model = "google/gemini-3.1-pro-preview"

    # ── custom video mode ─────────────────────────────────
    if args.video_path:
        return main_custom_video(model_name, step_name, llm_model)

    # ── standard evaluation mode ──────────────────────────
    with open(args.prompts_json, "r") as f:
        categories = json.load(f)

    master_path = os.path.join(PROJECT_ROOT, "experiment_prompts.json")
    if args.prompts_json != master_path:
        with open(master_path, "r") as f:
            master_categories = json.load(f)
        prompt_lookup = {}
        for cat in master_categories:
            for idx, q in enumerate(cat["prompts"]):
                prompt_lookup[q.strip()] = (cat["category"], idx)
    else:
        prompt_lookup = None

    model_name = args.model
    step_name = args.step

    llm_model = "google/gemini-3.1-pro-preview"

    script_generator = None
    generator = None
    extractor = None
    verifier = None
    helpfulness_eval = None

    if step_name in ("all", "script", "generate"):
        from kivi.generation.factory import create_video_generator
        from kivi.generation.script_generator import DynamicScriptGenerator
        script_generator = DynamicScriptGenerator(model_name=llm_model)
        generator = create_video_generator(model_name, gpu=args.gpu)

    if step_name in ("all", "extract"):
        from kivi.evaluation.claim_extraction import ExtractClaims
        extractor = ExtractClaims(mllm_model=llm_model)

    if step_name in ("all", "verify"):
        from kivi.evaluation.claim_verification import VerifyClaims
        verifier = VerifyClaims(mllm_model=llm_model)

    if step_name in ("all",):
        from kivi.evaluation.helpfulness_evaluation import HelpfulnessEvaluator
        helpfulness_eval = HelpfulnessEvaluator(mllm_model=llm_model)

    if step_name in ("all", "script"):
        CommonPromptRoot = os.path.join(OUTPUT_ROOT, "common_outlines")
        CommonScriptRoot = os.path.join(OUTPUT_ROOT, "common_script")
        os.makedirs(CommonPromptRoot, exist_ok=True)
        os.makedirs(CommonScriptRoot, exist_ok=True)

    for cat_data in categories:
        category_name = cat_data["category"]
        if args.category and category_name != args.category:
            continue

        prompts = cat_data["prompts"]
        for q_idx, prompt in enumerate(prompts):
            if args.prompt_index is not None and q_idx + 1 != args.prompt_index:
                continue

            if prompt_lookup is not None:
                lookup = prompt_lookup.get(prompt.strip())
                if lookup:
                    category_name, q_idx = lookup

            print(f"\n=== Category: {category_name} | Q{q_idx + 1}: {prompt} ===")

            outline_dir, script_dir, model_dir = _resolve_output_dirs(
                model_name, category_name, q_idx, prompt)

            outline = None

            # ---- step: script ----
            if step_name in ("all", "script"):
                outline = step_script(script_generator, prompt, outline_dir, script_dir)
            else:
                outline_path = os.path.join(model_dir, "outline.json")
                script_path = os.path.join(script_dir, "script.json")
                if os.path.exists(outline_path) and os.path.exists(script_path):
                    with open(outline_path, "r", encoding="utf-8") as f:
                        outline = json.load(f)
                else:
                    print(f"  -> [SKIP] No cached script found. Run --step script first.")
                    continue

            # ---- step: generate ----
            final_video_path = os.path.join(model_dir, "final_video.mp4")
            if step_name in ("all", "generate"):
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
                os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                final_video_path = step_generate(generator, script_generator, prompt, outline, model_dir)

                if not os.path.exists(final_video_path):
                    print(f"  -> Generation failed for {prompt}. Skipping downstream steps.")
                    continue
            elif not os.path.exists(final_video_path):
                print(f"  -> [SKIP] No video. Run --step generate first.")
                continue

            # ---- step: extract ----
            claims = None
            if step_name in ("all", "extract"):
                claims = step_extract(extractor, final_video_path, prompt, model_dir)
            else:
                claims_path = os.path.join(model_dir, "evaluation", "extracted_claims.json")
                if os.path.exists(claims_path):
                    with open(claims_path, "r", encoding="utf-8") as f:
                        claims = json.load(f)

            # ---- step: verify ----
            if step_name in ("all", "verify"):
                step_verify(verifier, claims, model_dir)

            # ---- helpfulness (all only) ----
            eval_dir = os.path.join(model_dir, "evaluation")
            helpfulness_path = os.path.join(eval_dir, "helpfulness_score.json")
            if step_name == "all" and not os.path.exists(helpfulness_path):
                print("  -> Evaluating helpfulness ...")
                helpfulness = helpfulness_eval.evaluate(video_path=final_video_path, question=prompt)
                with open(helpfulness_path, "w", encoding="utf-8") as f:
                    json.dump(helpfulness, f, indent=4, ensure_ascii=False)

            # ---- step: score ----
            if step_name in ("all", "score"):
                step_score(model_dir, model_name, category_name, prompt,
                           llm_model, llm_model, llm_model, llm_model)

            print(f"  -> Done: {prompt}")


if __name__ == "__main__":
    main()
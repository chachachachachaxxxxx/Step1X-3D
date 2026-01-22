import argparse
import os
import sys
import torch
import warnings
import yaml
from peft import LoraConfig
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline

# Suppress warnings
warnings.filterwarnings("ignore")


def load_lora_pipeline(base_model_path, lora_ckpt_path=None, device="cuda"):
    print(f"Loading base model from {base_model_path}...")
    pipeline = Step1X3DGeometryPipeline.from_pretrained(
        base_model_path, subfolder="Step1X-3D-Geometry-1300m", torch_dtype=torch.bfloat16
    ).to(device)

    if lora_ckpt_path is None:
        print("Skipping LoRA loading, returning base model pipeline.")
        return pipeline

    # Define LoRA Configuration (Must match training config)
    # Based on configs/train-geometry-diffusion/jewelry-3d-geometry.yaml
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        init_lora_weights="gaussian",
        target_modules=[
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ],
    )
    lora_config_1222 = LoraConfig(
        r=128,
        lora_alpha=128,
        init_lora_weights="gaussian",
        target_modules=[
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ],
    )

    # Add LoRA adapter to the transformer (FluxDenoiser's dit_model)
    # The pipeline.transformer is the FluxDenoiser
    # In training (RectifiedFlowSystem), it is self.denoiser_model.dit_model.add_adapter(...)
    pipeline.transformer.dit_model.add_adapter(lora_config)

    # Load converted weights
    print(f"Loading LoRA weights from {lora_ckpt_path}...")
    state_dict = torch.load(lora_ckpt_path, map_location="cpu")

    # Handle DeepSpeed checkpoint format (wrapped in 'module')
    if isinstance(state_dict, dict) and "module" in state_dict:
        print("Detected DeepSpeed checkpoint format, using 'module' key.")
        state_dict = state_dict["module"]

    # Filter and adjust keys
    # The training checkpoint (LightingModule) usually has keys prefixed with "denoiser_model."
    # We need to strip this prefix to match pipeline.transformer
    lora_state_dict = {}

    # Debug: Print first few keys to diagnose prefix
    # keys = list(state_dict.keys())
    # print("First 5 keys in ckpt:", keys[:5])

    for k, v in state_dict.items():
        if "lora" in k:  # Only load LoRA weights
            # Remove 'denoiser_model.' prefix if present (common in Lightning checkpoints)
            new_k = k.replace("denoiser_model.", "")
            # Ensure it targets dit_model if not already (training code attaches adapter to dit_model)
            if not new_k.startswith("dit_model.") and "dit_model" not in new_k:
                # If the checkpoint key is just "transformer.blocks...", we might need to adjust
                # But usually in training: self.denoiser_model = FluxDenoiser -> has .dit_model
                pass
            lora_state_dict[new_k] = v.to(torch.bfloat16)

    # Load weights into the transformer
    # strict=False because we are only loading LoRA weights, not the full model
    missing, unexpected = pipeline.transformer.load_state_dict(lora_state_dict, strict=False)

    print(f"LoRA weights loaded. Missing keys (expected for base weights): {len(missing)}")
    if len(unexpected) > 0:
        print(f"Unexpected keys: {unexpected[:5]} ...")

    return pipeline


def run_inference(image_path, save_path, pipeline):
    if not os.path.exists(image_path):
        print(f"Error: Image path {image_path} does not exist.")
        return

    print(f"Running inference on {image_path}...")
    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(2025)

    out = pipeline(
        image_path,
        guidance_scale=7.5,
        num_inference_steps=50,
        generator=generator,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out.mesh[0].export(save_path)
    print(f"Saved result to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step1X-3D LoRA inference (config driven or CLI overrides)."
    )
    parser.add_argument(
        "--config", type=str, default=None, help="YAML config path (optional if CLI args provided)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["base", "lora", "both"],
        default=None,
        help="Override inference mode (base/lora/both).",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="Override single input image path.",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=None,
        help="Override base output path.",
    )
    parser.add_argument(
        "--output-lora",
        type=str,
        default=None,
        help="Override LoRA output path.",
    )
    parser.add_argument(
        "--lora-ckpt",
        type=str,
        default=None,
        help="Override LoRA checkpoint path (supports .pt, .ckpt, or DeepSpeed mp_rank_00_model_states.pt).",
    )
    args = parser.parse_args()

    cfg = {}
    if args.config is not None:
        if not os.path.exists(args.config):
            print(f"Config not found: {args.config}")
            sys.exit(1)
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Paths
    BASE_MODEL = "stepfun-ai/Step1X-3D"
    MODE = str(args.mode or cfg.get("mode", "both")).lower()
    LORA_CKPT = args.lora_ckpt or cfg.get("lora_ckpt", None)

    # Input/Output (from config)
    INPUT_IMAGE = args.input_image or cfg.get("input_image", None)
    OUTPUT_BASE = args.output_base or cfg.get("output_base", None)
    OUTPUT_LORA = args.output_lora or cfg.get("output_lora", None)

    if MODE not in ["base", "lora", "both"]:
        print(f"Invalid mode: {MODE}. Must be one of: base, lora, both.")
        sys.exit(1)

    if INPUT_IMAGE is None:
        print("Missing required input_image (config field or --input-image).")
        sys.exit(1)

    if MODE in ["base", "both"] and OUTPUT_BASE is None:
        print("Missing required output_base (config field or --output-base) for mode=base/both.")
        sys.exit(1)

    if MODE in ["lora", "both"]:
        if OUTPUT_LORA is None:
            print("Missing required output_lora (config field or --output-lora) for mode=lora/both.")
            sys.exit(1)
        if LORA_CKPT is None:
            print("Missing required lora_ckpt (config field or --lora-ckpt) for mode=lora/both.")
            sys.exit(1)
        if not os.path.exists(LORA_CKPT):
            print(f"Checkpoint not found at {LORA_CKPT}")
            sys.exit(1)

    # 1. Run Base Model (No LoRA)
    if MODE in ["base", "both"]:
        print("\n=== Running Base Model Inference ===")
        pipeline_base = load_lora_pipeline(BASE_MODEL, lora_ckpt_path=None)
        run_inference(INPUT_IMAGE, OUTPUT_BASE, pipeline_base)

        # Clean up to save memory (optional, but good practice if memory is tight)
        del pipeline_base
        torch.cuda.empty_cache()

    # 2. Run LoRA Model
    if MODE in ["lora", "both"]:
        print("\n=== Running LoRA Model Inference ===")
        pipeline_lora = load_lora_pipeline(BASE_MODEL, LORA_CKPT)
        run_inference(INPUT_IMAGE, OUTPUT_LORA, pipeline_lora)

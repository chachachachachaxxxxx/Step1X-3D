import os
import torch
import warnings
from peft import LoraConfig
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline

# Suppress warnings
warnings.filterwarnings("ignore")

def load_lora_pipeline(base_model_path, lora_ckpt_path=None, device="cuda"):
    print(f"Loading base model from {base_model_path}...")
    pipeline = Step1X3DGeometryPipeline.from_pretrained(
        base_model_path, subfolder='Step1X-3D-Geometry-1300m',torch_dtype=torch.bfloat16
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
        ]
    )

    # Add LoRA adapter to the transformer (FluxDenoiser's dit_model)
    # The pipeline.transformer is the FluxDenoiser
    # In training (RectifiedFlowSystem), it is self.denoiser_model.dit_model.add_adapter(...)
    pipeline.transformer.dit_model.add_adapter(lora_config)

    # Load converted weights
    print(f"Loading LoRA weights from {lora_ckpt_path}...")
    state_dict = torch.load(lora_ckpt_path, map_location="cpu")

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
        generator=generator
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out.mesh[0].export(save_path)
    print(f"Saved result to {save_path}")

if __name__ == "__main__":
    # Paths
    BASE_MODEL = "stepfun-ai/Step1X-3D"
    # LORA_CKPT = "/home/wangxinxing/code/Step1X-3D/outputs/step1x-3d-geometry/jewelry-lora/michelangelo-autoencoder+n32768+AdamWlr0.0001@20251209-160940/converted_latest/pytorch_model.bin"
    # LORA_CKPT = "/home/wangxinxing/code/Step1X-3D/outputs/step1x-3d-geometry/jewelry-lora/michelangelo-autoencoder+n32768+AdamWlr0.0001@20251204-203327/converted_ckpt/pytorch_model.bin"

    LORA_CKPT = "/home/wangxinxing/code/Step1X-3D/outputs/step1x-3d-geometry/jewelry-lora-v1/michelangelo-autoencoder+n32768+AdamWlr0.0001@20251216-150521/converted/pytorch_model.bin"

    # Input/Output
    INPUT_IMAGE = "examples/jewelry/images/833-NJ010550-GD01_t.png"
    OUTPUT_BASE = "examples/jewelry/untexture_mesh/833-NJ010550-GD01_basev2.glb"
    OUTPUT_LORA = "examples/jewelry/untexture_mesh/833-NJ010550-GD01_lorav2100.glb"

    # Optional second test case (uncomment to use)
    # INPUT_IMAGE = "examples/jewelry/images/833-NJ010550-GD01.png"
    # OUTPUT_BASE = "examples/jewelry/untexture_mesh/833-NJ010550-GD01_base.glb"
    # OUTPUT_LORA = "examples/jewelry/untexture_mesh/833-NJ010550-GD01_lora2.glb"


    INPUT_IMAGE = "jewelry/remove_bg/images/BEN2 foreground.png"
    OUTPUT_BASE = "jewelry/remove_bg/images/833-NJ010550-GD01.glb"
    OUTPUT_LORA = "jewelry/remove_bg/images/833-NJ010550-GD01_lora.glb"
    
    # 1. Run Base Model (No LoRA)
    print("\n=== Running Base Model Inference ===")
    pipeline_base = load_lora_pipeline(BASE_MODEL, lora_ckpt_path=None)
    run_inference(INPUT_IMAGE, OUTPUT_BASE, pipeline_base)
    
    # Clean up to save memory (optional, but good practice if memory is tight)
    del pipeline_base
    torch.cuda.empty_cache()

    # 2. Run LoRA Model
    if not os.path.exists(LORA_CKPT):
        print(f"Checkpoint not found at {LORA_CKPT}")
        exit(1)
        
    print("\n=== Running LoRA Model Inference ===")
    pipeline_lora = load_lora_pipeline(BASE_MODEL, LORA_CKPT)
    run_inference(INPUT_IMAGE, OUTPUT_LORA, pipeline_lora)

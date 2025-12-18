import warnings

warnings.filterwarnings("ignore")
import os
import trimesh
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import (
    Step1X3DTexturePipeline,
)

from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
import torch

def geometry_pipeline(input_image_path, save_glb_path):
    """
    The base geometry model, input image generate glb
    """
    pipeline = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-1300m'
    ).to("cuda")

    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(2025)
    out = pipeline(input_image_path, guidance_scale=7.5, num_inference_steps=50, generator=generator)

    os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
    out.mesh[0].export(save_glb_path)


def geometry_label_pipeline(input_image_path, save_glb_path):
    """
    The label geometry model, support using label to control generation, input image generate glb
    """
    pipeline = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-Label-1300m'
    ).to("cuda")
    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(2025)

    out = pipeline(
        input_image_path,
        label={"symmetry": "x", "edge_type": "sharp"},
        guidance_scale=7.5,
        octree_resolution=384,
        max_facenum=400000,
        num_inference_steps=50,
        generator=generator
    )

    os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
    out.mesh[0].export(save_glb_path)


def texture_pipeline(input_image_path, input_glb_path, save_glb_path):
    """
    The texture model, input image and glb generate textured glb
    """
    mesh = trimesh.load(input_glb_path)
    pipeline = Step1X3DTexturePipeline.from_pretrained("stepfun-ai/Step1X-3D", subfolder="Step1X-3D-Texture")
    mesh = remove_degenerate_face(mesh)
    mesh = reduce_face(mesh)
    textured_mesh = pipeline(input_image_path, mesh, seed=2025)
    os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
    textured_mesh.export(save_glb_path)


if __name__ == "__main__":

    # example_index = "005"
    # geometry_pipeline(f"examples/images/{example_index}.png", f"examples/my_examples/untexture_mesh/u{example_index}.glb")
    # texture_pipeline(f"examples/images/{example_index}.png", f"examples/my_examples/untexture_mesh/u{example_index}.glb", f"examples/my_examples/textured_mesh/{example_index}.glb")

    # example_name="833-NJ010550-GD01"
    # # example_name="815-TE115315-SV02"
    # geometry_pipeline(f"examples/jewelry/images/{example_name}.png", f"examples/jewelry/untexture_mesh/u{example_name}.glb")
    # texture_pipeline(f"examples/jewelry/images/{example_name}.png", f"examples/jewelry/untexture_mesh/u{example_name}.glb", f"examples/jewelry/textured_mesh/{example_name}.glb")

    image = "/cache/wangxinxing/data/jewelry/clean_dataset_after_sampling/815-TE115315-SV02/512x512/1.jpg"
    untextured = "/cache/wangxinxing/data/jewelry/clean_dataset_after_sampling/815-TE115315-SV02/815-TE115315-SV02.glb"
    textured = "outputs/815-TE115315-SV02.glb"  
    texture_pipeline(image, untextured, textured)
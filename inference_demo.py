import torch
# Stage 1: 3D geometry generation
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline

def geometry_pipeline(input_image_path, untexture_mesh_path):
    # define the pipeline
    geometry_pipeline = Step1X3DGeometryPipeline.from_pretrained("stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-1300m'
    ).to("cuda")

    # run pipeline and obtain the untextured mesh 
    generator = torch.Generator(device=geometry_pipeline.device).manual_seed(2025)
    out = geometry_pipeline(input_image_path, guidance_scale=7.5, num_inference_steps=50)

    # export untextured mesh as .glb format
    out.mesh[0].export(untexture_mesh_path)


def texture_pipeline(input_image_path, untexture_mesh_path, textured_mesh_path):
    # Stage 2: 3D texure synthsis
    from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import (
        Step1X3DTexturePipeline,
    )
    from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face
    import trimesh

    # load untextured mesh
    untexture_mesh = trimesh.load(untexture_mesh_path)

    # define texture_pipeline
    texture_pipeline = Step1X3DTexturePipeline.from_pretrained("stepfun-ai/Step1X-3D", subfolder="Step1X-3D-Texture")

    # reduce face
    untexture_mesh = remove_degenerate_face(untexture_mesh)
    untexture_mesh = reduce_face(untexture_mesh)

    # texture mapping
    textured_mesh = texture_pipeline(input_image_path, untexture_mesh)

    # export textured mesh as .glb format
    textured_mesh.export(textured_mesh_path)


if __name__ == "__main__":

    # example_index = "005"
    # geometry_pipeline(f"examples/images/{example_index}.png", f"examples/my_examples/untexture_mesh/u{example_index}.glb")
    # texture_pipeline(f"examples/images/{example_index}.png", f"examples/my_examples/untexture_mesh/u{example_index}.glb", f"examples/my_examples/textured_mesh/{example_index}.glb")

    example_name="833-NJ010550-GD01"
    geometry_pipeline(f"examples/jewelry/images/{example_name}.png", f"examples/jewelry/untexture_mesh/u{example_name}.glb")
    texture_pipeline(f"examples/jewelry/images/{example_name}.png", f"examples/jewelry/untexture_mesh/u{example_name}.glb", f"examples/jewelry/textured_mesh/{example_name}.glb")
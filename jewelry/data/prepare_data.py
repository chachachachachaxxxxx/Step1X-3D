import os
import glob
import json
import shutil
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm
import subprocess
import argparse

def process_dataset(input_root, output_root, python_path="python"):
    """
    Convert the clean dataset into the training format expected by Step1X-3D.
    
    Args:
        input_root (str): Path to the cleaned jewelry dataset
        output_root (str): Path to save the processed training data
    """
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    # Create directory structure
    images_dir = output_path / "images"
    surfaces_dir = output_path / "surfaces"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    surfaces_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of project directories
    projects = [d for d in input_path.iterdir() if d.is_dir()]
    
    train_uids = []
    val_uids = []
    
    # Path to the watertight and sampling script
    watertight_script = Path("/home/wangxinxing/code/Step1X-3D/data/watertight_and_sampling.py").absolute()
    
    print(f"Found {len(projects)} projects to process")
    
    for idx, project_dir in enumerate(tqdm(projects)):
        uid = project_dir.name
        
        # 1. Handle Images
        # Source: project_dir/512x512/1.jpg and 2.jpg
        # Target: output_root/images/{uid}/0000_rgb.png and 0001_rgb.png
        
        src_img_dir = project_dir / "512x512"
        dst_img_dir = images_dir / uid
        dst_img_dir.mkdir(exist_ok=True)
        
        # We need to find 1.jpg/png and 2.jpg/png (case insensitive)
        img_files = sorted(list(src_img_dir.glob("*")))
        
        # Map source images to target indices
        # Assuming the clean dataset guarantees 1.jpg and 2.jpg exist
        processed_images = False
        for img_file in img_files:
            if img_file.stem == "1":
                # Convert to png if needed and save as 0000_rgb.png
                try:
                    import PIL.Image
                    img = PIL.Image.open(img_file)
                    # Ensure RGBA for consistency with transparency if needed, 
                    # though original might be jpg. The training pipeline expects png.
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    img.save(dst_img_dir / "0000_rgb.png")
                except Exception as e:
                    print(f"Error processing image 1 for {uid}: {e}")
            elif img_file.stem == "2":
                try:
                    import PIL.Image
                    img = PIL.Image.open(img_file)
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    img.save(dst_img_dir / "0001_rgb.png")
                except Exception as e:
                    print(f"Error processing image 2 for {uid}: {e}")
        
        # 2. Handle Geometry (Watertight & Sampling)
        # Source: project_dir/{uid}.glb
        # Target: output_root/surfaces/{uid}.npz
        
        glb_file = project_dir / f"{uid}.glb"
        if not glb_file.exists():
            print(f"Warning: GLB file not found for {uid}, skipping geometry processing")
            continue
            
        # We need to run the watertight script
        # The script outputs to a specific directory structure, we might need to move files after
        
        # Create a temporary output dir for this mesh
        temp_output = surfaces_dir / "temp" / uid
        temp_output.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            python_path, str(watertight_script),
            "--input_mesh", str(glb_file),
            "--output_path", str(temp_output),
            "--grid_resolution", "256",
            # "--sample_sharp_edge", "True" # Using string "True" because argparse logic in script might be tricky with bools
        ]
        
        # Note: The provided script uses 'type=bool' which behaves unexpectedly in argparse (any non-empty string is True).
        # But 'default=True' is set. If we want to be safe we can just omit it to use default True, 
        # or pass a non-empty string.
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # The script saves samples.npz in {output_path}/{input_filename_stem}/samples.npz
            # Our temp_output is .../temp/{uid}
            # The script logic: output_path = f"{args.output_path}/{args.input_mesh.split('/')[-1].split('.')[0]}"
            # So actual path is temp_output / uid / samples.npz
            
            generated_npz = temp_output / uid / "samples.npz"
            
            if generated_npz.exists():
                shutil.move(str(generated_npz), str(surfaces_dir / f"{uid}.npz"))
                
                # Add to dataset lists
                # Simple split: last 2 for validation, rest for training
                if idx >= len(projects) - 2:
                    val_uids.append(uid)
                else:
                    train_uids.append(uid)
            else:
                print(f"Error: .npz file not generated for {uid}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running watertight script for {uid}: {e}")
        except Exception as e:
            print(f"Unexpected error for {uid}: {e}")
        finally:
            # Clean up temp
            if (surfaces_dir / "temp").exists():
                shutil.rmtree(surfaces_dir / "temp")

    # 3. Create Split JSONs
    # The dataloader expects train.json/val.json to be lists of UIDs relative to root or just UIDs?
    # Checking ObjaverseDataset:
    # self.uids = json.load(open(f"{cfg.root_dir}/{split}.json"))
    # _load_shape...: np.load(f"{self.cfg.root_dir}/surfaces/{self.uids[index]}.npz")
    # _load_image...: f"{self.cfg.root_dir}/images/" + "/".join(self.uids[index].split("/")[-2:]) ...
    
    # Wait, the Objaverse dataloader seems to expect UIDs to have some directory structure like "000-000/uid" 
    # based on: "/".join(self.uids[index].split("/")[-2:])
    # But our structure is flat: images/{uid} and surfaces/{uid}.npz
    
    # Let's verify ObjaverseDataset._load_image logic in Step1X-3D/step1x3d_geometry/data/Objaverse.py or base.py
    # From read_file of base.py:
    # img_path = f"{self.cfg.root_dir}/images/" + "/".join(self.uids[index].split("/")[-2:]) + ...
    
    # If our UID is just "815-TE...", split("/")[-2:] will just be ["815-TE..."] (if no slashes)? 
    # Python split on string without delimiter returns list with string.
    # Actually if uids[index] is "abc", split is ["abc"]. slice [-2:] is ["abc"]. join is "abc".
    # So f"{root}/images/abc/..." works perfectly for flat structure too.
    
    with open(output_path / "train.json", "w") as f:
        json.dump(train_uids, f, indent=4)
        
    with open(output_path / "val.json", "w") as f:
        json.dump(val_uids, f, indent=4)
        
    print(f"Processing complete.")
    print(f"Training samples: {len(train_uids)}")
    print(f"Validation samples: {len(val_uids)}")
    print(f"Output directory: {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/cache/wangxinxing/data/jewelry/clean_dataset", help="Path to clean dataset")
    parser.add_argument("--output", default="data/shape_diffusion/jewelry", help="Output path for training data")
    args = parser.parse_args()
    
    process_dataset(args.input, args.output)


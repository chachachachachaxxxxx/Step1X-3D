#!/usr/bin/env python3

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
RENDER_TOOL = Path("/home/wangxinxing/code/bpy-renderer/scripts/render_6ortho")


def run(cmd):
    cmd = [str(x) for x in cmd]
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def resolve_lora(source, project_root):
    if not source:
        return None
    src = Path(source).expanduser().resolve()
    ckpt_dir = src / "ckpts" / "last.ckpt"
    if ckpt_dir.is_dir():
        out_dir = src / "converted"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not (out_dir / "pytorch_model.bin").exists():    # 如果转换后的ckpt不存在，则进行转换
            run([sys.executable, project_root / "zero_to_fp32.py", ckpt_dir, out_dir])
        else:
            print(f"转换后的ckpt已存在：{out_dir / 'pytorch_model.bin'}")
        ckpt = out_dir / "pytorch_model.bin"
        if not ckpt.exists():
            raise SystemExit(f"转换失败，未找到 {ckpt}")
        return ckpt
    if src.is_file():
        return src
    raise SystemExit(f"无法识别的 LoRA 来源：{source}")


def iter_images(folder):
    files = sorted(
        p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES
    )
    if not files:
        raise SystemExit(f"目录下没有 png/jpg/jpeg/webp：{folder}")
    return files


def main():
    

    parser = argparse.ArgumentParser(description="Step1X-3D 推理流水线 (convert+infer+render)")
    parser.add_argument("images_dir")
    parser.add_argument("mode", nargs="?", default="both")
    parser.add_argument("lora_source", nargs="?", default="")
    parser.add_argument("render_flag", nargs="?", default="render")
    parser.add_argument("result_root", nargs="?", default="jewelry/inference/results")
    args = parser.parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"找不到图片目录：{images_dir}")

    mode = args.mode.lower()
    if mode not in {"base", "lora", "both"}:
        raise SystemExit("mode 只能是 base/lora/both")

    lora_source = args.lora_source or ""
    render_flag = (args.render_flag or "render").lower()
    if render_flag == "render" and lora_source.lower() in {"render", "skip"}:
        render_flag = lora_source.lower()
        lora_source = ""
    if render_flag not in {"render", "skip"}:
        raise SystemExit("render_flag 只能是 render/skip")

    project_root = Path(__file__).resolve().parent
    result_root = Path(args.result_root).expanduser().resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    result_dir = result_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir.mkdir(parents=True, exist_ok=True)

    # 记录 LoRA 来源
    if lora_source:
        (result_dir / "lora_source.txt").write_text(str(lora_source))

    t_start = perf_counter()
    convert_time = 0.0
    lora_ckpt = None
    if lora_source:
        t_convert = perf_counter()
        lora_ckpt = resolve_lora(lora_source, project_root)
        convert_time = perf_counter() - t_convert
    else:
        lora_ckpt = None

    if mode != "base" and not (lora_ckpt and lora_ckpt.is_file()):
        raise SystemExit("mode=lora/both 需要有效 LoRA 权重")

    inference_py = project_root / "inference_lora.py"
    generated = []
    infer_start = perf_counter()
    for img in iter_images(images_dir):
        base_out = result_dir / f"{img.stem}_base.glb" if mode != "lora" else None
        lora_out = result_dir / f"{img.stem}_lora.glb" if mode != "base" else None

        cmd = [
            sys.executable,
            inference_py,
            "--mode",
            mode,
            "--input-image",
            img,
        ]
        if base_out:
            cmd += ["--output-base", base_out]
        if lora_out:
            cmd += ["--output-lora", lora_out, "--lora-ckpt", lora_ckpt]

        print(f"[inference] {img}")
        run(cmd)
        if base_out and base_out.exists():
            generated.append(base_out)
        if lora_out and lora_out.exists():
            generated.append(lora_out)

    infer_time = perf_counter() - infer_start

    render_time = 0.0
    if render_flag == "render":
        render_start = perf_counter()
        if not RENDER_TOOL.exists():
            raise SystemExit(f"render_6ortho 不存在：{RENDER_TOOL}")
        for glb in generated:
            render_dir = glb.parent / f"{glb.stem}_renders"
            render_dir.mkdir(parents=True, exist_ok=True)
            print(f"[render] {glb}")
            run([RENDER_TOOL, "--input_path", glb, "--output_dir", render_dir])
        render_time = perf_counter() - render_start
    else:
        print("[render] skip")

    total_time = perf_counter() - t_start
    print("完成，输出目录：", result_dir)
    print(
        f"耗时统计 -> 总计 {total_time:.1f}s | 转换 {convert_time:.1f}s | 推理 {infer_time:.1f}s | 渲染 {render_time:.1f}s"
    )


if __name__ == "__main__":
    main()

"""
带显存监控的推理脚本

基于 inference_demo.py，添加了显存占用监控和可视化功能。
会记录各个推理阶段的显存使用情况，并生成显存占用图表。
"""

import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# 显存监控数据
memory_stats = defaultdict(list)
timestamps = []
start_time = time.time()

def record_memory(stage_name):
    """记录当前显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        memory_stats['allocated'].append(allocated)
        memory_stats['reserved'].append(reserved)
        memory_stats['max_allocated'].append(max_allocated)
        timestamps.append(stage_name)
        
        elapsed_time = time.time() - start_time
        print(f"[{stage_name}] 显存占用: {allocated:.2f} GB (已分配), "
              f"{reserved:.2f} GB (已保留), {max_allocated:.2f} GB (峰值) "
              f"[耗时: {elapsed_time:.2f}s]")
    else:
        print(f"[{stage_name}] CPU 模式，无法监控显存")
        timestamps.append(stage_name)
        memory_stats['allocated'].append(0)
        memory_stats['reserved'].append(0)
        memory_stats['max_allocated'].append(0)

def plot_memory_usage(save_path='memory_usage.png', show_plot=True):
    """绘制显存占用图表"""
    if not timestamps:
        print("警告: 没有记录到任何显存数据")
        return
    
    print("\n绘制显存占用图表...")
    plt.figure(figsize=(12, 6))
    x_positions = range(len(timestamps))
    
    # 绘制三条曲线
    plt.plot(x_positions, memory_stats['allocated'], 'o-', 
            label='已分配显存 (GB)', linewidth=2, markersize=8, color='#2E86AB')
    plt.plot(x_positions, memory_stats['reserved'], 's-', 
            label='已保留显存 (GB)', linewidth=2, markersize=8, color='#A23B72')
    plt.plot(x_positions, memory_stats['max_allocated'], '^-', 
            label='峰值显存 (GB)', linewidth=2, markersize=8, alpha=0.7, color='#F18F01')
    
    # 设置标签和标题
    plt.xlabel('推理阶段', fontsize=12, fontweight='bold')
    plt.ylabel('显存占用 (GB)', fontsize=12, fontweight='bold')
    plt.title('推理过程中的显存占用变化', fontsize=14, fontweight='bold', pad=20)
    
    # 设置 x 轴标签
    plt.xticks(x_positions, timestamps, rotation=45, ha='right')
    
    # 添加图例和网格
    plt.legend(fontsize=10, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加数值标注（如果点不太多的话）
    if len(timestamps) <= 10:
        for i, alloc in enumerate(memory_stats['allocated']):
            plt.annotate(f'{alloc:.1f}GB', 
                       (i, alloc), 
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center', 
                       fontsize=8,
                       alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"显存占用图表已保存为 {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def print_summary():
    """打印显存使用摘要"""
    if not memory_stats['allocated']:
        return
    
    print("\n" + "="*60)
    print("显存使用摘要")
    print("="*60)
    print(f"峰值已分配显存: {max(memory_stats['allocated']):.2f} GB")
    print(f"峰值已保留显存: {max(memory_stats['reserved']):.2f} GB")
    print(f"峰值显存: {max(memory_stats['max_allocated']):.2f} GB")
    print(f"最终已分配显存: {memory_stats['allocated'][-1]:.2f} GB")
    print(f"最终已保留显存: {memory_stats['reserved'][-1]:.2f} GB")
    print(f"监控阶段数: {len(timestamps)}")
    print("="*60)

# Stage 1: 3D geometry generation
record_memory("初始状态")
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline

# define the pipeline
print("加载几何生成模型...")
geometry_pipeline = Step1X3DGeometryPipeline.from_pretrained("stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-1300m'
).to("cuda")
record_memory("几何模型加载完成")

# input image
input_image_path = "examples/images/020.png"

# run pipeline and obtain the untextured mesh 
print("开始几何生成...")
generator = torch.Generator(device=geometry_pipeline.device).manual_seed(2025)
out = geometry_pipeline(input_image_path, guidance_scale=7.5, num_inference_steps=50)
record_memory("几何生成完成")

# export untextured mesh as .glb format
out.mesh[0].export("untexture_mesh.glb")
record_memory("几何网格导出完成")


# Stage 2: 3D texure synthsis
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import (
    Step1X3DTexturePipeline,
)
from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face
import trimesh

# load untextured mesh
print("加载未纹理网格...")
untexture_mesh = trimesh.load("untexture_mesh.glb")
record_memory("未纹理网格加载完成")

# define texture_pipeline
print("加载纹理合成模型...")
texture_pipeline = Step1X3DTexturePipeline.from_pretrained("stepfun-ai/Step1X-3D", subfolder="Step1X-3D-Texture")
record_memory("纹理模型加载完成")

# reduce face
print("处理网格面...")
untexture_mesh = remove_degenerate_face(untexture_mesh)
untexture_mesh = reduce_face(untexture_mesh)
record_memory("网格面处理完成")

# texture mapping
print("开始纹理合成...")
textured_mesh = texture_pipeline(input_image_path, untexture_mesh)
record_memory("纹理合成完成")

# export textured mesh as .glb format
textured_mesh.export("textured_mesh.glb")
record_memory("纹理网格导出完成")

# 绘制显存占用图表并打印摘要
plot_memory_usage()
print_summary()


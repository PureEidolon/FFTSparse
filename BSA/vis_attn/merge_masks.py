import os
import re
from pathlib import Path
from PIL import Image
import math

output_dir = Path("output")
datasets = [d for d in output_dir.iterdir() if d.is_dir()]

for dataset_dir in datasets:
    for img_type in ["stripe_variance", "block_level_attn"]:
        # 找到所有该类型的图，按层号排序
        imgs = sorted(
            [f for f in dataset_dir.glob(f"*_{img_type}.png")],
            key=lambda f: int(re.search(r"layer(\d+)", f.name).group(1))
        )
        if not imgs:
            continue

        print(f"合并 {dataset_dir.name}/{img_type}: {len(imgs)} 张")

        # 读取所有图
        pil_imgs = [Image.open(f) for f in imgs]

        # 每行放几张（按层数开平方取整）
        n = len(pil_imgs)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

        # 统一缩放到第一张图的尺寸
        w, h = pil_imgs[0].size

        canvas = Image.new("RGB", (ncols * w, nrows * h), color=(255, 255, 255))
        for idx, img in enumerate(pil_imgs):
            img_resized = img.resize((w, h))
            row = idx // ncols
            col = idx % ncols
            canvas.paste(img_resized, (col * w, row * h))

        save_path = dataset_dir / f"0_merged_{img_type}.png"
        canvas.save(save_path)
        print(f"  => 保存到 {save_path}")

print("全部完成")
import argparse
import os
import torch
from PIL import Image
from pathlib import Path
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from PySide6.QtCore import QCoreApplication


def test_transform(img, size):
    h, w, _ = np.shape(img)
    if h < w:
        newh = size
        neww = w / h * size
    else:
        neww = size
        newh = h / w * size
    neww = int(neww // 4 * 4)
    newh = int(newh // 4 * 4)
    transform_list = [
        transforms.Resize((newh, neww)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def ArtFlow_batch(mode, content_dir, style_dir, output_dir, update_callback, size: int = 512):

    ckpt_map = {
        'adain':    'lib/function/Image_Style_Transfer/ArtFlow/checkpoint/ArtFlow-AdaIN/glow.pth',
        'wct':      'lib/function/Image_Style_Transfer/ArtFlow/checkpoint/ArtFlow-WCT/glow.pth',
        'portrait': 'lib/function/Image_Style_Transfer/ArtFlow/checkpoint/ArtFlow-AdaIN-Portrait/glow.pth'
    }
    if mode not in ckpt_map:
        raise ValueError(f"Unknown mode '{mode}', choose from 'adain', 'wct', 'portrait'.")
    checkpoint_path = ckpt_map[mode]

    # 导入对应 Glow 实现
    if mode == 'wct':
        from .glow_wct import Glow
    else:
        from .glow_adain import Glow

    # 模型配置（根据 ArtFlow 默认设置）
    n_flow = 8
    n_block = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建并加载模型
    glow = Glow(3, n_flow, n_block, affine=False, conv_lu=True).to(device)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    glow.load_state_dict(ckpt['state_dict'])
    glow.eval()

    # 准备文件列表
    content_paths = list(Path(content_dir).glob('*.*'))
    style_paths   = list(Path(style_dir).glob('*.*'))

    # 输出目录
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for c_path in content_paths:
            content_img = Image.open(str(c_path)).convert('RGB')
            content_tensor = test_transform(content_img, size)(content_img).unsqueeze(0).to(device)
            z_c = glow(content_tensor, forward=True)

            for s_path in style_paths:
                style_img = Image.open(str(s_path)).convert('RGB')
                style_tensor = test_transform(style_img, size)(style_img).unsqueeze(0).to(device)
                z_s = glow(style_tensor, forward=True)

                output = glow(z_c, forward=False, style=z_s).cpu()
                save_name = f"{c_path.stem}_{s_path.stem}.jpg"
                save_image(output, str(out_dir / save_name))
                message1 = f"[{os.path.basename(c_path)} ⟷ {os.path.basename(s_path)}] → Saved: {save_name}\n"
                update_callback(message1)
                QCoreApplication.processEvents()
                # print(message1)

    message2 = f"Style Transfer Successful!"
    update_callback(message2)
    QCoreApplication.processEvents()
    # print(message2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--operator', type=str, default='adain', choices=['adain', 'wct', 'portrait'], help='Select network configuration')
    parser.add_argument('--content',  type=str, default='data/content', help='Directory of content images')
    parser.add_argument('--style',    type=str, default='data/style',   help='Directory of style images')
    parser.add_argument('--out_dir',  type=str, default='output',      help='Directory to save output')
    args = parser.parse_args()

    def update_callback(msg):
        return

    ArtFlow_batch(mode=args.operator,
                  content_dir=args.content,
                  style_dir=args.style,
                  output_dir=args.out_dir,
                  update_callback=update_callback)

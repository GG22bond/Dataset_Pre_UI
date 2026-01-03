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


def ArtFlow_single(mode, content_path, style_path, output_path, update_callback, size: int = 512):

    ckpt_map = {
        'adain':    'lib/function/Image_Style_Transfer/ArtFlow/checkpoint/ArtFlow-AdaIN/glow.pth',
        'wct':      'lib/function/Image_Style_Transfer/ArtFlow/checkpoint/ArtFlow-WCT/glow.pth',
        'portrait': 'lib/function/Image_Style_Transfer/ArtFlow/checkpoint/ArtFlow-AdaIN-Portrait/glow.pth'
    }
    if mode not in ckpt_map:
        raise ValueError(f"Unknown mode '{mode}', choose from {list(ckpt_map.keys())}.")
    checkpoint_path = ckpt_map[mode]

    # 动态导入 Glow
    if mode == 'wct':
        from .glow_wct import Glow
    else:
        from .glow_adain import Glow

    # 模型参数
    n_flow, n_block = 8, 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    glow = Glow(3, n_flow, n_block, affine=False, conv_lu=True).to(device)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    glow.load_state_dict(ckpt['state_dict'])
    glow.eval()

    # 读取并预处理图像
    content_img = Image.open(content_path).convert('RGB')
    style_img   = Image.open(style_path).convert('RGB')
    content_t   = test_transform(content_img, size)(content_img).unsqueeze(0).to(device)
    style_t     = test_transform(style_img,   size)(style_img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        z_c = glow(content_t, forward=True)
        z_s = glow(style_t,   forward=True)
        output = glow(z_c, forward=False, style=z_s).cpu()

    # 保存
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(output, output_path)
    message1 = f"[{os.path.basename(content_path)} ⟷ {os.path.basename(style_path)}] successfully\n"
    update_callback(message1)
    QCoreApplication.processEvents()

    message2 = f"Saved stylized image to: {output_path}"
    update_callback(message2)
    QCoreApplication.processEvents()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--operator', type=str, default='adain', choices=['adain', 'wct', 'portrait'],
                        help='Select network configuration')
    parser.add_argument('--content',  type=str, default='data/content/6snyyart2t84dhcp.jpg',
                        help='Path to the content image')
    parser.add_argument('--style',    type=str, default='data/style/55f9ce419aab4488b0336099a840e1a7.jpg',
                        help='Path to the style image')
    parser.add_argument('--out_dir',  type=str, default='output/3.jpg',
                        help='Path to save the stylized image')

    args = parser.parse_args()

    def update_callback(msg):
        return

    ArtFlow_single(mode=args.operator,
                    content_path=args.content,
                    style_path=args.style,
                    output_path=args.out_dir,
                    update_callback=update_callback)



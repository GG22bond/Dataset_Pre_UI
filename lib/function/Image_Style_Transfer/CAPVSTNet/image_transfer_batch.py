import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as utils
from PySide6.QtCore import QCoreApplication

from .utils.utils import img_resize
from .models.cWCT import cWCT
from .models.RevResNet import RevResNet


def cap_vstnet_images_transfer(mode, content_dir, style_dir, out_dir, update_callback, max_size: int = 1280, alpha_c: float = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(f'→ Generating images from "{content_dir}" × "{style_dir}" ')
    message1 = f"→ Generating images from {content_dir} × {style_dir}\n"
    update_callback(message1)
    QCoreApplication.processEvents()

    if mode.lower() == "photorealistic":
        ckpoint = 'lib/function/Image_Style_Transfer/CAPVSTNet/checkpoints/photo_image.pt'
        rev_net = RevResNet(
            nBlocks=[10, 10, 10], nStrides=[1, 2, 2],
            nChannels=[16, 64, 256], in_channel=3,
            mult=4, hidden_dim=16, sp_steps=2)
    elif mode.lower() == "artistic":
        ckpoint = 'lib/function/Image_Style_Transfer/CAPVSTNet/checkpoints/art_image.pt'
        rev_net = RevResNet(
            nBlocks=[10, 10, 10], nStrides=[1, 2, 2],
            nChannels=[16, 64, 256], in_channel=3,
            mult=4, hidden_dim=64, sp_steps=1)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # 加载预训练权重
    state = torch.load(ckpoint)
    rev_net.load_state_dict(state['state_dict'])
    rev_net = rev_net.to(device).eval()

    cwct = cWCT()

    def list_images(dir_path):
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        files = [f for f in os.listdir(dir_path)
                 if os.path.splitext(f.lower())[1] in exts]
        files.sort()
        return [os.path.join(dir_path, f) for f in files]

    content_files = list_images(content_dir)
    style_files   = list_images(style_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 全组合：每张内容图 × 每张风格图
    for content_path in content_files:
        for style_path in style_files:

            content = Image.open(content_path).convert('RGB')
            style   = Image.open(style_path).convert('RGB')

            content = img_resize(content, max_size, down_scale=rev_net.down_scale)
            style   = img_resize(style, max_size, down_scale=rev_net.down_scale)

            content_t = transforms.ToTensor()(content).unsqueeze(0).to(device)
            style_t   = transforms.ToTensor()(style).unsqueeze(0).to(device)

            with torch.no_grad():
                z_c = rev_net(content_t, forward=True)
                z_s = rev_net(style_t,   forward=True)

                if alpha_c is not None:
                    z_cs = cwct.interpolation(
                        z_c,
                        styl_feat_list=[z_s],
                        alpha_s_list=[1.0],
                        alpha_c=alpha_c)
                else:
                    z_cs = cwct.transfer(z_c, z_s)

                stylized = rev_net(z_cs, forward=False)

            cn = os.path.splitext(os.path.basename(content_path))[0]
            sn = os.path.splitext(os.path.basename(style_path))[0]

            cn_ = os.path.basename(content_path)
            sn_ = os.path.basename(style_path)

            out_name = f"{cn}_{sn}.png"
            out_path = os.path.join(out_dir, out_name)

            grid  = utils.make_grid(stylized.data, nrow=1, padding=0)
            ndarr = (grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
            out_img = Image.fromarray(ndarr)
            out_img.save(out_path, quality=100)

            # print(f"[{cn} ⟷ {sn}] → Saved: {out_path}")
            message2 = f"[{cn_} ⟷ {sn_}] → Saved: {out_name}\n"
            update_callback(message2)
            QCoreApplication.processEvents()

    # print("Style Transfer Successful!")
    message3 = f"Style Transfer Successful!"
    update_callback(message3)
    QCoreApplication.processEvents()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='photorealistic', choices=['photorealistic', 'artistic'], help='Select network configuration')
    parser.add_argument('--content', type=str, default='data/content', help='Directory of content images')
    parser.add_argument('--style', type=str, default='data/style', help='Directory of style images')
    parser.add_argument('--out_dir', type=str, default='output', help='Directory to save output')

    args = parser.parse_args()

    def update_callback(msg):
        return

    cap_vstnet_images_transfer(mode=args.mode,
                               content_dir=args.content,
                               style_dir=args.style,
                               out_dir=args.out_dir,
                               update_callback=update_callback)



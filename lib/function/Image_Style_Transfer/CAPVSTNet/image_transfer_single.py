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

def cap_vstnet_image_transfer(mode, content_path, style_path, out_path, update_callback, max_size: int = 1280, alpha_c: float = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine checkpoint based on mode
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

    state = torch.load(ckpoint)
    rev_net.load_state_dict(state['state_dict'])
    rev_net = rev_net.to(device)
    rev_net.eval()

    cwct = cWCT()

    content = Image.open(content_path).convert('RGB')
    style = Image.open(style_path).convert('RGB')

    content = img_resize(content, max_size, down_scale=rev_net.down_scale)
    style = img_resize(style, max_size, down_scale=rev_net.down_scale)

    content_tensor = transforms.ToTensor()(content).unsqueeze(0).to(device)
    style_tensor = transforms.ToTensor()(style).unsqueeze(0).to(device)

    with torch.no_grad():
        z_c = rev_net(content_tensor, forward=True)
        z_s = rev_net(style_tensor,   forward=True)

        if alpha_c is not None:
            assert 0.0 <= alpha_c <= 1.0, "alpha_c must be in [0,1]"
            z_cs = cwct.interpolation(
                z_c,
                styl_feat_list=[z_s],
                alpha_s_list=[1.0],
                alpha_c=alpha_c)
        else:
            z_cs = cwct.transfer(z_c, z_s)

        stylized = rev_net(z_cs, forward=False)

    save_dir = os.path.dirname(out_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    grid = utils.make_grid(stylized.data, nrow=1, padding=0)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    out_img = Image.fromarray(ndarr)

    ext = os.path.splitext(out_path)[1].lower().strip('.')
    kwargs = {}
    if ext in ['jpg', 'jpeg']:
        kwargs['quality'] = 100
    out_img.save(out_path, **kwargs)
    # print(f"Saved stylized image at: {out_path}")
    message1 = f"[{os.path.basename(content_path)} ⟷ {os.path.basename(style_path)}] successfully\n"
    # print(message1)
    update_callback(message1)
    QCoreApplication.processEvents()

    return out_path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='photorealistic', choices=['photorealistic', 'artistic'], help='Select network configuration')
    parser.add_argument('--content', type=str, default='data/content/01.jpg', help='Path to content image')
    parser.add_argument('--style', type=str, default='data/style/01.jpg', help='Path to style image')
    parser.add_argument('--out_path', type=str, default='output/1.jpg', help='File path (with extension) to save output')

    args = parser.parse_args()

    def update_callback(msg):
        return

    cap_vstnet_image_transfer(mode=args.mode,
                              content_path=args.content,
                              style_path=args.style,
                              out_path=args.out_path,
                              update_callback=update_callback)



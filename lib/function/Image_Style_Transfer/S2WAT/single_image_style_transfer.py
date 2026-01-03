import argparse
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from torchvision.utils import save_image
from PySide6.QtCore import QCoreApplication

from .model.configuration import TransModule_Config
from .model.s2wat import S2WAT
from .net import TransModule, Decoder_MVGG
from .tools import Sample_Test_Net


def content_style_transTo_pt(i_c_path, i_s_path, i_c_size=None):
    """Resize the pics of arbitrary size to the shape of content image
    """
    i_c_pil = Image.open(i_c_path)
    i_s_pil = Image.open(i_s_path)

    if not i_c_size is None:
        i_c_tf = transforms.Compose([
            transforms.Resize(i_c_size),
            transforms.ToTensor()
        ])
    else:
        i_c_tf = transforms.Compose([
            transforms.ToTensor()
        ])

    i_s_size = min(i_c_pil.size[1], i_c_pil.size[0])
    i_s_tf = transforms.Compose([
        transforms.Resize(i_s_size),
        transforms.ToTensor()
    ])

    i_c_pt = i_c_tf(i_c_pil).unsqueeze(dim=0)
    i_s_pt = i_s_tf(i_s_pil).unsqueeze(dim=0)

    return i_c_pt, i_s_pt


def build_network(device):

    # Models Config
    transModule_config = TransModule_Config(
        nlayer=3,
        d_model=768,
        nhead=8,
        mlp_ratio=4,
        qkv_bias=False,
        attn_drop=0.,
        drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_first=True
    )

    # Instantiate models
    encoder = S2WAT(
        img_size=224,
        patch_size=2,
        in_chans=3,
        embed_dim=192,
        depths=[2, 2, 2],
        nhead=[3, 6, 12],
        strip_width=[2, 4, 7],
        drop_path_rate=0.,
        patch_norm=True
    )
    decoder = Decoder_MVGG(d_model=768, seq_input=True)
    transModule = TransModule(transModule_config)

    # Wrap into test network
    network = Sample_Test_Net(encoder, decoder, transModule).to(device)

    return network


def load_checkpoint(network, checkpoint_path, device):

    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.encoder.load_state_dict(checkpoint['encoder'])
    network.decoder.load_state_dict(checkpoint['decoder'])
    network.transModule.load_state_dict(checkpoint['transModule'])

# @torch.no_grad()
# def s2wat_transfer_img(network, content_path, style_path, output_dir, device):
#     """Perform style transfer for a single content-style pair and save the result."""
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"→ Generating stylized image from '{content_path}' with style '{style_path}'...")
#
#     # Prepare tensors
#     c_tensor, s_tensor = content_style_transTo_pt(content_path, style_path)
#     c_tensor = c_tensor.to(device)
#     s_tensor = s_tensor.to(device)
#
#     # Forward pass
#     out = network(c_tensor, s_tensor, arbitrary_input=True)
#
#     # Construct output filename
#     c_name = Path(content_path).stem
#     s_name = Path(style_path).stem
#     ext = Path(content_path).suffix
#     out_name = f"{c_name}_{s_name}{ext}"
#     out_path = os.path.join(output_dir, out_name)
#
#     # Save image
#     save_image(out, out_path)
#     print(f"[{os.path.basename(content_path)} ⟷ {os.path.basename(style_path)}] → Saved: {out_name}")



# Global setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network_ = build_network(DEVICE)
load_checkpoint(network_, 'lib/function/Image_Style_Transfer/S2WAT/checkpoint/checkpoint_40000_epoch.pkl', DEVICE)


@torch.no_grad()
def s2wat_transfer_img(content_path, style_path, output_path, update_callback, network=network_, device=DEVICE):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # print(f"→ Generating stylized image from '{content_path}' with style '{style_path}'...")
    # message1 = f"→ Generating stylized image from {content_path} with style {style_path}\n"
    # update_callback(message1)
    # QCoreApplication.processEvents()


    c_tensor, s_tensor = content_style_transTo_pt(content_path, style_path)
    c_tensor, s_tensor = c_tensor.to(device), s_tensor.to(device)

    out = network(c_tensor, s_tensor, arbitrary_input=True)

    save_image(out, output_path)
    # print(f"[{os.path.basename(content_path)} ⟷ {os.path.basename(style_path)}] → Saved: {os.path.basename(output_path)}")
    message2 = f"[{os.path.basename(content_path)} ⟷ {os.path.basename(style_path)}] successfully\n"
    update_callback(message2)
    QCoreApplication.processEvents()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type=str, default='input/Test/Content/image.jpg', help='Content images directory')
    parser.add_argument('--style_dir', type=str, default='input/Test/Style/blue_swirls.jpg', help='Style images directory')
    parser.add_argument('--output_dir', type=str, default='./output/image_blue_swirls.jpg', help='Output directory')

    args = parser.parse_args()

    def update_callback(msg):
        return

    s2wat_transfer_img(content_path=args.content_dir,
                       style_path=args.style_dir,
                       output_path=args.output_dir,
                       update_callback=update_callback)



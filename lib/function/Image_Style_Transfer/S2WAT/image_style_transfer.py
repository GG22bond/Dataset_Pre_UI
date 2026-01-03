import argparse
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
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

# Global setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network_ = build_network(DEVICE)
load_checkpoint(network_, 'lib/function/Image_Style_Transfer/S2WAT/checkpoint/checkpoint_40000_epoch.pkl', DEVICE)

@torch.no_grad()
def s2wat_transfer_imgs(content_dir, style_dir, output_dir, update_callback, network=network_, device= DEVICE):

    os.makedirs(output_dir, exist_ok=True)
    # print(f'→ Generating images from "{content_dir}" × "{style_dir}"\n ')
    message1 = f"→ Generating images from {content_dir} × {style_dir}\n"
    update_callback(message1)
    QCoreApplication.processEvents()


    c_names = sorted(os.listdir(content_dir))
    s_names = sorted(os.listdir(style_dir))

    for c_name in c_names:
        c_path = os.path.join(content_dir, c_name)

        if not os.path.isfile(c_path):
            continue

        c_tensor, _ = content_style_transTo_pt(c_path, c_path)
        for s_name in s_names:
            s_path = os.path.join(style_dir, s_name)
            if not os.path.isfile(s_path):
                continue

            _, s_tensor = content_style_transTo_pt(c_path, s_path)

            out = network(c_tensor.to(device), s_tensor.to(device), arbitrary_input=True)

            stem_c, ext_c = os.path.splitext(c_name)
            stem_s, _ = os.path.splitext(s_name)
            out_name = f'{stem_c}_{stem_s}{ext_c}'
            out_path = os.path.join(output_dir, out_name)

            save_image(out, out_path)

            # print(f"[{os.path.basename(c_name)} ⟷ {os.path.basename(s_path)}] → Saved: {out_name}\n")
            message2 = f"[{os.path.basename(c_name)} ⟷ {os.path.basename(s_path)}] → Saved: {out_name}\n"
            update_callback(message2)
            QCoreApplication.processEvents()

    message3 = f"Style Transfer Successful!"
    update_callback(message3)
    QCoreApplication.processEvents()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type=str, default='input/Test/Content', help='Content images directory')
    parser.add_argument('--style_dir', type=str, default='input/Test/Style', help='Style images directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')

    args = parser.parse_args()

    def update_callback(msg):
        return

    s2wat_transfer_imgs(content_dir=args.content_dir,
                        style_dir=args.style_dir,
                        output_dir=args.output_dir,
                        update_callback=update_callback)



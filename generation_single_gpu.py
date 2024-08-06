# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import argparse
import torch
from torchvision.utils import save_image
from masked_diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from masked_diffusion.models import MDTv2_XL_2
from tqdm import tqdm   


def main(args):
    if args.tf32: # True: fast but may lead to some small numerical differences
        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')
        print(f"Fast inference mode is enabled🏎️🏎️🏎️. TF32: {tf32}")
    else:
        print("Fast inference mode is disabled🐢🐢🐢, you may enable it by passing the '--tf32' flag!")

    
    # Set up the total_number counter
    total_samples = 0
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(args.set_grad_enabled)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    image_size = 256
    assert image_size in [256], "We provide pre-trained models for 256x256 resolutions for now."
    latent_size = image_size // 8
    model = MDTv2_XL_2(input_size=latent_size, decode_layer=4).to(device)

    state_dict = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    model = torch.compile(model)

    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    if args.start_class is not None and args.end_class is not None:
        class_labels = np.arange(args.start_class, args.end_class, 1)
    else:
        class_labels = np.arange(args.num_classes) 
    for class_label in tqdm(class_labels):
        batch_labels = torch.tensor([class_label] * args.images_per_class, device=device)
        with torch.no_grad():
            generate_samples(batch_labels, model, diffusion, vae, latent_size, device, total_samples, args.cfg_scale)
        total_samples += args.images_per_class


def generate_samples(batch_labels, model, diffusion, vae, latent_size, device, total_samples, cfg_scale=3.8, pow_scale=4.0):
    # Create sampling noise:
    # Labels to condition the model with:

    n = len(batch_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = batch_labels.clone().detach()

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.full_like(y, 1000)
    y = torch.cat([y, y_null], 0).long()
    model_kwargs = dict(y=y, cfg_scale=cfg_scale, scale_pow=pow_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save images:
    for i in range(samples.shape[0]):
        os.makedirs("samples", exist_ok=True)
        save_image(samples[i], f"samples/{total_samples + i:06d}.png", normalize=True, value_range=(-1, 1))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--start-class", type=int, default=None)
    parser.add_argument("--end-class", type=int, default=None)
    parser.add_argument("--tf32", action="store_true", help="Enable TensorFloat32 precision")
    parser.add_argument("--cfg-scale", type=float, default=3.8)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images_per_class", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--set_grad_enabled", type=bool, default=False, help="Set grad enabled or not")
    parser.add_argument("--model_path", type=str, default='mdt_xl2_v2_ckpt.pt', help="Path to the model checkpoint")
    args = parser.parse_args()
    main(args)

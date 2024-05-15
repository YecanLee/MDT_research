# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
from torchvision.utils import save_image
from masked_diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from masked_diffusion.models import MDTv2_XL_2

def main():
    # Setup PyTorch:
    torch.manual_seed(1000)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_sampling_steps = 250
    cfg_scale = 4.0
    pow_scale = 0.01 # large pow_scale increase the diversity, small pow_scale increase the quality.
    model_path = 'mdt_xl2_v2_ckpt.pt'

    # Load model:
    image_size = 256
    assert image_size in [256], "We provide pre-trained models for 256x256 resolutions for now."
    latent_size = image_size // 8
    model = MDTv2_XL_2(input_size=latent_size, decode_layer=4).to(device)

    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    model = torch.compile(model)

    model.eval()
    diffusion = create_diffusion(str(num_sampling_steps))
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    class_labels = np.arange(1000)

    batch_size = 1
    
    for i in range(0, len(class_labels), batch_size):
        batch_labels = class_labels[i:i + batch_size]
        generate_samples(batch_labels, model, diffusion, vae, latent_size, device, cfg_scale, pow_scale)
        torch.cuda.empty_cache()

def generate_samples(batch_labels, model, diffusion, vae, latent_size, device, cfg_scale=4.0, pow_scale=0.01):
    # Create sampling noise:
    # Labels to condition the model with:

    n = len(batch_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(batch_labels, device=device)

    # Sample 51 images for each class:
    z = z.repeat(51, 1, 1, 1)
    y = y.repeat(51)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.full_like(y, 1000)
    y = torch.cat([y, y_null], 0).long()
    model_kwargs = dict(y=y, cfg_scale=cfg_scale, scale_pow=pow_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Rewrite the codes to generate 51 images for each class then save them separately.
    # Save all the generated images starting from name "000000.jpg" to "050999.jpg"
    for i in range(samples.shape[0]):
        class_label = batch_labels[i // 51]
        save_image(samples[i], f"/personal_storage/scout/fid-flaws/data/gen_img_MDT/{class_label}_{i:06d}.png", normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    main()

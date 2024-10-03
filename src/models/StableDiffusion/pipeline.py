import torch
import numpy as np
from tqdm import tqdm
from src.models.StableDiffusion.sampler import DDPM

IMG_WIDTH = 512
IMG_HEIGHT = 512
VAE_LATENT_WIDTH = IMG_WIDTH // 8
VAE_LATENT_HEIGHT = IMG_HEIGHT // 8


def generate(
        prompt: str,
        uncond_prompt: str,
        input_image=None,
        strength=0.8, # how much we want to pay attention to the original image (in image-to-image)
        classifier_free_guidance=True,
        cfg_scale=7.5,
        sampler="ddpm",
        n_inference_steps=50,
        pretrained_models={},
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None,
        ):

    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = pretrained_models["clip"]
        clip = clip.to(device)

        if classifier_free_guidance:
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids  # seq len is 77
            cond_tokens = torch.Tensor(cond_tokens, dtype=torch.long, device=device)
            # convert to embeddings of size 768
            cond_context = clip(cond_tokens) # this returns (B_size, seq_len, dim)

            # same for negative prompt
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77).input_ids
            uncond_tokens = torch.Tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens) # this returns (B_size, seq_len, dim)

            context = torch.cat([cond_context, uncond_context])  # 2, 77, 768
        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            tokens = torch.Tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)  # 1, 77, 768

        to_idle(clip)

        if sampler == "ddpm":
            sampler_model = DDPM(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Invalid sampler value {sampler}. Available ones are: [ddpm]")

        latent_shape = (1, 4, VAE_LATENT_HEIGHT, VAE_LATENT_WIDTH)

        if input_image:
            encoder = pretrained_models["encoder"]
            encoder.to(device)

            input_image_tensor = torch.tensor(np.array(input_image.resize((IMG_WIDTH, IMG_HEIGHT))))
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            input_image_tensor.unsqueeze(0)  # B_size, H, W, chans
            input_image_tensor.permute(0, 3, 1, 2)  # B_size, chans, H, W, so default setup

            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
            # image through the decoder of the VAE
            latent_image = encoder(input_image, encoder_noise)

            sampler.set_strength(strength=strength)  # strength of noise (how much noise we start with)
            latent_image = sampler.add_noise(latent_image, sampler.timesteps[0])

            to_idle(encoder)
        else:
            latent_image = torch.randn(latent_shape, generator=generator, device=device)

        diffusion = pretrained_models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            t_emb = get_time_embedding(timestep).to(device)
            model_in = latent_image # B_size, 4 channels, latent_height, latent_width

            # if CFG we send latent img with and without the prompt
            if classifier_free_guidance:
                # B_size, 4 channels, latent_height, latent_width -> 2 * B_size, 4 channels, latent_height, latent_width
                model_in = model_in.repeat(2, 1, 1, 1)

            # noise predicted by UNet
            model_out = diffusion(model_in, context, time_embedding)

            if classifier_free_guidance:
                output_cond, output_uncond = model_out.chunk(2)  # split by 0th dim
                model_out = cfg_scale * (output_cond - output_uncond) + output_uncond

            # remove noise predicted by UNet
            latent_image = sampler.step(timestep, latent_image, model_out)

        to_idle(diffusion)

        # VAE decoder
        decoder = pretrained_models["decoder"]
        out_image = decoder(latent_image)

        to_idle(decoder)

        out_image_rescaled = rescale(out_image, (-1, 1), (0, 255))
        out_image_rescaled.permute(0, 2, 3, 1) # back to B_Size, H, W, Channels
        out_image_rescaled.to("cpu", torch.uint8).numpy() # out as np array
        return out_image_rescaled[0]

def rescale(x: torch.Tensor, old_range: tuple[int, int], new_range: tuple[int, int], clamp: bool=False) -> torch.Tensor:
    old_min, old_max = old_range
    new_min, new_max = new_range
    scale_fac = (new_max - new_min) / (old_max - old_min)
    x -= old_min
    x *= scale_fac
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # transformer positional encoding formula
    frequencies = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32).unsqueeze() * frequencies[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
"""ComfyUI sampler node for the 4-step MiniMax-H3 DMAD student."""

from __future__ import annotations

import torch
from tqdm.auto import trange


@torch.no_grad()
def dmad_renoise_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    """DMAD's DMD-style step: predict x0, then add fresh noise at next sigma."""
    import comfy.k_diffusion.sampling

    extra_args = {} if extra_args is None else extra_args
    noise_sampler = comfy.k_diffusion.sampling.default_noise_sampler(
        x, seed=extra_args.get("seed")
    )
    s_in = x.new_ones((x.shape[0],))
    for index in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        denoised = model(x, sigma * s_in, **extra_args)
        if float(sigma_next) == 0.0:
            x = denoised
        else:
            noise = noise_sampler(sigma, sigma_next)
            x = (1.0 - sigma_next) * denoised + sigma_next * noise
        if callback is not None:
            callback({
                "i": index,
                "denoised": denoised,
                "x": x,
                "sigma": sigma,
                "sigma_hat": sigma,
            })
    return x


class MiniMaxH3DMADSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = (
        "DMAD 4-step re-noise sampler. Use a simple scheduler with 4 steps and "
        "MiniMaxH3SigmaShift set to video=12, audio=2."
    )

    def get_sampler(self):
        import comfy.samplers

        return (comfy.samplers.KSAMPLER(dmad_renoise_sampler),)


NODE_CLASS_MAPPINGS = {"MiniMaxH3DMADSampler": MiniMaxH3DMADSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3DMADSampler": "MiniMax-H3 DMAD Sampler (4-step)"}

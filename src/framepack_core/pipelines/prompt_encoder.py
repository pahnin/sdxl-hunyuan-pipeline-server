import torch
from ..diffusers_helper.hunyuan import (
    encode_prompt_conds,
    crop_or_pad_yield_mask,
)


# =============================================================================
# PROMPT ENCODER
# =============================================================================
class PromptEncoder:
    """Handles prompt encoding with caching."""

    def __init__(
        self,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        prompt_embedding_cache,
    ):
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.cache = prompt_embedding_cache

    @torch.no_grad()
    def encode_prompt(self, prompt: str, target_device):
        """
        Retrieve prompt embeddings from cache or encode them.
        Returns embeddings on target_device.
        """
        if prompt in self.cache:
            print(f"Cache hit for prompt: {prompt[:60]}...")
            llama_vec_cpu, llama_mask_cpu, clip_l_pooler_cpu = self.cache[prompt]
            return (
                llama_vec_cpu.to(target_device),
                (
                    llama_mask_cpu.to(target_device)
                    if llama_mask_cpu is not None
                    else None
                ),
                clip_l_pooler_cpu.to(target_device),
            )

        print(f"Cache miss for prompt: {prompt[:60]}...")
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt,
            self.text_encoder,
            self.text_encoder_2,
            self.tokenizer,
            self.tokenizer_2,
        )
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)

        # Store CPU copies in cache
        self.cache[prompt] = (
            llama_vec.cpu(),
            llama_attention_mask.cpu() if llama_attention_mask is not None else None,
            clip_l_pooler.cpu(),
        )

        return llama_vec, llama_attention_mask, clip_l_pooler

    def encode_multiple_prompts(self, prompts: list, target_device):
        """Encode multiple unique prompts and return a dictionary."""
        encoded = {}
        for prompt in prompts:
            encoded[prompt] = self.encode_prompt(prompt, target_device)
        return encoded

    def encode_negative_prompt(
        self, n_prompt: str, cfg: float, reference_prompt: str, target_device
    ):
        """Encode negative prompt or return zeros if cfg == 1."""
        if cfg == 1:
            ref_vec, ref_mask, ref_pooler = self.cache.get(
                reference_prompt, self.encode_prompt(reference_prompt, target_device)
            )
            return (
                torch.zeros_like(ref_vec),
                torch.zeros_like(ref_mask),
                torch.zeros_like(ref_pooler),
            )

        n_prompt_str = str(n_prompt) if n_prompt is not None else ""
        return self.encode_prompt(n_prompt_str, target_device)

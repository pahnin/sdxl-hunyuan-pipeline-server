# =============================================================================
# PROMPT BLENDER
# =============================================================================
class PromptBlender:
    """Handles prompt blending logic for multi-section prompts."""

    def __init__(
        self, prompt_sections: list, encoded_prompts: dict, blend_sections: int
    ):
        self.prompt_sections = prompt_sections
        self.encoded_prompts = encoded_prompts
        self.blend_sections = self._parse_blend_sections(blend_sections)
        self.prompt_change_indices = self._build_change_indices()

    @staticmethod
    def _parse_blend_sections(blend_sections) -> int:
        """Parse blend_sections parameter safely."""
        try:
            return int(blend_sections)
        except ValueError:
            print(
                f"Warning: blend_sections ('{blend_sections}') is not valid. Disabling blending."
            )
            return 0

    def _build_change_indices(self) -> list:
        """Build list of (section_idx, prompt) for each prompt change."""
        indices = []
        last_prompt = None
        for idx, section in enumerate(self.prompt_sections):
            if section.prompt != last_prompt:
                indices.append((idx, section.prompt))
                last_prompt = section.prompt
        return indices

    def get_prompt_for_section(self, section_idx: int, current_time_position: float):
        """
        Get the appropriate prompt(s) and blending alpha for a section.
        Returns (llama_vec, llama_attention_mask, clip_l_pooler).
        """
        # Find current prompt based on time
        current_prompt = self.prompt_sections[0].prompt
        for section in self.prompt_sections:
            if section.start_time <= current_time_position and (
                section.end_time is None or current_time_position < section.end_time
            ):
                current_prompt = section.prompt
                break

        # Check if we should blend
        blend_alpha, prev_prompt, next_prompt = self._calculate_blend(section_idx)

        if blend_alpha is not None and prev_prompt != next_prompt:
            # Blend embeddings
            prev_vec, prev_mask, prev_pooler = self.encoded_prompts[prev_prompt]
            next_vec, next_mask, next_pooler = self.encoded_prompts[next_prompt]

            llama_vec = (1 - blend_alpha) * prev_vec + blend_alpha * next_vec
            llama_attention_mask = prev_mask
            clip_l_pooler = (1 - blend_alpha) * prev_pooler + blend_alpha * next_pooler

            print(
                f"Blending prompts: '{prev_prompt[:30]}...' -> '{next_prompt[:30]}...', "
                f"alpha={blend_alpha:.2f}"
            )
            return llama_vec, llama_attention_mask, clip_l_pooler

        # No blending - return current prompt embeddings
        return self.encoded_prompts[current_prompt]

    def _calculate_blend(self, section_idx: int):
        """Calculate blend alpha and prompts for the current section."""
        if (
            self.blend_sections <= 0
            or not self.prompt_change_indices
            or len(self.prompt_sections) <= 1
        ):
            return None, None, None

        for i, (change_idx, prompt) in enumerate(self.prompt_change_indices):
            if section_idx < change_idx:
                prev_prompt = self.prompt_change_indices[i - 1][1] if i > 0 else prompt
                next_prompt = prompt

                if change_idx <= section_idx < change_idx + self.blend_sections:
                    blend_alpha = (section_idx - change_idx + 1) / self.blend_sections
                    return blend_alpha, prev_prompt, next_prompt
                break
            elif section_idx == change_idx:
                if i > 0:
                    prev_prompt = self.prompt_change_indices[i - 1][1]
                    next_prompt = prompt
                    blend_alpha = 1.0 / self.blend_sections
                    return blend_alpha, prev_prompt, next_prompt
                break

        return None, None, None

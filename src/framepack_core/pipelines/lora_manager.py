from .. import DUMMY_LORA_NAME


# =============================================================================
# LORA MANAGER
# =============================================================================
class LoRAManager:
    """Manages LoRA loading, unloading, and filtering."""

    @staticmethod
    def filter_dummy_loras(selected_loras) -> list:
        """Filter out dummy LoRAs from selection."""
        if isinstance(selected_loras, list):
            filtered = [lora for lora in selected_loras if lora != DUMMY_LORA_NAME]
            if DUMMY_LORA_NAME in selected_loras:
                print(f"Worker: Filtered out '{DUMMY_LORA_NAME}' from selected LoRAs")
            return filtered
        elif selected_loras is not None and selected_loras != DUMMY_LORA_NAME:
            return [selected_loras]
        return []

    @staticmethod
    def verify_lora_state(generator):
        """Verify and print LoRA state of the transformer."""
        if not generator or not generator.transformer:
            return

        has_loras = False

        if hasattr(generator.transformer, "peft_config"):
            adapter_names = (
                list(generator.transformer.peft_config.keys())
                if generator.transformer.peft_config
                else []
            )
            if adapter_names:
                has_loras = True
                print(f"Transformer has LoRAs: {', '.join(adapter_names)}")
            else:
                print("Transformer has no LoRAs in peft_config")
        else:
            print("Transformer has no peft_config attribute")

        # Check for LoRA modules
        for name, module in generator.transformer.named_modules():
            if hasattr(module, "lora_A") and module.lora_A:
                has_loras = True
            if hasattr(module, "lora_B") and module.lora_B:
                has_loras = True

        if not has_loras:
            print("No LoRA components found in transformer")

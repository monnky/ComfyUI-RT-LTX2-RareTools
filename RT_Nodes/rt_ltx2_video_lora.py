import torch
import comfy.utils
import comfy.sd
import folder_paths

class RT_LTX2_Video_LoRA_Injector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": ("MODEL",),
                "lora_filename": (folder_paths.get_filename_list("loras"), ),
                "injection_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_video_weights"
    CATEGORY = "RareTutor"
    TITLE = "RT LTX-2 Video-Only LoRA"

    def apply_video_weights(self, base_model, lora_filename, injection_strength):
        if injection_strength == 0:
            return (base_model,)

        target_path = folder_paths.get_full_path("loras", lora_filename)
        raw_weights = comfy.utils.load_torch_file(target_path, safe_load=True)
        
        # Filter keys to exclude audio-specific weights
        video_only_weights = {}
        for weight_key, weight_tensor in raw_weights.items():
            # Targets audio blocks and vocoder pathways in LTX-2
            if not any(keyword in weight_key.lower() for keyword in ["audio", "vocoder", "speech", "conditioning.audio"]):
                video_only_weights[weight_key] = weight_tensor
            else:
                print(f"[RT-LTX2 Video LoRA] Stripped audio-related weight: {weight_key}")

        patched_model, _ = comfy.sd.load_lora_for_models(base_model, None, video_only_weights, injection_strength, 0)
        
        return (patched_model,)

# ── ComfyUI Registration ─────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "RT_LTX2_Video_LoRA_Injector": RT_LTX2_Video_LoRA_Injector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RT_LTX2_Video_LoRA_Injector": "RT LTX-2 Video-Only LoRA"
}
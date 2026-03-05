import torch
import comfy.model_management

def rt_inject_fp8_optimization(target_model, feedforward_splits):
    """
    Applies the Raw Math patch. Best for FP8/BF16/FP32.
    It decomposes the layer to minimize intermediate tensor creation.
    """
    print(f"[RT-LTX2] Mode: FP8/Native | Chunks: {feedforward_splits}")
    
    # Access the underlying diffusion model
    core_unet_model = target_model.model.diffusion_model

    def rt_sliced_forward_fp8(self_layer, input_tensor):
        if feedforward_splits <= 1:
            return self_layer.original_rt_forward(input_tensor)

        B, S, C = input_tensor.shape
        slice_size = (S + feedforward_splits - 1) // feedforward_splits
        processed_slices = []

        for idx in range(0, S, slice_size):
            end_bound = min(idx + slice_size, S)
            tensor_slice = input_tensor[:, idx:end_bound, :]
            
            # RAW MATH OPTIMIZATION
            if hasattr(self_layer, "net"):
                # Project Up -> GELU -> Project Down
                out_slice = self_layer.net[0](tensor_slice) 
                out_slice = self_layer.net[1](out_slice)
                out_slice = self_layer.net[2](out_slice)
                processed_slices.append(out_slice)
            else:
                # Fallback if structure is weird
                processed_slices.append(self_layer.original_rt_forward(tensor_slice))

        return torch.cat(processed_slices, dim=1)

    # Apply Patch
    modified_count = 0
    for block_layer in core_unet_model.transformer_blocks:
        if hasattr(block_layer, "ff") and block_layer.ff is not None:
            if not hasattr(block_layer.ff, "original_rt_forward"):
                block_layer.ff.original_rt_forward = block_layer.ff.forward
            
            block_layer.ff.forward = lambda t, layer_ref=block_layer.ff: rt_sliced_forward_fp8(layer_ref, t)
            modified_count += 1
            
    print(f"[RT-LTX2] Patched {modified_count} blocks with FP8 Logic.")
    return target_model


def rt_inject_gguf_blackbox(target_model, feedforward_splits):
    """
    Applies the Blackbox patch. Best for GGUF.
    It feeds slices to the layer, letting the GGUF node handle dequantization.
    """
    print(f"[RT-LTX2] Mode: GGUF Safe | Chunks: {feedforward_splits}")

    core_unet_model = target_model.model.diffusion_model

    def rt_sliced_forward_gguf(self_layer, input_tensor):
        if feedforward_splits <= 1:
            return self_layer.original_rt_forward(input_tensor)

        B, S, C = input_tensor.shape
        slice_size = (S + feedforward_splits - 1) // feedforward_splits
        processed_slices = []

        for idx in range(0, S, slice_size):
            end_bound = min(idx + slice_size, S)
            tensor_slice = input_tensor[:, idx:end_bound, :]
            
            # BLACKBOX OPTIMIZATION
            out_slice = self_layer.original_rt_forward(tensor_slice)
            processed_slices.append(out_slice)

        return torch.cat(processed_slices, dim=1)

    # Apply Patch
    modified_count = 0
    for block_layer in core_unet_model.transformer_blocks:
        if hasattr(block_layer, "ff") and block_layer.ff is not None:
            if not hasattr(block_layer.ff, "original_rt_forward"):
                block_layer.ff.original_rt_forward = block_layer.ff.forward
            
            block_layer.ff.forward = lambda t, layer_ref=block_layer.ff: rt_sliced_forward_gguf(layer_ref, t)
            modified_count += 1

    print(f"[RT-LTX2] Patched {modified_count} blocks with GGUF Logic.")
    return target_model


class RT_LTX2_Extended_Duration_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": ("MODEL",),
                "weight_format": (["FP8/BF16 (Standard)", "GGUF (Quantized)"],),
                "feedforward_splits": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
                "activate_optimization": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_duration_fix"
    CATEGORY = "RareTutor"
    TITLE = "RT LTX-2 Long Video"

    def apply_duration_fix(self, base_model, weight_format, feedforward_splits, activate_optimization):
        if not activate_optimization:
            return (base_model,)

        # Clone to keep things clean
        cloned_model = base_model.clone()

        # Router Logic
        if weight_format == "FP8/BF16 (Standard)":
            cloned_model = rt_inject_fp8_optimization(cloned_model, feedforward_splits)
        else:
            cloned_model = rt_inject_gguf_blackbox(cloned_model, feedforward_splits)
        
        return (cloned_model,)


# ── ComfyUI Registration ─────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "RT_LTX2_Extended_Duration_Node": RT_LTX2_Extended_Duration_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RT_LTX2_Extended_Duration_Node": "RT LTX-2 Long Video"
}
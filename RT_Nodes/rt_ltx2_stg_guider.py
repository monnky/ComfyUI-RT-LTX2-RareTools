import torch
import comfy.samplers
from comfy.model_patcher import ModelPatcher
from typing import List

# ==============================================================================
# STG HELPERS
# ==============================================================================
class STGFlag:
    def __init__(self, do_skip: bool = False, skip_layers: List[int] = None):
        self.do_skip = do_skip
        self.skip_layers = skip_layers if skip_layers else []

class STGBlockWrapper:
    def __init__(self, stg_flag: STGFlag, block_idx: int):
        self.stg_flag = stg_flag
        self.block_idx = block_idx

    def __call__(self, args, extra_args):
        # 1. SKIP LOGIC (STG Perturbed Pass)
        if self.stg_flag.do_skip and self.block_idx in self.stg_flag.skip_layers:
            # ComfyUI's LTXV architecture expects the block to return a dictionary: {"img": (vx, ax)}
            # 'args' is a tuple of all positional inputs: (hidden_states, context, ...)
            # We must return args[0], which is the hidden_states dictionary, unchanged.
            if isinstance(args, tuple) and len(args) > 0:
                return args[0]
            return args

        # 2. CRASH FIX: Safely handle 'NoneType' transformer_options
        # transformer_options lives inside extra_args in ComfyUI's dit patcher.
        if "transformer_options" not in extra_args or extra_args["transformer_options"] is None:
            extra_args["transformer_options"] = {}
        
        # 3. STANDARD EXECUTION: Let ComfyUI handle the internal unpacking
        return extra_args["original_block"](args)

# ==============================================================================
# STG GUIDER LOGIC
# ==============================================================================
class STGGuider(comfy.samplers.CFGGuider):
    def __init__(self, model: ModelPatcher, stg_scale: float, rescale: float, skip_blocks: List[int]):
        model = model.clone()
        super().__init__(model)
        self.stg_scale = stg_scale
        self.rescale_factor = rescale
        self.stg_flag = STGFlag(do_skip=False, skip_layers=skip_blocks)
        self.patch_model(model, self.stg_flag)

    @classmethod
    def patch_model(cls, model: ModelPatcher, stg_flag: STGFlag):
        transformer_blocks = cls.get_transformer_blocks(model)
        for i, _ in enumerate(transformer_blocks):
            model.set_model_patch_replace(
                STGBlockWrapper(stg_flag, i), "dit", "double_block", i
            )
            # Catch single_blocks just in case the model architecture differentiates them
            model.set_model_patch_replace(
                STGBlockWrapper(stg_flag, i), "dit", "single_block", i
            )

    @staticmethod
    def get_transformer_blocks(model: ModelPatcher):
        diffusion_model = model.get_model_object("diffusion_model")
        if hasattr(diffusion_model, "transformer_blocks"):
            return diffusion_model.transformer_blocks
        elif hasattr(diffusion_model, "transformer"):
            return diffusion_model.transformer.transformer_blocks
        return model.get_model_object("diffusion_model.transformer_blocks")

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        pos_cond = self.conds.get("positive")
        neg_cond = self.conds.get("negative")
        
        if "transformer_options" not in model_options:
            model_options["transformer_options"] = {}

        # 1. Standard Pass (Attention ON)
        self.stg_flag.do_skip = False
        (pos_pred, neg_pred) = comfy.samplers.calc_cond_batch(
            self.inner_model, [pos_cond, neg_cond], x, timestep, model_options
        )

        # 2. Perturbed Pass (Attention OFF for skipped layers)
        if self.stg_scale != 0:
            self.stg_flag.do_skip = True
            ptb_options = model_options.copy()
            (perturbed_pred,) = comfy.samplers.calc_cond_batch(
                self.inner_model, [pos_cond], x, timestep, ptb_options
            )
            self.stg_flag.do_skip = False
        else:
            perturbed_pred = pos_pred

        # 3. Combine: CFG + STG
        cfg_result = neg_pred + self.cfg * (pos_pred - neg_pred)
        stg_residual = pos_pred - perturbed_pred
        final_pred = cfg_result + (self.stg_scale * stg_residual)

        # 4. Rescale (Phi-Rescale)
        if self.rescale_factor > 0:
            std_pos = pos_pred.std()
            std_final = final_pred.std()
            if std_final > 0 and std_pos > 0:
                factor = std_pos / std_final
                final_pred = final_pred * (factor * self.rescale_factor + (1.0 - self.rescale_factor))

        return final_pred

# ==============================================================================
# NODE MAPPING
# ==============================================================================
class STGGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "stg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 200.0, "step": 0.1, "label": "STG Scale"}),
                "rescale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "skip_blocks": ("STRING", {"default": "19,20,21", "multiline": False}), 
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "RT_Nodes"

    def get_guider(self, model, positive, negative, cfg, stg, rescale, skip_blocks):
        try:
            skip_list = [int(x.strip()) for x in skip_blocks.split(",") if x.strip()]
        except:
            print("Warning: STG Guider - Invalid skip_blocks format. Using default []")
            skip_list = []
        guider = STGGuider(model, stg, rescale, skip_list)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        return (guider,)

NODE_CLASS_MAPPINGS = {"RT_LTX2_STG_Guider": STGGuiderNode}
NODE_DISPLAY_NAME_MAPPINGS = {"RT_LTX2_STG_Guider": "RT LTX-2 STG Guider"}
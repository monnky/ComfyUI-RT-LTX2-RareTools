from .RT_Nodes.rt_ltx2_self_refining_patch import NODE_CLASS_MAPPINGS as PATCH_NODES, NODE_DISPLAY_NAME_MAPPINGS as PATCH_NAMES
from .RT_Nodes.rt_ltx2_stg_guider import NODE_CLASS_MAPPINGS as STG_NODES, NODE_DISPLAY_NAME_MAPPINGS as STG_NAMES
from .RT_Nodes.rt_ltx2_royal_prompt import NODE_CLASS_MAPPINGS as PROMPT_NODES, NODE_DISPLAY_NAME_MAPPINGS as PROMPT_NAMES
from .RT_Nodes.rt_ltx2_longvideo import NODE_CLASS_MAPPINGS as LONG_VIDEO_NODES, NODE_DISPLAY_NAME_MAPPINGS as LONG_VIDEO_NAMES
from .RT_Nodes.rt_ltx2_video_lora import NODE_CLASS_MAPPINGS as VIDEO_LORA_NODES, NODE_DISPLAY_NAME_MAPPINGS as VIDEO_LORA_NAMES


NODE_CLASS_MAPPINGS = {
    **PATCH_NODES, 
    **STG_NODES,
    **PROMPT_NODES,
    **LONG_VIDEO_NODES, # Added the new Long Video node
    **VIDEO_LORA_NODES, # Added the new Video-Only LoRA node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **PATCH_NAMES, 
    **STG_NAMES,
    **PROMPT_NAMES,
    **LONG_VIDEO_NAMES, # Added the new Long Video node names
    **VIDEO_LORA_NAMES, # Added the new Video-Only LoRA node names
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

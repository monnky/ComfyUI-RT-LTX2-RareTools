import os
import gc
import base64
import io
import re
import numpy as np
import torch
from PIL import Image
import folder_paths

# ── Dependency Check ─────────────────────────────────────────────────────────
try:
    import llama_cpp
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    print("\n\033[91m[RT-LTX-2] ERROR: 'llama-cpp-python' is missing.\033[0m")
    print("Please run: pip install llama-cpp-python\n")
    raise

class RT_LTX2_RoyalPrompt:
    
    # ── SYSTEM PROMPT: Hyper-Literal LTX-2 Optimization ─────────────
    SYSTEM_PROMPT = """You are an elite video prompt engineer for LTX-2.3. LTX-2 is a fragile AI model that hallucinates if given poetic or complex instructions. 

CRITICAL FORMATTING RULES:
1. STYLE TAG FIRST: The absolute first line MUST be: [Style : <3D Animation OR Live-Action>, <texture>, <lighting>].
2. NO CHATTY FILLER: NEVER start with "Here is the prompt". Start instantly with the [Style : ...] tag.
3. AMBIENT TAG: The final line MUST be the [AMBIENT: ...] audio tag.

CRITICAL LTX-2 PHYSICS RULES (PREVENT NONSENSE):
- BE ULTRA-LITERAL: Do not write "he tells a story". Write "his mouth opens and closes, his hand raises". 
- KEEP MOTION SIMPLE: Limit the scene to 1 or 2 basic, slow movements. (e.g., "The man walks forward. The boy looks up.")
- NO POETRY: Do not use words like "magical," "whimsical," "epic," or "passionate." Describe literal visual data.
- NO OVERLAPPING ACTION: Do not describe 5 things happening at once. LTX-2 will fail.
- NO FAST CAMERA MOVES: Stick to "static camera", "slow pan left", or "slow push in".

=== PERFECT OUTPUT EXAMPLES ===

EXAMPLE 1 (For a 3D/Cartoon Image):
[Style : 3D Animation, detailed textures, soft diffuse sunlight]
A wide static shot of a park. An old man walks forward slowly. His mouth moves. He raises his right hand. A young boy walks next to him. The boy looks up at the man. The background trees are out of focus.
[AMBIENT: wind blowing leaves, birds chirping]

EXAMPLE 2 (For a Live Action Image):
[Style : Photorealistic Live-Action, cinematic lighting, gritty textures]
A medium shot tracking backward. A cyborg walks forward down a wet street. Rain falls from the sky. Water drips down the cyborg's metal faceplate. Neon lights reflect on the wet pavement. 
[AMBIENT: heavy rain, distant sirens]
==============================="""

    @staticmethod
    def get_supported_models():
        unique_models = set()
        search_dirs = []
        if "text_encoders" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("text_encoders"))
        if "llm" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("llm"))
        if "unet" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("unet"))
            
        for base_path in search_dirs:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.lower().endswith((".gguf", ".safetensors")):
                            rel_path = os.path.relpath(os.path.join(root, file), base_path)
                            rel_path = rel_path.replace("\\", "/")
                            unique_models.add(rel_path)
                            
        if not unique_models:
            return ["No supported models (.gguf or .safetensors) found in text_encoders/llm"]
        return sorted(list(unique_models))

    @classmethod
    def INPUT_TYPES(s):
        valid_models = s.get_supported_models()
        return {
            "required": {
                "image": ("IMAGE",),
                "llm_model": (valid_models, {"default": valid_models[0]}), 
                "vision_model": (valid_models, {"default": valid_models[0]}),
                "user_input": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe action... (Keep it very simple for LTX-2)"
                }),
                "max_tokens": (["256", "512", "800", "1024", "2048"], {"default": "1024"}),
                "creativity": (["0.7 - Literal", "0.9 - Balanced", "1.1 - Artistic"], {"default": "0.7 - Literal"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "debug_console": ("BOOLEAN", {"default": True}), 
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "n_ctx": ("INT", {"default": 8192, "min": 2048, "max": 32768}),
                "frame_count": ("INT", {"default": 120, "min": 24, "max": 960}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("PROMPT", "PREVIEW", "FRAMES")
    FUNCTION = "generate"
    CATEGORY = "RareTutor"

    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.loaded_model_path = None
        self.loaded_vision_path = None
        self.banned_tokens = {}

    def _tensor_to_base64(self, image_tensor):
        if len(image_tensor.shape) == 4:
            img = image_tensor[0].cpu().numpy()
        else:
            img = image_tensor.cpu().numpy()
            
        img = (img * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def _find_absolute_path(self, filename, search_dirs):
        """Robustly searches for the exact absolute path of the selected model."""
        # First, try direct join (handles relative paths returned by dropdown)
        for base_path in search_dirs:
            direct_path = os.path.join(base_path, filename)
            if os.path.exists(direct_path):
                return direct_path
                
        # If not found directly, deeply scan all subfolders
        for base_path in search_dirs:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        # Match if the end of the full path matches what the dropdown provided
                        full_path = os.path.join(root, file)
                        if file == filename or full_path.replace("\\", "/").endswith(filename.replace("\\", "/")):
                            return full_path
        return None

    def load_model(self, llm_name, vision_name, n_ctx):
        if llm_name.lower().endswith(".safetensors") or vision_name.lower().endswith(".safetensors"):
            raise ValueError("SAFETENSORS_ERROR")

        search_dirs = []
        if "text_encoders" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("text_encoders"))
        if "llm" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("llm"))
        if "unet" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("unet"))

        # Robust Deep Scan for files
        llm_path = self._find_absolute_path(llm_name, search_dirs)
        vision_path = self._find_absolute_path(vision_name, search_dirs)
        
        if not llm_path: raise FileNotFoundError(f"LLM '{llm_name}' not found. Checked folders and subfolders.")
        if not vision_path: raise FileNotFoundError(f"Vision '{vision_name}' not found. Checked folders and subfolders.")
        
        if (self.llm is not None and self.loaded_model_path == llm_path and self.loaded_vision_path == vision_path):
            return

        self.unload_model()
        print(f"\n[RT-LTX-2] Loading Models into VRAM...")
        print(f"  -> Text Path: {llm_path}")
        print(f"  -> Vision Path: {vision_path}")
        
        try:
            self.chat_handler = Llava15ChatHandler(clip_model_path=vision_path)
            self.llm = Llama(
                model_path=llm_path,
                chat_handler=self.chat_handler,
                n_gpu_layers=-1,
                n_ctx=n_ctx,
                logits_all=True,  
                verbose=False
            )
            self.loaded_model_path = llm_path
            self.loaded_vision_path = vision_path
            
            banned_strings = ["ASSISTANT:", "Assistant:", "USER:", "User:", "Here is the", "Okay,", "Sure,"]
            self.banned_tokens = {}
            for bad_str in banned_strings:
                tokens = self.llm.tokenize(bad_str.encode('utf-8'), add_bos=False)
                if len(tokens) > 0:
                    self.banned_tokens[tokens[0]] = -100.0
                    
        except Exception as e:
            print(f"[RT-LTX-2] CRITICAL ERROR: {e}")
            self.unload_model()
            raise RuntimeError(f"Load failed: {e}")

    def unload_model(self):
        if self.llm: del self.llm
        if self.chat_handler: del self.chat_handler
        self.llm = None
        self.chat_handler = None
        gc.collect()
        torch.cuda.empty_cache()

    def _clean_output(self, text):
        text = re.sub(r"^(Sure|Okay|Here is|Here's).*?:\n+", "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"^\s*\**Here is a cinematic.*?\**\s*\n+", "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"^\x60\x60\x60(?:text)?\n(.*?)\n\x60\x60\x60$", r"\1", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"\x60\x60\x60[a-z]*\n", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\x60\x60\x60", "", text)
        text = re.sub(r"\*\*(\[.*?\])\*\*", r"\1", text)
        
        ambient_matches = list(re.finditer(r"\[AMBIENT:.*?\]", text, re.IGNORECASE))
        if ambient_matches:
            last_match = ambient_matches[-1]
            text = text[:last_match.end()]
        else:
            text = re.sub(r"(ASSISTANT|USER REQUEST|USER:|\[SCENE START\]).*", "", text, flags=re.IGNORECASE | re.DOTALL)
            
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
            
        text = text.strip()

        if "[Style" not in text and "[style" not in text:
            text = "[Style : highly detailed, accurate to visual reference]\n" + text
        else:
            text = re.sub(r"^.*?(\[Style\s*:)", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
            
        return text.strip()

    def generate(self, image, llm_model, vision_model, user_input, max_tokens, creativity, seed, debug_console, keep_model_loaded, n_ctx, frame_count):
        
        try:
            self.load_model(llm_model, vision_model, n_ctx)
        except ValueError as e:
            if str(e) == "SAFETENSORS_ERROR":
                error_msg = ("[ERROR] You selected a .safetensors model. Please select a `.gguf` file.")
                return (error_msg, error_msg, frame_count)
            else:
                raise e
            
        base64_image = self._tensor_to_base64(image)
        token_val = int(max_tokens.split(" - ")[0]) if " - " in max_tokens else int(max_tokens)
        
        temp = 0.6 if "Literal" in creativity else float(creativity.split(" - ")[0])
        
        duration_secs = max(1, frame_count // 24)
        pacing_rule = (
            f"CRITICAL: The target video is {duration_secs} seconds long. "
            f"Keep the description ULTRA-SIMPLE and literal. Describe ONLY what happens visually. "
            f"Write exactly 3 to 5 short sentences."
        )
        
        clean_user_input = user_input.strip()
        is_empty_input = not clean_user_input or "Describe style" in clean_user_input or "Pixar-style" in clean_user_input
        
        blueprint = (
            f"{pacing_rule}\n\n"
            "REQUIRED OUTPUT FORMAT:\n"
            "Line 1: [Style : <MUST STATE IF 3D ANIMATION, PHOTOREALISTIC, OR 2D ANIME>, <texture, lighting>]\n"
            "Line 2+: <Ultra-literal, simple scene description>\n"
            "Final Line: [AMBIENT: <soundscape>]\n\n"
            "START IMMEDIATELY with the '[Style : ' tag."
        )

        if is_empty_input:
            final_prompt = f"Analyze the image and write an ultra-literal, simple prompt describing the exact visual mechanics of the scene.\n\n{blueprint}"
        else:
            final_prompt = f"USER REQUEST:\n'{clean_user_input}'\n\nWrite an ultra-literal, simple prompt based on the image and request.\n\n{blueprint}"
        
        user_content_block = [
            {"type": "text", "text": final_prompt},
            {"type": "image_url", "image_url": {"url": base64_image}}
        ]

        stop_tokens = ["<end_of_turn>", "<eos>", "<|eot_id|>", "User:", "ASSISTANT:", "Assistant:", "REAL TASK:", "USER REQUEST:", "**USER**", "====="]

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_content_block}
                ],
                max_tokens=token_val,
                temperature=temp,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=stop_tokens,
                logit_bias=self.banned_tokens,
                seed=seed if seed != -1 else None
            )
            
            raw_result = response['choices'][0]['message']['content'].strip()
            final_result = self._clean_output(raw_result)
            
            if not final_result or final_result == "":
                final_result = f"[Style : highly detailed]\n{user_input}\n\n[AMBIENT: subtle sounds]"

        except Exception as e:
            final_result = f"Error: {e}"

        if not keep_model_loaded:
            self.unload_model()

        return (final_result, final_result, frame_count)

NODE_CLASS_MAPPINGS = {
    "RT_LTX2_RoyalPrompt": RT_LTX2_RoyalPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RT_LTX2_RoyalPrompt": "RT-LTX-2 Royal Prompt by RareTutor"
}
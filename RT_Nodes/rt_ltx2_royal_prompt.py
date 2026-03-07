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
    
    # ── SYSTEM PROMPT: Built for LTX-2.3 Capabilities ────────────────────────
    SYSTEM_PROMPT = """You are a master cinematic director and video scene describer for LTX-2.3.

INSTRUCTIONS:
1. ANALYZE the image for spatial layout (left/right, foreground/background), textures, and lighting.
2. BE HIGHLY SPECIFIC & AMBITIOUS: Combine detailed environments with character performance. Use the user's text strictly.
3. USE VERBS & AVOID STATIC PROMPTS: Describe action. Specify who moves, what moves, how they move, and camera motion (e.g., "The camera tracks right as his coat flaps"). 
4. TEXTURES & MATERIALS: Explicitly describe fabric types, hair texture, surface finish, environmental wear, and edge detail.
5. LONG SHOT STRUCTURE (Write as ONE flowing paragraph):
   - Scene Header: Establish place and time.
   - Atmosphere & Textures: Tone, lighting, weather.
   - Camera & Blocking: Smooth motion. Direct the spatial relationships (distance, facing toward/away).
   - Soft Closing Action: End with a lingering, soft motion (e.g., 'the camera slowly drifts upward').
6. NO INTERNAL STATES: Describe physical cues instead of internal feelings.
7. DIALOGUE & AUDIO:
   - If there is a conversation, preserve speaker names like a screenplay (e.g., Reporter: "Words". Punch: "Words"). DO NOT use bolding or markdown for names.
   - ALWAYS end with an audio tag describing environmental audio, tone, and intensity. Example: [AMBIENT: low, pulsing energy hum, sharp metallic alarm]
8. STRICT OUTPUT FORMAT: Write ONLY the prompt paragraph followed by the audio tag. NO conversational filler."""

    @staticmethod
    def get_gguf_models():
        unique_models = set()
        paths = folder_paths.get_folder_paths("text_encoders")
        for path in paths:
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.lower().endswith(".gguf"):
                        unique_models.add(file)
        if not unique_models:
            return ["No .gguf files found in text_encoders"]
        return sorted(list(unique_models))

    @classmethod
    def INPUT_TYPES(s):
        valid_ggufs = s.get_gguf_models()
        return {
            "required": {
                "image": ("IMAGE",),
                "llm_model": (valid_ggufs, {"default": valid_ggufs[0]}), 
                "vision_model": (valid_ggufs, {"default": valid_ggufs[0]}),
                "user_input": ("STRING", {
                    "multiline": True,
                    "default": "Pixar-style animation...",
                    "placeholder": "Describe style, action, and dialogue..."
                }),
                "max_tokens": (["256", "512", "800", "1024", "2048"], {"default": "800"}),
                "creativity": (["0.7 - Literal", "0.9 - Balanced", "1.1 - Artistic"], {"default": "0.9 - Balanced"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "debug_console": ("BOOLEAN", {"default": True}), # NEW: Debug Toggle
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

    def _tensor_to_base64(self, image_tensor):
        # Robust check to ensure we only grab the first frame if fed a batch
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

    def load_model(self, llm_name, vision_name, n_ctx):
        paths = folder_paths.get_folder_paths("text_encoders")
        llm_path = None
        vision_path = None
        for path in paths:
            potential_llm = os.path.join(path, llm_name)
            potential_vis = os.path.join(path, vision_name)
            if os.path.exists(potential_llm): llm_path = potential_llm
            if os.path.exists(potential_vis): vision_path = potential_vis
        
        if not llm_path: raise FileNotFoundError(f"LLM '{llm_name}' not found.")
        if not vision_path: raise FileNotFoundError(f"Vision '{vision_name}' not found.")
        
        if (self.llm is not None and self.loaded_model_path == llm_path and self.loaded_vision_path == vision_path):
            return

        self.unload_model()
        print(f"\n[RT-LTX-2] Loading Models into VRAM...")
        print(f"  -> Text: {llm_name}")
        print(f"  -> Vision: {vision_name}")
        
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
        except Exception as e:
            print(f"[RT-LTX-2] CRITICAL ERROR during model load: {e}")
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
        # Safely remove chatty prefixes without destroying the actual prompt
        patterns = [
            r"^Okay, here's a description.*?:",
            r"^Here's a description.*?:",
            r"^Here is the prompt.*?:",
            r"^Sure, here is.*?:",
            r"REAL TASK:.*", r"USER:.*", r"ACTION:.*", r"RULES:.*", r"\[INSTRUCTION:\]",
            r"\[SCENE START\]", r"\[SCENE END\]"
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)
            
        # Kill Hallucination Loops (If the AI starts talking to itself as the assistant)
        text = re.sub(r"ASSISTANT:.*", "", text, flags=re.IGNORECASE | re.DOTALL)
        
        text = text.strip()
        
        # Strip surrounding quotes safely
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
            
        return text.strip()

    def generate(self, image, llm_model, vision_model, user_input, max_tokens, creativity, seed, debug_console, keep_model_loaded, n_ctx, frame_count):
        
        if debug_console:
            print(f"\n" + "="*50)
            print(f"🚀 [RT-LTX-2] ROYAL PROMPT DIAGNOSTICS START")
            print(f"==================================================")
            print(f"📝 USER INPUT:\n{user_input}\n")
        
        self.load_model(llm_model, vision_model, n_ctx)
        base64_image = self._tensor_to_base64(image)
        token_val = int(max_tokens.split(" - ")[0]) if " - " in max_tokens else int(max_tokens)
        temp = float(creativity.split(" - ")[0]) if " - " in creativity else 0.9
        
        final_prompt = (
            f"USER REQUEST:\n"
            f"'{user_input}'\n\n"
            f"Use the image as a visual reference. Write a highly specific, action-driven cinematic video prompt based on the LTX-2.3 instructions. "
            f"Format any dialogue cleanly, and end ONLY with an [AMBIENT: ...] tag."
        )
        
        user_content_block = [
            {"type": "text", "text": final_prompt},
            {"type": "image_url", "image_url": {"url": base64_image}}
        ]

        # Stop Tokens to prevent hallucination loops immediately
        stop_tokens = ["<end_of_turn>", "<eos>", "<|eot_id|>", "User:", "ASSISTANT:", "Assistant:", "REAL TASK:"]

        print(f"[RT-LTX-2] Generating Prompt...")

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
                seed=seed if seed != -1 else None
            )
            
            raw_result = response['choices'][0]['message']['content'].strip()
            
            if debug_console:
                print(f"🤖 RAW LLM OUTPUT (Before Clean):\n{raw_result}\n")
                
            final_result = self._clean_output(raw_result)
            
            # Robust Fallback: If LLM output is completely empty, use the user input safely
            if not final_result or final_result == "":
                print("\n[RT-LTX-2] ⚠️ WARNING: LLM returned an empty string. Falling back to User Input.")
                final_result = f"{user_input}\n\n[AMBIENT: subtle environmental sounds]"

        except Exception as e:
            print(f"\n[RT-LTX-2] ❌ CRITICAL GENERATION ERROR: {e}")
            final_result = f"Error during generation: {e}"

        if not keep_model_loaded:
            print(f"[RT-LTX-2] Unloading models to free VRAM...")
            self.unload_model()

        if debug_console:
            print(f"✨ FINAL CLEANED PROMPT (Sent to LTX-2):\n{final_result}")
            print(f"==================================================")
            print(f"🏁 [RT-LTX-2] DIAGNOSTICS END\n" + "="*50 + "\n")

        return (final_result, final_result, frame_count)

NODE_CLASS_MAPPINGS = {
    "RT_LTX2_RoyalPrompt": RT_LTX2_RoyalPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RT_LTX2_RoyalPrompt": "RT-LTX-2 Royal Prompt by RareTutor"
}
import torch
import sys

class RTLTX2SelfRefiningPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "refinement_strength": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 5.0, "step": 0.1}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "end_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_ltx2"
    CATEGORY = "RT_Nodes"

    def patch_ltx2(self, model, refinement_strength, start_percent, end_percent):
        m = model.clone()
        
        print(f"\n[RT-LTX2] Self-Correction Patch Loaded. Target: {start_percent*100}% to {end_percent*100}%")

        def get_status(current_sigma, sigmas):
            if len(sigmas) < 2: return 0.0, False
            try:
                # Calculate progress based on sigma (High Sigma = Start, Low Sigma = End)
                # We find the closest index in the sigmas list
                idx = (torch.abs(sigmas - current_sigma)).argmin().item()
                pct = idx / (len(sigmas) - 1)
                is_active = start_percent <= pct <= end_percent
                return pct, is_active
            except:
                return 0.0, False

        def refinement_proxy(args):
            input_x = args["input"]
            cond = args.get("cond")
            uncond = args.get("uncond")
            cond_scale = args.get("cond_scale", 1.0)
            model_options = args.get("model_options", {})

            # 1. SAFETY FALLBACK (Must return NOISE)
            if cond is None or uncond is None:
                return args.get("uncond") if args.get("uncond") is not None else torch.zeros_like(input_x)

            # 2. STANDARD CFG CALCULATION (Noise Prediction)
            diff = cond - uncond
            standard_cfg = uncond + (cond_scale * diff)

            # 3. CHECK REFINEMENT STATUS
            should_refine = False
            current_pct = 0.0
            
            if "sigmas" in model_options:
                ts = args.get("timestep")
                # Handle tensor vs float timestep
                current_sigma = ts[0] if isinstance(ts, torch.Tensor) and ts.numel() > 0 else ts
                current_pct, should_refine = get_status(current_sigma, model_options["sigmas"])

            pct_display = int(current_pct * 100)

            # 4. APPLY REFINEMENT (Paper 2 Logic Approximation)
            if should_refine:
                # The paper suggests pushing the latent towards the manifold.
                # In CFG terms, this is effectively boosting the guidance signal slightly
                # to "sharpen" the physics before the ODE step.
                
                safe_strength = min(max(refinement_strength, 0.0), 3.0)
                
                # We refine the NOISE prediction, not the latent
                # A small boost (safe_strength * 0.05) mimics the "Iterative Refinement" delta
                refined_result = standard_cfg + (diff * (safe_strength * 0.05))
                
                sys.stdout.write(f"\r[RT-LTX2] {pct_display}% | ACTIVE | Strength: {safe_strength:.1f}   ")
                sys.stdout.flush()
                
                return refined_result
            else:
                if pct_display % 5 == 0: 
                    sys.stdout.write(f"\r[RT-LTX2] {pct_display}% | PASSTHROUGH                  ")
                    sys.stdout.flush()
                
                return standard_cfg

        m.set_model_sampler_cfg_function(refinement_proxy)
        return (m,)

NODE_CLASS_MAPPINGS = {
    "RT_LTX2_SelfRefiningPatch": RTLTX2SelfRefiningPatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RT_LTX2_SelfRefiningPatch": "RT LTX-2 Self-Refining Video Patch"
}
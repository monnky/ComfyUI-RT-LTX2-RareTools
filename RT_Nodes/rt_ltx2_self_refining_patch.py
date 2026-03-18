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
                idx = (torch.abs(sigmas - current_sigma)).argmin().item()
                pct = idx / (len(sigmas) - 1)
                is_active = start_percent <= pct <= end_percent
                return pct, is_active
            except:
                return 0.0, False

        def refinement_proxy(args):
            x_t = args["input"]
            cond = args.get("cond")
            uncond = args.get("uncond")
            cond_scale = args.get("cond_scale", 1.0)
            model_options = args.get("model_options", {})
            
            ts = args.get("timestep")
            sigma = ts[0] if isinstance(ts, torch.Tensor) and ts.numel() > 0 else ts

            # 1. SAFETY FALLBACK
            if cond is None or uncond is None:
                return uncond if uncond is not None else torch.zeros_like(x_t)
            
            # 2. STANDARD CFG (Velocity Prediction)
            diff = cond - uncond
            v_pred = uncond + (cond_scale * diff)

            # 3. CHECK REFINEMENT STATUS
            current_pct = 0.0
            should_refine = False
            if "sigmas" in model_options:
                current_pct, should_refine = get_status(sigma, model_options["sigmas"])

            pct_display = int(current_pct * 100)

            # 4. APPLY ROBUST MANIFOLD REFINEMENT
            if should_refine and refinement_strength > 0:
                # Map the user's 0-5.0 strength to the 0.0-1.0 scale needed for manifold blending
                safe_scale = min(refinement_strength * 0.1, 1.0)
                clamp_threshold = 2.5 # Hardcoded to keep inputs identical to original
                
                # Approximate the clean latent (x_0)
                pred_x0 = x_t - (v_pred * sigma)
                
                # Calculate mean and variance
                mean_x0 = torch.mean(pred_x0, dim=[-2, -1], keepdim=True)
                std_x0 = torch.std(pred_x0, dim=[-2, -1], keepdim=True) + 1e-5
                
                # Identify extreme outliers (hallucinated physics)
                z_scores = torch.abs((pred_x0 - mean_x0) / std_x0)
                outlier_mask = (z_scores > clamp_threshold).float()
                
                # Clamp outliers back towards the valid manifold boundary
                clamped_x0 = torch.where(
                    pred_x0 > mean_x0, 
                    mean_x0 + (std_x0 * clamp_threshold), 
                    mean_x0 - (std_x0 * clamp_threshold)
                )
                
                # Blend the original prediction with the clamped prediction
                refined_x0 = (pred_x0 * (1.0 - outlier_mask * safe_scale)) + (clamped_x0 * outlier_mask * safe_scale)
                
                # Re-derive the velocity
                refined_v_pred = (x_t - refined_x0) / sigma
                
                sys.stdout.write(f"\r[RT-LTX2] {pct_display}% | REFINING MANIFOLD | Strength: {refinement_strength:.1f}   ")
                sys.stdout.flush()
                
                return refined_v_pred
            
            else:
                if pct_display % 5 == 0:
                    sys.stdout.write(f"\r[RT-LTX2] {pct_display}% | PASSTHROUGH                  ")
                    sys.stdout.flush()
                return v_pred

        m.set_model_sampler_cfg_function(refinement_proxy)
        return (m,)

NODE_CLASS_MAPPINGS = {
    "RT_LTX2_SelfRefiningPatch": RTLTX2SelfRefiningPatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RT_LTX2_SelfRefiningPatch": "RT LTX-2 Self-Refining Video Patch"
}
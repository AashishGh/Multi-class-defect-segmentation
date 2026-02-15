# # pip install thop torch pandas

# import os
# import torch
# import pandas as pd
# from thop import profile

# from models.models_mapping import models_dict


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# IN_CH = 3
# H, W = 512, 256   # <-- set your paper resolution here

# OUT_CSV = "results/model_complexity.csv"


# def count_params(model: torch.nn.Module) -> int:
#     return sum(p.numel() for p in model.parameters())


# def unwrap_output(out):
#     """Ensure THOP sees a Tensor."""
#     if isinstance(out, torch.Tensor):
#         return out
#     if isinstance(out, (list, tuple)):
#         return unwrap_output(out[0])
#     if isinstance(out, dict):
#         return unwrap_output(next(iter(out.values())))
#     raise TypeError(f"Unsupported output type: {type(out)}")


# class ForwardOnly(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, x):
#         return unwrap_output(self.model(x))


# @torch.no_grad()
# def compute_stats(model: torch.nn.Module, input_shape, device="cpu"):
#     model = model.to(device).eval()
#     dummy = torch.randn(1, *input_shape, device=device)

#     wrapped = ForwardOnly(model)

#     macs, _ = profile(wrapped, inputs=(dummy,), verbose=False)

#     total_params = count_params(model)

#     params_m = total_params / 1e6
#     gmacs = macs / 1e9
#     gflops_ref = macs / 1e9
#     gflops_2x = (2 * macs) / 1e9

#     return params_m, gmacs, gflops_ref, gflops_2x


# def main():
#     os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)

#     input_shape = (IN_CH, H, W)
#     rows = []

#     for name, model in models_dict.items():
#         try:
#             print(f"\nComputing: {name}")

#             params_m, gmacs, gflops_ref, gflops_2x = compute_stats(
#                 model, input_shape, device=DEVICE
#             )

#             rows.append({
#                 "model": name,
#                 "input": f"{IN_CH}x{H}x{W}",
#                 "params_M": round(params_m, 2),
#                 "GMACs": round(gmacs, 2),
#                 "GFLOPs_ref(MACs/1e9)": round(gflops_ref, 2),
#                 "GFLOPs_2x(2*MACs/1e9)": round(gflops_2x, 2),
#             })

#             print(
#                 f"{name:15s} | Params(M): {params_m:7.2f} | "
#                 f"GMACs: {gmacs:7.2f} | "
#                 f"GFLOPs(ref): {gflops_ref:7.2f} | "
#                 f"GFLOPs(2x): {gflops_2x:7.2f}"
#             )

#         except Exception as e:
#             print(f"[ERROR] {name} failed: {e}")

#     df = pd.DataFrame(rows)
#     df.to_csv(OUT_CSV, index=False)

#     print(f"\n✅ Saved CSV → {OUT_CSV}")


# if __name__ == "__main__":
#     main()


# # # pip install thop torch torchvision
# # import torch
# # from thop import profile
# # import importlib

# # MODEL_IMPORT_PATH = "models.vmunetv2"
# # MODEL_CLASS_NAME = "VMUNetV2"
# # INPUT_CHANNELS = 3
# # INPUT_RESOLUTION = (512,256)
# # def load_model(model_path, class_name):
# #     module = importlib.import_module(model_path)
# #     model_class = getattr(module, class_name)
# #     model = model_class()
# #     return model

# # @torch.no_grad()
# # def compute_model_stats(model, input_shape=(1, 384, 384)):
# #     dummy_input = torch.randn(1, *input_shape)
# #     macs, params = profile(model, inputs=(dummy_input,), verbose=False)
# #     gflops = macs / 1e9
# #     params_m = params / 1e6
# #     print(f"\nModel Summary:")
# #     print(f"GFLOPs : {gflops:.2f}")
# #     print(f"Params : {params_m:.2f} M")

# # if __name__ == "__main__":
# #     model = load_model(MODEL_IMPORT_PATH, MODEL_CLASS_NAME).eval()
# #     compute_model_stats(model, (INPUT_CHANNELS, INPUT_RESOLUTION[0], INPUT_RESOLUTION[1]))


# fps_benchmark_all_models.py
# Benchmark FPS (forward-only) for every model in models_dict with clear logs.
#
# pip install torch pandas
# Usage:
#   python fps_benchmark_all_models.py --h 512 --w 256 --device cuda
#   python fps_benchmark_all_models.py --models unet vmunetv2 --iters 300 --warmup 50

import os
import time
import argparse
import torch
import pandas as pd

from models.models_mapping import models_dict  # expects factories OR instantiated models


def unwrap_output(out):
    """Ensure forward returns a Tensor (supports Tensor / tuple/list / dict)."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)) and len(out) > 0:
        return unwrap_output(out[0])
    if isinstance(out, dict) and len(out) > 0:
        # Prefer common keys if present
        for k in ["logits", "out", "pred", "mask"]:
            if k in out:
                return unwrap_output(out[k])
        return unwrap_output(next(iter(out.values())))
    raise TypeError(f"Unsupported model output type: {type(out)}")


class ForwardOnly(torch.nn.Module):
    """Wrap model so benchmark always measures a Tensor output."""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return unwrap_output(self.model(x))


def build_model(entry):
    """
    Supports BOTH:
      - instances: models_dict[name] is nn.Module
      - factories: models_dict[name] is callable returning nn.Module
    IMPORTANT: nn.Module is also callable, so we must check isinstance first.
    """
    if isinstance(entry, torch.nn.Module):
        return entry  # already instantiated model

    if callable(entry):
        m = entry()   # factory/lambda/class constructor
        if not isinstance(m, torch.nn.Module):
            raise TypeError("Factory did not return a torch.nn.Module")
        return m

    raise TypeError(f"Unsupported models_dict entry type: {type(entry)}")



@torch.no_grad()
def benchmark_fps(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    device: str,
    warmup_iters: int,
    measure_iters: int,
    log_every: int,
) -> float:
    """
    Forward-only FPS benchmark with detailed progress logs.
    Uses cuda.synchronize for correct GPU timing.
    """
    model.eval()

    # ---------------- Warmup ----------------
    print(f"    [Warmup] {warmup_iters} iters ...")
    for i in range(1, warmup_iters + 1):
        _ = model(dummy)
        if log_every > 0 and (i % log_every == 0 or i == warmup_iters):
            print(f"      warmup iter {i}/{warmup_iters}")
    if device == "cuda":
        torch.cuda.synchronize()

    # ---------------- Timed ----------------
    print(f"    [Measure] {measure_iters} iters ...")
    t0 = time.perf_counter()
    for i in range(1, measure_iters + 1):
        _ = model(dummy)
        if log_every > 0 and (i % log_every == 0 or i == measure_iters):
            print(f"      measure iter {i}/{measure_iters}")
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total = t1 - t0
    avg = total / measure_iters
    fps = 1.0 / avg
    return fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of model keys in models_dict. Default: all.")
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--h", type=int, default=512)
    parser.add_argument("--w", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)

    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"],
                        help="Use fp16 for inference timing if supported (GPU only).")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=25,
                        help="Print progress every N iterations (0 disables).")

    parser.add_argument("--out_csv", type=str, default="results/model_fps.csv")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if device == "cuda":
        # Helps choose best cuDNN kernels for fixed shapes (good for FPS benchmarking)
        torch.backends.cudnn.benchmark = True

    # Choose dtype
    dtype = torch.float16 if (args.dtype == "fp16" and device == "cuda") else torch.float32
    if args.dtype == "fp16" and device != "cuda":
        print("[WARN] fp16 requested but CUDA not available; using fp32.")

    # Select model keys
    keys = args.models if args.models is not None else list(models_dict.keys())

    # Dummy input
    dummy = torch.randn(args.batch, args.in_ch, args.h, args.w, device=device, dtype=dtype)
    print(f"Input: {args.batch}x{args.in_ch}x{args.h}x{args.w} | dtype: {dtype}")

    rows = []
    for name in keys:
        if name not in models_dict:
            print(f"\n[SKIP] Unknown model key: {name}")
            continue

        print("\n" + "=" * 90)
        print(f"[MODEL] {name}")

        try:
            # Build + move model
            base_model = build_model(models_dict[name])
            base_model = base_model.to(device).eval()

            # Match dtype (only if fp16 selected)
            if dtype == torch.float16:
                base_model = base_model.half()

            model = ForwardOnly(base_model).to(device).eval()

            # Quick sanity forward (logged)
            print("    [Sanity] Running 1 forward pass to confirm output shape...")
            out = model(dummy)
            print(f"    [Sanity] Output tensor shape: {tuple(out.shape)}")

            # FPS benchmark
            fps = benchmark_fps(
                model=model,
                dummy=dummy,
                device=device,
                warmup_iters=args.warmup,
                measure_iters=args.iters,
                log_every=args.log_every,
            )

            # Report + save
            print(f"    ✅ FPS = {fps:.2f}")

            rows.append({
                "model": name,
                "input": f"{args.in_ch}x{args.h}x{args.w}",
                "batch": args.batch,
                "device": device,
                "dtype": args.dtype,
                "warmup_iters": args.warmup,
                "measure_iters": args.iters,
                "FPS": round(fps, 2),
            })

        except Exception as e:
            print(f"    ❌ ERROR benchmarking '{name}': {e}")

        finally:
            # Free memory between models
            if device == "cuda":
                torch.cuda.empty_cache()

    # Save CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\n✅ Saved FPS CSV: {args.out_csv}\n")


if __name__ == "__main__":
    main()

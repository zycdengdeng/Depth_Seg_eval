"""
新服务器环境检查：依赖版本、GPU、各指标模型能否加载并跑通一个样本

有网络时会顺带下载权重；无网络时按提示从已有服务器拷贝缓存（见 README「模型权重」）。
"""

import importlib
import os
import traceback
from typing import Iterable

import numpy as np

PACKAGES = ["torch", "torchvision", "transformers", "huggingface_hub", "numpy", "scipy",
            "PIL", "skimage", "lpips", "cleanfid", "cv2", "tqdm", "yaml"]

CLEANFID_INCEPTION = "/tmp/inception-2015-12-05.pt"


def _ok(flag: bool) -> str:
    return "OK  " if flag else "FAIL"


def check_environment(device: str = "cuda", groups: Iterable[str] = ()) -> bool:
    all_ok = True

    print("== 依赖 ==")
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  OK   {name:<16} {getattr(mod, '__version__', '')}")
        except Exception as e:
            all_ok = False
            print(f"  FAIL {name:<16} {e}")

    print("\n== GPU ==")
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  {_ok(cuda)} CUDA 可用: {cuda}"
              + (f"，{torch.cuda.device_count()} 张卡，{torch.cuda.get_device_name(0)}" if cuda else ""))
        if device.startswith("cuda") and not cuda:
            all_ok = False
    except Exception as e:
        all_ok = False
        print(f"  FAIL {e}")

    print("\n== 缓存位置 ==")
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    torch_home = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
    print(f"  HuggingFace: {os.path.join(hf_home, 'hub')}"
          f"{'（HF_HUB_OFFLINE=1）' if os.environ.get('HF_HUB_OFFLINE') == '1' else ''}")
    print(f"  torch hub:   {os.path.join(torch_home, 'hub', 'checkpoints')}")
    print(f"  clean-fid:   {CLEANFID_INCEPTION}"
          f"（{'已存在' if os.path.exists(CLEANFID_INCEPTION) else '不存在，首次计算 KID/FID 时下载'}）")

    print("\n== 模型加载 + 单样本试跑 ==")
    from .scorers import Scorer
    rng = np.random.default_rng(0)
    gt = (rng.uniform(0, 255, (360, 640, 3))).astype(np.uint8)
    gen = np.clip(gt.astype(int) + rng.integers(-20, 20, gt.shape), 0, 255).astype(np.uint8)

    for group in [g for g in ("pixel", "depth", "seg") if g in groups]:
        try:
            scorer = Scorer([group], device=device)
            values, _ = scorer.score(gen, gt, scorer.reference(gt))
            print(f"  OK   {group:<6} " + ", ".join(f"{k}={v:.3f}" for k, v in values.items()))
        except Exception:
            all_ok = False
            print(f"  FAIL {group}")
            traceback.print_exc()

    if "dist" in groups:
        try:
            import torch
            from cleanfid.features import build_feature_extractor
            try:
                build_feature_extractor("clean", torch.device(device), use_dataparallel=False)
            except TypeError:
                build_feature_extractor("clean", torch.device(device))
            print("  OK   dist   clean-fid Inception 已加载")
        except Exception:
            all_ok = False
            print("  FAIL dist")
            traceback.print_exc()

    print("\n" + ("全部通过" if all_ok else "存在失败项，见上方 FAIL；无网络时参考 evalkit/README.md「模型权重」拷贝缓存"))
    return all_ok

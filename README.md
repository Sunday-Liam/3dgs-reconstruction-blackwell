# 3D Gaussian Splatting Reconstruction on a Blackwell (RTX 50-series) GPU

A from-scratch, end-to-end practice project: capturing a multi-view image set of a real object with a phone, recovering camera poses with COLMAP, and training a **3D Gaussian Splatting (3DGS)** model with **Nerfstudio** for free-viewpoint rendering — all on a brand-new **NVIDIA Blackwell (sm_120)** GPU, which required adapting the full CUDA toolchain and fixing custom CUDA kernel compilation from scratch.

> This repository documents not just the result, but the full process — including the engineering problems I had to solve to make a cutting-edge GPU architecture work with the existing 3DGS ecosystem.

---

## Result

| Front view | Top-down view |
|---|---|
| ![front](assets/front_view.png) | ![top](assets/top_view.png) |

*Reconstruction of a small potted plant from 47 phone-captured images. COLMAP matched all 47 images; the 3DGS model was trained to 30,000 iterations.*

---

## Pipeline

```
Phone capture (47 images)
        │
        ▼
COLMAP  ──►  Structure-from-Motion (camera pose estimation)
        │     47 / 47 images matched
        ▼
Nerfstudio (splatfacto)  ──►  3D Gaussian Splatting training
        │
        ▼
Free-viewpoint rendering in the web viewer
```

1. **Capture** — ~47 phone photos around a textured object, with high overlap between adjacent frames, covering multiple heights.
2. **Pose estimation** — `ns-process-data` runs COLMAP to extract features, match them across images, and recover per-image camera poses (extrinsics + intrinsics).
3. **Training** — `ns-train splatfacto` optimizes a set of 3D Gaussians to reproduce the captured views.
4. **Rendering** — the trained model is explored interactively in the Nerfstudio viewer.

---

## Environment

| Component | Version / Detail |
|---|---|
| GPU | NVIDIA RTX 50-series Laptop (Blackwell, **sm_120**) |
| OS | Windows 11 + WSL2 (Ubuntu 24.04) |
| PyTorch | 2.x **+cu128** |
| CUDA Toolkit | 12.8 |
| Framework | Nerfstudio (splatfacto) + gsplat |
| SfM | COLMAP |

---

## The hard part: making 3DGS work on Blackwell (sm_120)

The RTX 50-series uses NVIDIA's new Blackwell architecture (compute capability **sm_120**), which most of the existing 3D vision ecosystem did not yet support out of the box. The main issues I ran into and solved:

**1. PyTorch did not support sm_120 by default.**
Standard PyTorch builds only supported up to sm_90, causing `sm_120 is not compatible` errors. Fixed by installing the **cu128** build of PyTorch, which ships Blackwell kernels.

**2. `gsplat` custom CUDA kernels failed to compile.**
gsplat JIT-compiles its CUDA kernels on first run using the system `nvcc`. The system compiler was too old and threw:

```
nvcc fatal : Unsupported gpu architecture 'compute_120'
```

Fixed by installing **CUDA Toolkit 12.8** (which supports compute_120), pointing `CUDA_HOME` / `PATH` to it, clearing the stale `~/.cache/torch_extensions` build cache, and recompiling.

**3. Out-of-memory during compilation on a laptop GPU.**
The default parallel build (`MAX_JOBS=10`) exhausted RAM. Fixed by lowering it to `MAX_JOBS=2`, trading speed for a stable, lower memory footprint.

See [`notes/blackwell_setup.md`](notes/blackwell_setup.md) for the detailed step-by-step.

---

## What I learned

- **Camera pose is the bridge between 2D and 3D.** COLMAP recovers it from images; downstream tasks (rendering, and in generative settings, controllable generation) consume it. This made me curious about how the same geometric quantity sits on both the perception and the generation side of the problem.
- **Capture quality dominates reconstruction quality.** High overlap and even coverage are what made all 47 images match.
- **Bringing-up new hardware is its own skill.** A large fraction of the work was not "running 3DGS" but making the toolchain compile on an architecture the ecosystem hadn't caught up to yet.

---

## Acknowledgements

Built on [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio), [gsplat](https://github.com/nerfstudio-project/gsplat), and [COLMAP](https://github.com/colmap/colmap). The 3DGS method follows Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH 2023).

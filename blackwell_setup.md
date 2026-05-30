# Setting up 3D Gaussian Splatting on a Blackwell (sm_120) GPU — Step by Step

This note records the exact steps and fixes needed to run Nerfstudio's `splatfacto` (3D Gaussian Splatting) on an NVIDIA RTX 50-series (Blackwell, compute capability **sm_120**) GPU under WSL2 (Ubuntu 24.04). At the time, most of the ecosystem did not support sm_120 out of the box.

---

## 1. Verify the GPU is visible in WSL

```bash
nvidia-smi
```

Should list the RTX 50-series GPU and a recent driver version. If not, the issue is the Windows-side NVIDIA driver / WSL passthrough.

---

## 2. Install PyTorch with Blackwell support (cu128)

Default PyTorch builds only support up to sm_90 and will error with
`NVIDIA ... with CUDA capability sm_120 is not compatible`.
Install the cu128 build instead:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```python
import torch
print(torch.__version__)                       # 2.x.x+cu128
print(torch.cuda.is_available())               # True
print(torch.cuda.get_device_capability(0))     # (12, 0)  -> sm_120
```

---

## 3. Install Nerfstudio + COLMAP

```bash
pip install nerfstudio
sudo apt update && sudo apt install -y colmap ffmpeg
```

(`ffmpeg` is checked for at startup even when the input is images, not video.)

---

## 4. The main blocker: gsplat fails to compile for compute_120

On first run, gsplat JIT-compiles its CUDA kernels with the **system** `nvcc`. If that compiler is older than CUDA 12.8, it throws:

```
nvcc fatal : Unsupported gpu architecture 'compute_120'
```

### Fix: install CUDA Toolkit 12.8 and point the build at it

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

Add the new toolkit to the environment (persisted in `~/.bashrc`):

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version    # must now report release 12.8
```

Clear the stale build cache so gsplat recompiles with the new nvcc:

```bash
rm -rf ~/.cache/torch_extensions
```

---

## 5. Avoid out-of-memory during compilation

The default parallel build can exhaust RAM on a laptop. Lower the job count:

```bash
MAX_JOBS=2 ns-train splatfacto --data <processed_data_dir>
```

The first run will spend a few minutes compiling the CUDA kernels, then start training.

---

## 6. Full run

```bash
# 1) Estimate camera poses from images
ns-process-data images --data <images_dir> --output-dir <processed_dir>

# 2) Train 3D Gaussian Splatting
MAX_JOBS=2 ns-train splatfacto --data <processed_dir>

# 3) Open the printed viewer URL (replace 0.0.0.0 with localhost) in a browser
#    http://localhost:7007
```

---

## Summary of fixes

| Problem | Root cause | Fix |
|---|---|---|
| `sm_120 is not compatible` | PyTorch built only up to sm_90 | Install PyTorch cu128 |
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | System nvcc < 12.8 | Install CUDA Toolkit 12.8, set CUDA_HOME, clear extension cache |
| Compilation hangs / killed | RAM exhausted by parallel build | `MAX_JOBS=2` |

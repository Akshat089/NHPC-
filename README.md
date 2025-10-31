# NHPC-

## 🧠 Project Overview
**NHPC- (Next-Generation Hybrid Parallel Communication)** is a high-performance computing runtime that integrates **MPI** and **NCCL** to enable efficient, hybrid communication for distributed deep-learning workloads.  
This project is designed to experiment with combining traditional CPU-based communication (MPI) and GPU-based collectives (NCCL) for improved overlap and performance.

---

## ⚙️ System Requirements

| Component | Minimum Version | Notes |
|------------|----------------|-------|
| **Windows** | Windows 11 (22H2+) or Windows 10 (21H2+) | Must support WSL2 |
| **WSL** | Version 2 | Use Ubuntu 22.04 LTS |
| **NVIDIA GPU** | RTX 20-series or newer | Must support CUDA 12+ |
| **NVIDIA Driver** | ≥ 560.xx | WSL-compatible driver |
| **Internet Access** | – | Required for downloads |

---

## 🧩 1️⃣ Install and Verify WSL2

1. Enable WSL:
   ```powershell
   wsl --install
Install Ubuntu 22.04 LTS from the Microsoft Store.

Check WSL version:

wsl --status


✅ Should show Default Version: 2.

(If not) Convert Ubuntu to WSL2:

wsl --set-version Ubuntu-22.04 2

⚡ 2️⃣ Install NVIDIA Driver for CUDA on WSL

Go to: https://developer.nvidia.com/cuda/wsl

Download the latest R560+ driver.

Run the installer → choose Custom → Clean Installation.

Reboot your PC.

Test in PowerShell:

nvidia-smi


✅ You should see your GPU (e.g., RTX 3060) and CUDA Version ≥ 12.6.

🧰 3️⃣ Inside WSL (Ubuntu Terminal)

Run these commands:

sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl python3-pip python3-venv

🧩 4️⃣ Install CUDA Toolkit 12.6
sudo apt install -y cuda-toolkit-12-6
nvcc --version


✅ Expected:

Cuda compilation tools, release 12.6, V12.6.xx

🔗 5️⃣ Install NCCL + MPI (CUDA-Aware Communication)
sudo apt install -y libnccl2 libnccl-dev libopenmpi-dev openmpi-bin
mpirun --version


✅ Expected:

Open MPI 4.x.x

🧱 6️⃣ Create Python Environment and Install PyTorch
python3 -m venv ~/pytorch-env
source ~/pytorch-env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126


✅ Verify CUDA access:

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"


Expected output:

True NVIDIA GeForce RTX 3060

🧮 7️⃣ Verify Full GPU and Communication Stack
nvidia-smi
nvcc --version
mpirun --version
python -c "import torch; print(torch.cuda.is_available())"


✅ All commands should work without errors.

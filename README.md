# NHPC-

## 🧠 Project Overview
**NHPC- (Next-Generation Hybrid Parallel Communication)** is a high-performance computing runtime that integrates **MPI** and **NCCL** to enable efficient, hybrid communication for distributed deep-learning workloads.  
This project is designed to experiment with combining traditional CPU-based communication (MPI) and GPU-based collectives (NCCL) for improved overlap and performance.

---

## ⚙️ System Requirements

| Component       | Minimum Version             | Notes                       |
|-----------------|----------------------------|-----------------------------|
| **Windows**     | Windows 11 (22H2+) or Windows 10 (21H2+) | Must support WSL2 |
| **WSL**         | Version 2                  | Use Ubuntu 22.04 LTS        |
| **NVIDIA GPU**  | RTX 20-series or newer     | Must support CUDA 12+       |
| **NVIDIA Driver** | ≥ 560.xx                  | WSL-compatible driver       |
| **Internet Access** | –                       | Required for downloads      |

---

## 🧩 1️⃣ Install and Verify WSL2

1. **Enable WSL:**
    ```powershell
    wsl --install
    ```
2. **Install Ubuntu 22.04 LTS from the Microsoft Store.**

3. **Check WSL version:**
    ```powershell
    wsl --status
    ```
    ✅ Should show **Default Version: 2**.  

4. **If not, convert Ubuntu to WSL2:**
    ```powershell
    wsl --set-version Ubuntu-22.04 2
    ```

---

## 🧩 2️⃣ Install NVIDIA Driver for CUDA on WSL

1. Go to: [CUDA on WSL](https://developer.nvidia.com/cuda/wsl)  
2. Download the latest **R560+ driver**.  
3. Run the installer → choose **Custom → Clean Installation**.  
4. Reboot your PC.  

5. Test in PowerShell:
    ```powershell
    nvidia-smi
    ```
    ✅ You should see your GPU (e.g., RTX 3060) and CUDA Version ≥ 12.6.

---

## 🧰 3️⃣ Inside WSL (Ubuntu Terminal)

1. Run:
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y build-essential git wget curl python3-pip python3-venv

## 4️⃣ Install CUDA Toolkit 12.6

1. Run:
   ```bash
   sudo apt install -y cuda-toolkit-12-6
   nvcc --version

## 5️⃣ Install NCCL + MPI (CUDA-Aware Communication)

1. sudo apt install -y libnccl2 libnccl-dev libopenmpi-dev openmpi-bin
2. mpirun --version

## 6️⃣ Create Python Environment and Install PyTorch

1. python3 -m venv ~/pytorch-env
2. source ~/pytorch-env/bin/activate
3. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

## Verify CUDA access:
1. python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

## 7️⃣ Verify Full GPU and Communication Stack
1. nvidia-smi
2. nvcc --version
3. mpirun --version
4. python -c "import torch; print(torch.cuda.is_available())"




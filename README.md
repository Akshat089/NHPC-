# ⚡ NHPC — Next-Generation Hybrid Parallel Communication

## 🧠 Project Overview

**NHPC** (Next-Generation Hybrid Parallel Communication) is a high-performance computing runtime that integrates **MPI** and **NCCL** to enable efficient hybrid communication for distributed deep-learning workloads.  
This project explores combining traditional **CPU-based communication (MPI)** with **GPU-based collectives (NCCL)** for improved **overlap** and **performance**.

---

## ⚙️ System Requirements

| Component | Minimum Version | Notes |
|------------|----------------|-------|
| **Windows** | Windows 11 (22H2+) or Windows 10 (21H2+) | Must support **WSL2** |
| **WSL** | Version 2 | Use **Ubuntu 22.04 LTS** |
| **NVIDIA GPU** | RTX 20-series or newer | Must support **CUDA 12+** |
| **NVIDIA Driver** | ≥ 560.xx | Must be **WSL-compatible** |
| **Internet Access** | – | Required for downloads |

---

## 🧩 1️⃣ Install and Verify WSL2

### Step 1 — Enable WSL
```powershell
wsl --install

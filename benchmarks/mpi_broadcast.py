import torch
from mcrdl import Comm
import time

# Initialize communication backend
comm = Comm()
comm.init()
rank = comm.get_rank()
world_size = comm.get_world_size()

# Test different message sizes
for exp in range(10, 25, 2):  # 1KB to 16MB approx
    size = 2**exp
    # Only rank 0 initializes data
    if rank == 0:
        tensor = torch.arange(size, dtype=torch.float32, device='cuda') if torch.cuda.is_available() else torch.arange(size, dtype=torch.float32)
    else:
        tensor = torch.zeros(size, dtype=torch.float32, device='cuda') if torch.cuda.is_available() else torch.zeros(size, dtype=torch.float32)

    # Broadcast and time it
    start = time.time()
    comm.broadcast(tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()

    # Check correctness: mean value should match rank 0's tensor
    expected = (size - 1) / 2  # mean of 0..size-1
    mean_val = tensor.mean().item()

    print(f"[Rank {rank}] Broadcast size={size} | mean={mean_val:.2f}, expected={expected:.2f}, time={end-start:.6f}s")

comm.finalize()
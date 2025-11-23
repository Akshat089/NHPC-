import torch
from mcrdl import Comm
import time

# Initialize Comm
comm = Comm()  # will pick NCCL on GPU
comm.init()
rank = comm.get_rank()
world_size = comm.get_world_size()

# Test for different message sizes
for exp in range(10, 25, 2):  # 1KB to 16MB approx
    size = 2**exp
    tensor = torch.ones(size, device='cuda') * (rank + 1)

    start = time.time()
    comm.all_reduce(tensor)
    torch.cuda.synchronize()
    end = time.time()

    # Check correctness
    expected = sum(r + 1 for r in range(world_size))
    mean_val = tensor.mean().item()

    print(f"[Rank {rank}] AllReduce size={size} elements | mean={mean_val:.2f}, expected={expected}, time={end-start:.6f}s")

comm.finalize()
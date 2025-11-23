from mcrdl import Comm
import torch
import time

comm = Comm()
comm.init()  # initializes MPI internally
rank = comm.get_rank()
world_size = comm.get_world_size()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Broadcast benchmark
for exp in range(10, 25, 2):
    size = 2**exp
    if rank == 0:
        tensor = torch.arange(1, size + 1, dtype=torch.float32, device=device)
    else:
        tensor = torch.zeros(size, dtype=torch.float32, device=device)

    start = time.time()
    comm.broadcast(tensor)
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    expected_mean = (size + 1) / 2
    mean_val = tensor.float().mean().item()

    print(f"[Rank {rank}] Broadcast size={size} | mean={mean_val:.2f}, expected={expected_mean:.2f}, time={end-start:.6f}s")

comm.finalize()
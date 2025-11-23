import torch
import torch.distributed as dist
import os
import time

# Get rank and world size from MPI environment (set by mpirun)
rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize NCCL backend
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Prepare tensor
tensor_size = 1 << 20  # 1M elements
tensor = torch.ones(tensor_size, device=device) * (rank + 1)

# Synchronize, measure AllReduce time
torch.cuda.synchronize() if device=='cuda' else None
start = time.time()
dist.all_reduce(tensor)
torch.cuda.synchronize() if device=='cuda' else None
end = time.time()

print(f"[Rank {rank}] Raw NCCL AllReduce time: {end-start:.6f}s")

# Cleanup
dist.destroy_process_group()
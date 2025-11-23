import torch
import torch.distributed as dist
import time
import os

# Initialize environment variables for single-node MPI
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])

dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------
# Parameters
# ---------------------------
matrix_size = 2048          # heavy computation
tensor_size = 1 << 20       # communication size

# ---------------------------
# Prepare tensors
# ---------------------------
comp_tensor_a = torch.randn((matrix_size, matrix_size), device=device)
comp_tensor_b = torch.randn((matrix_size, matrix_size), device=device)
comm_tensor = torch.ones(tensor_size, device=device) * (rank + 1)

# ---------------------------
# Measure computation
# ---------------------------
torch.cuda.synchronize() if device=='cuda' else None
start_comp = time.time()
result = torch.matmul(comp_tensor_a, comp_tensor_b)
torch.cuda.synchronize() if device=='cuda' else None
end_comp = time.time()
comp_time = end_comp - start_comp

# ---------------------------
# Measure communication (AllReduce)
# ---------------------------
torch.cuda.synchronize() if device=='cuda' else None
start_comm = time.time()
dist.all_reduce(comm_tensor)
torch.cuda.synchronize() if device=='cuda' else None
end_comm = time.time()
comm_time = end_comm - start_comm

# ---------------------------
# Report results
# ---------------------------
expected = sum(r + 1 for r in range(world_size))
mean_val = comm_tensor.mean().item()
ratio = comm_time / comp_time if comp_time > 0 else float('inf')

print(f"[Rank {rank}] Computation time: {comp_time:.6f}s")
print(f"[Rank {rank}] Communication time: {comm_time:.6f}s, mean={mean_val:.2f}, expected={expected}")
print(f"[Rank {rank}] Comm/Comp ratio: {ratio:.6f}")

dist.destroy_process_group()
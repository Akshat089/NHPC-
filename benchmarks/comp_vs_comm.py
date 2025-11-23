import torch
from mcrdl import Comm
import time

# Initialize communication
comm = Comm()
comm.init()
rank = comm.get_rank()
world_size = comm.get_world_size()

# Parameters
matrix_size = 2048  # adjust for heavier compute
tensor_size = 1 << 20  # adjust for communication

# Prepare tensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'
comp_tensor_a = torch.randn((matrix_size, matrix_size), device=device)
comp_tensor_b = torch.randn((matrix_size, matrix_size), device=device)
comm_tensor = torch.ones(tensor_size, device=device) * (rank + 1)

# ---------------------------
# Measure computation time
# ---------------------------
torch.cuda.synchronize() if device=='cuda' else None
start_comp = time.time()
result = torch.matmul(comp_tensor_a, comp_tensor_b)
torch.cuda.synchronize() if device=='cuda' else None
end_comp = time.time()
comp_time = end_comp - start_comp

# ---------------------------
# Measure communication time
# ---------------------------
torch.cuda.synchronize() if device=='cuda' else None
start_comm = time.time()
comm.all_reduce(comm_tensor)
torch.cuda.synchronize() if device=='cuda' else None
end_comm = time.time()
comm_time = end_comm - start_comm

# ---------------------------
# Verification and reporting
# ---------------------------
expected = sum(r + 1 for r in range(world_size))
mean_val = comm_tensor.mean().item()
print(f"[Rank {rank}] Computation time: {comp_time:.6f}s")
print(f"[Rank {rank}] Communication time: {comm_time:.6f}s, mean={mean_val:.2f}, expected={expected}")

comm.finalize()

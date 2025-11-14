import torch
import mcrdl
import math

def test_scatter():
    comm = mcrdl.Comm("mpi")
    comm.init()
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    # Root rank creates the full tensor
    if rank == 0:
        full_tensor = torch.arange(1, 16, dtype=torch.float32)
        print(f"[Rank {rank}] Full tensor before scatter:\n{full_tensor}")
    else:
        full_tensor = None

    # Broadcast the total size
    if rank == 0:
        total_size = full_tensor.numel()
    else:
        total_size = 0

    total_size_tensor = torch.tensor([total_size], dtype=torch.int32)
    comm.broadcast(total_size_tensor)
    total_size = int(total_size_tensor.item())

    # Compute counts dynamically
    base_count = total_size // world_size
    remainder = total_size % world_size
    counts = [base_count + 1 if i < remainder else base_count for i in range(world_size)]
    recv_count = counts[rank]

    # Allocate buffer for receive — this will hold the slice
    recv_buf = torch.empty(recv_count, dtype=torch.float32)

    # **Pass the correct buffer to C++**
    if rank == 0:
        # On root, Scatter will use `full_tensor` as the source
        comm.scatter(full_tensor)
        # Copy root's received slice to recv_buf
        start = sum(counts[:rank])
        end = start + recv_count
        recv_buf.copy_(full_tensor[start:end])
    else:
        comm.scatter(recv_buf)

    print(f"[Rank {rank}] Received scatter tensor:\n{recv_buf}")
    comm.finalize()

def test_broadcast():
    comm = mcrdl.Comm("mpi")
    comm.init()
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    # Only rank 0 initializes the data
    if rank == 0:
        data = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
        print(f"[Rank {rank}] Broadcasting tensor:\n{data}")
    else:
        # Non-root ranks allocate empty buffer of same shape
        data = torch.empty(4, dtype=torch.float32)

    # Call the C++ broadcast
    comm.broadcast(data)

    print(f"[Rank {rank}] Tensor after broadcast:\n{data}")

    comm.finalize()


def test_cpu_broadcast():
    print("\n--- CPU Broadcast Test ---")
    comm = mcrdl.Comm("mpi")
    comm.init()
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    # Only rank 0 initializes the data
    if rank == 0:
        data = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
        print(f"[Rank {rank}] Broadcasting tensor:\n{data}")
    else:
        data = torch.zeros(4, dtype=torch.float32)

    comm.broadcast(data)
    print(f"[Rank {rank}] Tensor after broadcast:\n{data}")
    comm.finalize()

def test_cpu_allreduce():
    print("\n--- CPU AllReduce Test ---")
    comm = mcrdl.Comm("mpi")
    comm.init()
    x = torch.ones((4, 4), dtype=torch.float32) * 2
    print(f"[Before rank tensor] {x}")
    comm.all_reduce(x)
    print(f"[After all_reduce rank tensor] {x}")   # ← NEW
    
    print("\n--- CPU AlltoAll Test ---")
    y = torch.ones((4, 4), dtype=torch.float32) * 2
    print(f"[Before rank tensor] {y}")
    comm.all_to_all(y)
    print(f"[After all_to_all rank tensor] {y}")
    comm.finalize()

def test_cpu_alltoall():
    print("\n--- CPU AlltoAll Test ---")
    comm = mcrdl.Comm("mpi")
    comm.init()
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    # Create a unique matrix per rank
    # Each row = rank * 10 + column index
    x = torch.zeros((world_size, world_size), dtype=torch.float32)
    for i in range(world_size):
        for j in range(world_size):
            x[i, j] = rank * 10 + i * world_size + j

    print(f"[Rank {rank}] Before all_to_all tensor:\n{x}")
    comm.all_to_all(x)
    print(f"[Rank {rank}] After all_to_all tensor:\n{x}")
    comm.finalize()


def test_cpu_gather():
    print("\n--- CPU Gather (AllGather) Test ---")
    comm = mcrdl.Comm("mpi")
    comm.init()
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    # Each rank creates a unique tensor to contribute
    rows, cols = 2, 2
    x = torch.arange(rank * rows * cols, (rank + 1) * rows * cols, dtype=torch.float32).reshape(rows, cols)
    print(f"[Rank {rank}] Before gather tensor:\n{x}")

    # Perform the gather operation (all ranks get full tensor)
    comm.gather(x)

    # Print the gathered result
    print(f"[Rank {rank}] After gather tensor:\n{x}")

    comm.finalize()




if __name__ == "__main__":
    # test_cpu_allreduce()
    test_cpu_gather()
    # test_cuda_allreduce()
    # test_scatter()
    # test_broadcast()
    # test_cpu_alltoall()

import torch
import mcrdl

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
    x = torch.ones((4, 4), dtype=torch.float32) * 2
    print(f"[Before rank tensor] {x}")
    comm.all_to_all(x)
    print(f"[After all_reduce rank tensor] {x}")   # ← NEW
    comm.finalize()

def test_cpu_gather():
    print("\n--- CPU Gather Test ---")
    comm = mcrdl.Comm("mpi")
    comm.init()

    # Try to get rank and world size for printing
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        world_size = MPI.COMM_WORLD.Get_size()
    except Exception:
        print("[Warning] mpi4py not installed — assuming rank=0, size=1")
        rank, world_size = 0, 1

    # Each rank creates a unique tensor to contribute
    x = torch.ones((2, 2), dtype=torch.float32) * (rank + 1)
    print(f"[Rank {rank}] Before gather tensor:\n{x}")

    # Perform the gather operation (root rank 0 collects all)
    comm.gather(x)

    # Note: after gather, only rank 0 (root) has the full data in your C++ implementation
    print(f"[Rank {rank}] After gather tensor:\n{x}")

    comm.finalize()


# def test_cuda_allreduce():
#     if not torch.cuda.is_available():
#         print("\n[Skipping CUDA test: No GPU available]")
#         return
#     print("\n--- CUDA AllReduce Test ---")
#     comm = mcrdl.Comm("nccl")
#     comm.init()
#     x = torch.ones((2, 2), dtype=torch.float32, device='cuda') * 5
#     print("Input tensor (CUDA):", x)
#     comm.all_reduce(x)
#     comm.finalize()

if __name__ == "__main__":
    # test_cpu_allreduce()
    test_cpu_gather()
    # test_cuda_allreduce()

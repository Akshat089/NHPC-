import torch
import mcrdl

def test_cpu_allreduce():
    print("\n--- CPU AllReduce Test ---")
    comm = mcrdl.Comm("mpi")
    comm.init()
    x = torch.ones((4, 4), dtype=torch.float32) * 2
    print("Input tensor (CPU):", x)
    comm.all_reduce(x)
    comm.finalize()

def test_cuda_allreduce():
    if not torch.cuda.is_available():
        print("\n[Skipping CUDA test: No GPU available]")
        return
    print("\n--- CUDA AllReduce Test ---")
    comm = mcrdl.Comm("nccl")
    comm.init()
    x = torch.ones((2, 2), dtype=torch.float32, device='cuda') * 5
    print("Input tensor (CUDA):", x)
    comm.all_reduce(x)
    comm.finalize()

if __name__ == "__main__":
    test_cpu_allreduce()
    test_cuda_allreduce()

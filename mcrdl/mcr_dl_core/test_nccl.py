import torch
import mcrdl   # Your PyBind module

def check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA NOT AVAILABLE. NCCL requires GPU.")
    print("✔ CUDA Available")
    print("GPU Name:", torch.cuda.get_device_name(0))


def test_nccl_allreduce():
    print("\n=== NCCL ALLREDUCE TEST ===")
    comm = mcrdl.Comm("nccl")
    comm.init()

    x = torch.ones(4, dtype=torch.float32).cuda()
    print("[Before]:", x)
    comm.all_reduce(x)
    print("[After ]:", x)  # With 1 GPU → stays same

    comm.finalize()


def test_nccl_broadcast():
    print("\n=== NCCL BROADCAST TEST ===")
    comm = mcrdl.Comm("nccl")
    comm.init()

    if comm.get_rank() == 0:
        x = torch.tensor([5,10,15,20], dtype=torch.float32).cuda()
        print("[Rank 0] Broadcasting:", x)
    else:
        x = torch.empty(4, dtype=torch.float32).cuda()

    comm.broadcast(x)
    print(f"[Rank {comm.get_rank()}] After broadcast:", x)
    comm.finalize()


def test_nccl_allgather():  # using gather → allgather
    print("\n=== NCCL ALLGATHER TEST ===")
    comm = mcrdl.Comm("nccl")
    comm.init()

    rank = comm.get_rank()
    x = torch.tensor([rank], dtype=torch.float32).cuda()
    print(f"[Rank {rank}] Before:", x)

    comm.gather(x)
    print(f"[Rank {rank}] After :", x)  # 1 GPU → no change
    comm.finalize()


if __name__ == "__main__":
    check_cuda()
    test_nccl_allreduce()
    test_nccl_broadcast()
    test_nccl_allgather()

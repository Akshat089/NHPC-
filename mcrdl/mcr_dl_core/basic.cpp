#include <torch/extension.h>
using torch::Tensor;
#include <bits/stdc++.h>
#include <memory>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <mpi.h>

#include <nccl.h>
#include <cuda_runtime.h>

#define NCCL_CHECK(cmd) do {                                     \
    ncclResult_t r = cmd;                                        \
    if (r != ncclSuccess) {                                      \
        cerr << "[NCCL ERROR] " << ncclGetErrorString(r) << "\n"; \
        throw runtime_error("NCCL failure");                     \
    }                                                            \
} while(0)

#define CUDA_CHECK(cmd) do {                                   \
    cudaError_t e = cmd;                                       \
    if (e != cudaSuccess) {                                    \
        cerr << "[CUDA ERROR] " << cudaGetErrorString(e) << "\n"; \
        throw runtime_error("CUDA failure");                    \
    }                                                          \
} while(0)

namespace py = pybind11;
using namespace std;

class Buffer {
private:
    void* data;             // raw pointer to tensor memory
    size_t num_bytes;       // total number of bytes
    bool is_cuda;           // true if tensor is on GPU
    string dtype;           // e.g., "float32", "int64"
    int device_id;          // GPU device index (-1 if CPU)

public:
    Buffer()
        : data(nullptr), num_bytes(0), is_cuda(false), dtype("unknown"), device_id(-1) {}

    Buffer(void* ptr, size_t bytes, bool cuda_flag, string dt, int dev_id)
        : data(ptr), num_bytes(bytes), is_cuda(cuda_flag), dtype(move(dt)), device_id(dev_id) {}

    // Getters
    void* get_data() const { return data; }
    size_t get_num_bytes() const { return num_bytes; }
    bool get_is_cuda() const { return is_cuda; }
    string get_dtype() const { return dtype; }
    int get_device_id() const { return device_id; }

    // Utility function for debugging
    void describe() const {
        cout << "[Buffer] "
             << (is_cuda ? "GPU" : "CPU")
             << " | bytes=" << num_bytes
             << " | dtype=" << dtype
             << " | device=" << device_id
             << endl;
    }
};

Buffer make_buffer_from_tensor(const torch::Tensor& t) {
    TORCH_CHECK(t.is_contiguous(), "Tensor must be contiguous");
    void* data_ptr = t.data_ptr();
    size_t num_bytes = t.numel() * t.element_size();
    bool is_cuda = t.is_cuda();
    int dev_id = -1;
    if (is_cuda) {
        dev_id = t.device().index();
    }
    string dtype(t.dtype().name());
    return Buffer(data_ptr, num_bytes, is_cuda, dtype, dev_id);
}

class Request {
public:
    virtual void test() = 0;
    virtual void wait() = 0;
    virtual ~Request() {}
};

class SimpleRequest : public Request {
public:
    void test() override { cout << "doneeee" << endl; }
    void wait() override { cout << "waiting...\n"; }
};

class Backend {
public:
    virtual void init() { cout << "[Backend] init (default)\n"; }
    virtual void finalize() { cout << "[Backend] finalize (default)\n"; }
    virtual int get_rank() const = 0;
    virtual int get_world_size() const = 0;

    // Core collectives
    virtual void all_reduce(const Buffer& buffer) {
        throw runtime_error("Backend does not support all_reduce()");
    }
    virtual void all_to_all(const Buffer& buffer) {
        throw runtime_error("Backend does not support all_to_all()");
    }
    virtual void gather(const Buffer& buffer) {
        throw runtime_error("Backend does not support gather()");
    }
    virtual void scatter(const Buffer& buffer) {
        throw runtime_error("Backend does not support scatter()");
    }
    virtual void broadcast(const Buffer& buffer) {
        throw runtime_error("Backend does not support broadcast()");
    }

    virtual ~Backend() {}
};

class MPIBackend : public Backend {
    static bool initialized;
    int world_size = 1;
    int rank = 0;

public:
    int get_rank() const override { return rank; }
    int get_world_size() const override { return world_size; }

    void init() override {
        if (!initialized) {
            int no_of_threads;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &no_of_threads);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            initialized = true;

            string level;
            switch (no_of_threads) {
                case MPI_THREAD_SINGLE: level = "MPI_THREAD_SINGLE"; break;
                case MPI_THREAD_FUNNELED: level = "MPI_THREAD_FUNNELED"; break;
                case MPI_THREAD_SERIALIZED: level = "MPI_THREAD_SERIALIZED"; break;
                case MPI_THREAD_MULTIPLE: level = "MPI_THREAD_MULTIPLE"; break;
                default: level = "Unknown"; break;
            }

            cout << "[MPIBackend] init (rank=" << rank
                 << ", size=" << world_size
                 << ", thread_level=" << level << ")\n";
        }
    }

    void finalize() override {
        if (initialized) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            initialized = false;
            cout << "[MPIBackend] finalize\n";
        }
    }

    void all_reduce(const Buffer& buf) override {
        cout << "[MPIBackend] Performing MPI_Allreduce\n";
        buf.describe();

        if (buf.get_dtype() != "float32" && buf.get_dtype() != "float") {
            throw runtime_error("Only float32 supported for now");
        }

        int count = buf.get_num_bytes() / sizeof(float);
        MPI_Allreduce(MPI_IN_PLACE, buf.get_data(), count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }

    void all_to_all(const Buffer& buf) override {
        cout << "[MPIBackend] Performing MPI_Alltoall\n";
        buf.describe();

        if (buf.get_dtype() != "float32" && buf.get_dtype() != "float") {
            throw runtime_error("Only float32 supported for now");
        }

        if (world_size <= 0) {
            throw runtime_error("MPI world_size is invalid (<=0)");
        }

        const int total_count = static_cast<int>(buf.get_num_bytes() / sizeof(float));
        if (total_count <= 0) throw runtime_error("Buffer contains no floats");

        if (world_size == 1) {
            cout << "[MPIBackend] world_size==1: all_to_all is a no-op\n";
            return;
        }

        if (total_count % world_size != 0) {
            throw runtime_error(
                "Tensor size must be divisible by world_size for this all_to_all implementation; use alltoallv for irregular sizes");
        }

        const int send_count = total_count / world_size;
        const int recv_count = send_count;
        vector<float> recvbuf(static_cast<size_t>(total_count), 0.0f);

        cout << "[MPIBackend] world_size=" << world_size
             << " total_count=" << total_count
             << " send_count(per-dest)=" << send_count
             << " buf.addr=" << buf.get_data() << "\n";

        MPI_Alltoall(buf.get_data(), send_count, MPI_FLOAT, recvbuf.data(), recv_count, MPI_FLOAT, MPI_COMM_WORLD);

        memcpy(buf.get_data(), recvbuf.data(), static_cast<size_t>(total_count * sizeof(float)));

        cout << "[MPIBackend] AlltoAll complete (each rank sent " << send_count << " floats to others)\n";
    }

    void gather(const Buffer& buf) override {
        buf.describe();
        int rank = get_rank();
        int world = get_world_size();

        int count = buf.get_num_bytes() / sizeof(float);

        cout << "[DEBUG] Rank " << rank << " local count = " << count << "\n";
        cout << "[DEBUG] Rank " << rank << " buffer before gather: ";
        float* local_ptr = static_cast<float*>(buf.get_data());
        for (int i = 0; i < count; i++) cout << local_ptr[i] << " ";
            cout << "\n";

    // Allocate receive buffer
        vector<float> recvbuf(count * world);

    // Perform Allgather
        MPI_Allgather(local_ptr, count, MPI_FLOAT, recvbuf.data(), count, MPI_FLOAT, MPI_COMM_WORLD);

    // Copy back to local buffer (overwrite or append depending on your convention)
        memcpy(local_ptr, recvbuf.data(), recvbuf.size() * sizeof(float));

        cout << "[DEBUG] Rank " << rank << " buffer after Allgather: ";
        for (size_t i = 0; i < recvbuf.size(); i++) cout << local_ptr[i] << " ";
        cout << "\n";

        cout << "[MPIBackend] AllGather complete — all ranks have full tensor of size "
            << recvbuf.size() << " floats\n";
    }

    void scatter(const Buffer& buf) override {
        buf.describe();
        float* data = static_cast<float*>(buf.get_data());
        size_t count = buf.get_num_bytes() / sizeof(float);

        cout << "[DEBUG] Rank " << get_rank() << " buffer contents: ";
        for (size_t i = 0; i < count; i++) cout << data[i] << " ";
            cout << "\n";
        int rank = get_rank();
        int world = get_world_size();

        size_t total_count = 0;
        if (rank == 0) {
            total_count = buf.get_num_bytes() / sizeof(float);
        }

        MPI_Bcast(&total_count, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        int base_count = total_count / world;
        int remainder = total_count % world;
        vector<int> counts(world, base_count);
        for (int i = 0; i < remainder; i++) counts[i]++;

        vector<int> displs(world, 0);
        for (int i = 1; i < world; i++) {
            displs[i] = displs[i - 1] + counts[i - 1];
        }

        int recv_count = counts[rank];
        vector<float> recvbuf(recv_count);

        float* send_ptr = (rank == 0) ? static_cast<float*>(buf.get_data()) : nullptr;
        float* recv_ptr = static_cast<float*>(buf.get_data());

        cout << "[DEBUG] Rank " << rank << " total_count=" << total_count << "\n";
        cout << "[DEBUG] Rank " << rank << " recv_count=" << recv_count << "\n";
        cout << "[DEBUG] Rank " << rank << " counts: ";
        for (auto c : counts) cout << c << " ";
        cout << "\n";
        cout << "[DEBUG] Rank " << rank << " displs: ";
        for (auto d : displs) cout << d << " ";
        cout << "\n";
        if (rank == 0) {
            cout << "[DEBUG] Rank 0 send buffer: ";
            float* data = static_cast<float*>(buf.get_data());
            for (size_t i = 0; i < total_count; i++) cout << data[i] << " ";
            cout << "\n";
        }

        MPI_Scatterv(send_ptr, counts.data(), displs.data(), MPI_FLOAT,
                     recv_ptr, recv_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

        cout << "[DEBUG] Rank " << rank << " recv buffer after Scatterv: ";
        for (int i = 0; i < recv_count; i++) cout << recv_ptr[i] << " ";
        cout << "\n";

        cout << "[MPIBackend] Scatter complete (rank=" << rank
             << ", recv_count=" << recv_count << ")\n";
    }

    void broadcast(const Buffer& buf) override {
        cout << "[MPIBackend] Performing broadcast\n";
        buf.describe();
        MPI_Bcast(buf.get_data(), buf.get_num_bytes() / sizeof(float), MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
};

bool MPIBackend::initialized = false;


class NCCLBackend : public Backend {
    int rank = 0;
    int world = 1;

    ncclComm_t comm;
    cudaStream_t stream;

public:
    NCCLBackend() : comm(nullptr), stream(nullptr) {};
    int get_rank() const override { return rank; }
    int get_world_size() const override { return world; }

    void init() override {
        cout << "[NCCLBackend] init (SINGLE GPU MODE)\n";
        int dev = 0;
        cudaSetDevice(dev);
        int devs[1] = { dev };

    // This must happen once and must store "comm"
        NCCL_CHECK(ncclCommInitAll(&comm, 1, devs));
        cout << "comm ptr = " << comm << endl;  // Debug line

        cudaStreamCreate(&stream);}


    void finalize() override {
        if (comm) ncclCommDestroy(comm);
        if (stream) cudaStreamDestroy(stream);
        comm = nullptr;
         stream = nullptr;
        }



    // =====================================================================
    // NCCL ALL-REDUCE
    // =====================================================================
    void all_reduce(const Buffer& buf) override {
        cout << "[NCCLBackend] AllReduce\n";
        buf.describe();
        if (!comm) {
                throw std::runtime_error("NCCL COMM IS NULL BEFORE ALLREDUCE");
        }


        if (!buf.get_is_cuda()) {
            throw runtime_error("NCCL requires CUDA tensor");
        }

        if (buf.get_dtype() != "float32" && buf.get_dtype() != "float") {
            throw runtime_error("NCCL demo only supports float32 and float");
        }

        int count = buf.get_num_bytes() / sizeof(float);

        NCCL_CHECK(
            ncclAllReduce(
                buf.get_data(),
                buf.get_data(),
                count,
                ncclFloat32,
                ncclSum,
                comm,
                stream
            )
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // =====================================================================
    // NCCL BROADCAST
    // =====================================================================
    void broadcast(const Buffer& buf) override {
        cout << "[NCCLBackend] Broadcast\n";
        buf.describe();

        if (!buf.get_is_cuda()) {
            throw runtime_error("NCCL requires CUDA memory");
        }

        int count = buf.get_num_bytes() / sizeof(float);

        NCCL_CHECK(
            ncclBroadcast(
                buf.get_data(),
                buf.get_data(),
                count,
                ncclFloat32,
                0,               // root
                comm,
                stream
            )
        );

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // =====================================================================
    // NCCL DOES NOT SUPPORT ALL-TO-ALL NATIVELY
    // You must implement with send/recv + CUDA streams.
    // =====================================================================
    void all_to_all(const Buffer& buf) override {
        buf.describe();
        throw runtime_error("NCCL does not provide native all_to_all — requires custom kernel");
    }

    // =====================================================================
    // NCCL ALL-GATHER (supported)
    // =====================================================================
    void gather(const Buffer& buf) override {
        cout << "[NCCLBackend] AllGather\n";
        buf.describe();

        if (!buf.get_is_cuda()) {
            throw runtime_error("NCCL requires CUDA memory");
        }

        int local_count = buf.get_num_bytes() / sizeof(float);
        int total_count = local_count * world;

        vector<float> recv(total_count);
        float* recv_d = nullptr;

        CUDA_CHECK(cudaMalloc(&recv_d, total_count * sizeof(float)));

        NCCL_CHECK(
            ncclAllGather(
                buf.get_data(),
                recv_d,
                local_count,
                ncclFloat32,
                comm,
                stream
            )
        );

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(buf.get_data(), recv_d,
                              total_count * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaFree(recv_d));
    }

    // =====================================================================
    // NCCL SCATTER — NOT SUPPORTED
    // =====================================================================
    void scatter(const Buffer& buf) override {
        throw runtime_error("NCCL scatter must be manually implemented using send/recv operations");
    }
};

unique_ptr<Backend> create_backend(const string& name) {
    if (name == "mpi") return make_unique<MPIBackend>();
    if (name == "nccl") return make_unique<NCCLBackend>();
    throw runtime_error("Unknown backend: " + name);
}

class Comm {
private:
      unique_ptr<Backend> mpi_backend;
      unique_ptr<Backend> nccl_backend;


public:
    Comm() {
        mpi_backend = create_backend("mpi");
        nccl_backend = create_backend("nccl");
    }

    void init() {
        mpi_backend->init();
        nccl_backend->init();
    }

    void finalize() {
        mpi_backend->finalize();
        nccl_backend->finalize();
    }
    int get_rank() const { return mpi_backend->get_rank(); }
    int get_world_size() const { return mpi_backend->get_world_size(); }

    void all_reduce(const at::Tensor &tensor) {
        Buffer buf = make_buffer_from_tensor(tensor);
        if (buf.get_is_cuda())
            nccl_backend->all_reduce(buf);
        else
            mpi_backend->all_reduce(buf);
    }

   void all_to_all(const at::Tensor& tensor) {
        Buffer buf = make_buffer_from_tensor(tensor);
        if (buf.get_is_cuda())
            nccl_backend->all_to_all(buf);
        else
            mpi_backend->all_to_all(buf);
    }

    void gather(const at::Tensor& tensor) {
        Buffer buf = make_buffer_from_tensor(tensor);
        if (buf.get_is_cuda())
            nccl_backend->gather(buf);
        else
            mpi_backend->gather(buf);
    }

    void scatter(const at::Tensor& tensor) {
        Buffer buf = make_buffer_from_tensor(tensor);
        if (buf.get_is_cuda())
            nccl_backend->scatter(buf);
        else
            mpi_backend->scatter(buf);
    }

    void broadcast(const at::Tensor& tensor) {
        Buffer buf = make_buffer_from_tensor(tensor);
        if (buf.get_is_cuda())
            nccl_backend->broadcast(buf);
        else
            mpi_backend->broadcast(buf);
    }
};

PYBIND11_MODULE(mcrdl, m) {
    py::class_<Comm>(m, "Comm")
        .def(py::init<>())  // no backend name now, Comm manages both internally
        .def("init", &Comm::init)
        .def("finalize", &Comm::finalize)
        .def("all_reduce", &Comm::all_reduce, py::arg("tensor"))
        .def("all_to_all", &Comm::all_to_all, py::arg("tensor"))
        .def("gather", &Comm::gather, py::arg("tensor"))
        .def("scatter", &Comm::scatter, py::arg("tensor"))
        .def("broadcast", &Comm::broadcast, py::arg("tensor"))
        .def("get_rank", &Comm::get_rank)
        .def("get_world_size", &Comm::get_world_size);
}

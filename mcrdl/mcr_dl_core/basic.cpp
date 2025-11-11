#include <bits/stdc++.h>
#include <memory>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <mpi.h>


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
Buffer make_buffer_from_tensor(const torch::Tensor& t){
    TORCH_CHECK(t.is_contiguous(), "Tensor must be contiguous");
    void* data_ptr = t.data_ptr();
    size_t num_bytes = t.numel() * t.element_size();
    bool is_cuda = t.is_cuda();
    int dev_id = -1;
    if(is_cuda){
        dev_id = t.device().index();
    }
    std::string dtype(t.dtype().name());
    return Buffer(data_ptr,num_bytes,is_cuda,dtype,dev_id);
}
class Request {
public:
    virtual void test() = 0;
    virtual void wait() = 0;
    virtual ~Request() {}
};

class SimpleRequest : public Request {
public:
    void test() override { cout << "doneeee"<<endl; }
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
    int get_rank() const { return rank; }
    int get_world_size() const { return world_size; }

    void init() override {
        if(!initialized){
            int no_of_threads;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &no_of_threads);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            initialized = true;
            string level;
            switch (no_of_threads){
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
        MPI_Allreduce(
            MPI_IN_PLACE,
            buf.get_data(),
            count,
            MPI_FLOAT,
            MPI_SUM,
            MPI_COMM_WORLD
        );
    }

    void all_to_all(const Buffer& buf) override {
    cout << "[MPIBackend] Performing MPI_Alltoall\n";
    buf.describe();

    // Validate dtype
    if (buf.get_dtype() != "float32" && buf.get_dtype() != "float") {
        throw runtime_error("Only float32 supported for now");
    }

    // Make sure MPI is initialized and we have a sane world size
    if (world_size <= 0) {
        throw runtime_error("MPI world_size is invalid (<=0)");
    }

    const int total_count = static_cast<int>(buf.get_num_bytes() / sizeof(float));
    if (total_count <= 0) throw runtime_error("Buffer contains no floats");

    // If single rank, nothing to do
    if (world_size == 1) {
        cout << "[MPIBackend] world_size==1: all_to_all is a no-op\n";
        return;
    }

    if (total_count % world_size != 0) {
        // If you want to support arbitrary sizes, switch to Alltoallv.
        throw runtime_error("Tensor size must be divisible by world_size for this all_to_all implementation; use alltoallv for irregular sizes");
    }

    const int send_count = total_count / world_size; // floats per destination
    const int recv_count = send_count; // symmetric case

    // debug
    cout << "[MPIBackend] world_size=" << world_size
         << " total_count=" << total_count
         << " send_count(per-dest)=" << send_count
         << " buf.addr=" << buf.get_data() << "\n";

    // allocate recv buffer (exact size total_count floats)
    vector<float> recvbuf(static_cast<size_t>(total_count), 0.0f);

    // Perform all-to-all safely. Note:
    // - buf.get_data() must be pointer to contiguous float memory.
    // - send_count is per-destination count (floats).
    MPI_Alltoall(
        buf.get_data(),   // sendbuf (void*, treated as float*)
        send_count,       // sendcount per destination
        MPI_FLOAT,
        recvbuf.data(),   // recvbuf
        recv_count,       // recvcount per source
        MPI_FLOAT,
        MPI_COMM_WORLD
    );

    // optional debug sample
    #ifdef DEBUG_MPI
    for (int i = 0; i < min(8, total_count); ++i) cout << recvbuf[i] << " ";
    cout << "\n";
    #endif

    // copy back
    memcpy(buf.get_data(), recvbuf.data(), static_cast<size_t>(total_count * sizeof(float)));

    cout << "[MPIBackend] AlltoAll complete (each rank sent " << send_count << " floats to others)\n";
}


    void gather(const Buffer& buf) override {
    cout << "[MPIBackend] Performing gather\n";
    buf.describe();

    int count = static_cast<int>(buf.get_num_bytes() / sizeof(float));
    int root = 0;

    // Root allocates enough space to hold all data
    std::vector<float> host_recv;
    if (rank == root) {
        host_recv.resize(count * world_size);
    }

    // Perform gather safely
    MPI_Gather(
        buf.get_data(),           // send buffer
        count,                    // send count
        MPI_FLOAT,
        rank == root ? host_recv.data() : nullptr,  // recv buffer only on root
        count,
        MPI_FLOAT,
        root,
        MPI_COMM_WORLD
    );

    if (rank == root) {
        cout << "[MPIBackend] Gather complete (root has "
             << host_recv.size() << " floats)\n";

        // Copy back only to root’s tensor, if needed
        memcpy(buf.get_data(), host_recv.data(), count * sizeof(float));
    }
}

    void scatter(const Buffer& buf) override {
        cout << "[MPIBackend] Performing scatter\n";
        buf.describe();
        int count = buf.get_num_bytes() / sizeof(float) / world_size;
        vector<float> recvbuf(count);
        MPI_Scatter(
            buf.get_data(),
            count,
            MPI_FLOAT,
            recvbuf.data(),
            count,
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD
        );
        memcpy(buf.get_data(), recvbuf.data(), count * sizeof(float));
    }

    void broadcast(const Buffer& buf) override {
        cout << "[MPIBackend] Performing broadcast\n";
        buf.describe();
        MPI_Bcast(
            buf.get_data(),
            buf.get_num_bytes() / sizeof(float),
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD
        );
    }
};


class NCCLBackend : public Backend {
public:
    void init() override {
        cout << "[NCCLBackend] init\n";
    }

    void finalize() override {
        cout << "[NCCLBackend] finalize\n";
    }

    void all_reduce(const Buffer& buf) override {
        cout << "[NCCLBackend] Performing all_reduce\n";
        buf.describe();
    }

    void all_to_all(const Buffer& buf) override {
        cout << "[NCCLBackend] Performing all_to_all\n";
        buf.describe();
    }

    void gather(const Buffer& buf) override {
        cout << "[NCCLBackend] Performing gather\n";
        buf.describe();
    }

    void scatter(const Buffer& buf) override {
        cout << "[NCCLBackend] Performing scatter\n";
        buf.describe();
    }

    void broadcast(const Buffer& buf) override {
        cout << "[NCCLBackend] Performing broadcast\n";
        buf.describe();
    }

    int get_rank() const override { return 0; }         // Dummy implementation
    int get_world_size() const override { return 1; }   // Dummy implementation


};


unique_ptr<Backend> create_backend(const string& name) {
    if (name == "mpi") return make_unique<MPIBackend>();
    if (name == "nccl") return make_unique<NCCLBackend>();
    throw runtime_error("Unknown backend: " + name);
}

class Comm {
private:
    unique_ptr<Backend> backend;

public:
    Comm(const string& backend_name = "mpi") {
        backend = create_backend(backend_name);
    }

    void init() { backend->init(); }
    void finalize() { backend->finalize(); }
    int get_rank() const { return backend->get_rank(); }
    int get_world_size() const { return backend->get_world_size(); }
    // Tensor-based collective
    void all_reduce(const torch::Tensor& tensor) {
        Buffer buf = make_buffer_from_tensor(tensor);

        if (buf.get_is_cuda())
            backend = create_backend("nccl");
        else
            backend = create_backend("mpi");

        backend->all_reduce(buf);
    }

    void all_to_all(const torch::Tensor& tensor) {
        Buffer buf = make_buffer_from_tensor(tensor);
        backend->all_to_all(buf);
    }
    void gather(const torch::Tensor& tensor){
        Buffer buf = make_buffer_from_tensor(tensor);
        backend->gather(buf);
    }
};

PYBIND11_MODULE(mcrdl, m) {
    py::class_<Comm>(m, "Comm")
        .def(py::init<const string&>(), py::arg("backend_name") = "mpi")
        .def("init", &Comm::init)
        .def("finalize", &Comm::finalize)
        .def("all_reduce", &Comm::all_reduce)
        .def("all_to_all", &Comm::all_to_all)
        .def("gather", &Comm::gather)
        .def("get_rank", &Comm::get_rank)
        .def("get_world_size", &Comm::get_world_size);
}
bool MPIBackend::initialized = false;

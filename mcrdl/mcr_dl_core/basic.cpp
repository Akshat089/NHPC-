#include <bits/stdc++.h>
#include <memory>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <torch/extension.h>


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
    void test() override { cout << "doneeee\n"; }
    void wait() override { cout << "waiting...\n"; }
};

class Backend {
public:
    virtual void init() { cout << "[Backend] init (default)\n"; }
    virtual void finalize() { cout << "[Backend] finalize (default)\n"; }

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
public:
    void init() override {
        cout << "[MPIBackend] init\n";
    }

    void finalize() override {
        cout << "[MPIBackend] finalize\n";
    }

    void all_reduce(const Buffer& buf) override {
        cout << "[MPIBackend] Performing all_reduce\n";
        buf.describe();
    }

    void all_to_all(const Buffer& buf) override {
        cout << "[MPIBackend] Performing all_to_all\n";
        buf.describe();
    }

    void gather(const Buffer& buf) override {
        cout << "[MPIBackend] Performing gather\n";
        buf.describe();
    }

    void scatter(const Buffer& buf) override {
        cout << "[MPIBackend] Performing scatter\n";
        buf.describe();
    }

    void broadcast(const Buffer& buf) override {
        cout << "[MPIBackend] Performing broadcast\n";
        buf.describe();
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
};

PYBIND11_MODULE(mcrdl, m) {
    py::class_<Comm>(m, "Comm")
        .def(py::init<const string&>(), py::arg("backend_name") = "mpi")
        .def("init", &Comm::init)
        .def("finalize", &Comm::finalize)
        .def("all_reduce", &Comm::all_reduce)
        .def("all_to_all", &Comm::all_to_all);
}

#include <bits/stdc++.h>
#include <memory>
#include <cstddef>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace std;

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

// ✅ Base backend interface
class Backend {
public:
    virtual void init() { cout << "[Backend] init (default)\n"; }
    virtual void finalize() { cout << "[Backend] finalize (default)\n"; }

    // Core collectives
    virtual void all_reduce() {
        throw runtime_error("Backend does not support all_reduce()");
    }
    virtual void all_to_all() {
        throw runtime_error("Backend does not support all_to_all()");
    }
    virtual void gather() {
        throw runtime_error("Backend does not support gather()");
    }
    virtual void scatter() {
        throw runtime_error("Backend does not support scatter()");
    }
    virtual void broadcast() {
        throw runtime_error("Backend does not support broadcast()");
    }

    virtual ~Backend() {}
};

class HelloBackend : public Backend {
public:
    void init() override { cout << "[HelloBackend] Initialized!\n"; }
    void finalize() override { cout << "[HelloBackend] Finalized!\n"; }

    void all_reduce() override { cout << "[HelloBackend] Fake all_reduce\n"; }
};

class MPIBackend : public Backend {
public:
    void init() override { cout << "[MPIBackend] init\n"; }
    void finalize() override { cout << "[MPIBackend] finalize\n"; }
    void all_to_all() override { cout << "[MPIBackend] Performing all to all\n"; }
    void gather() override {
        cout<<"[MPIBackend] Performing Gather"<<endl;
    }
    void scatter() override{
        cout<<"[MPIBackend] Performing Scatter"<<endl;
    }
};

class NCCLBackend: public Backend{
public:
    void init() override { cout << "[NCCLBackend] init\n"; }
    void finalize() override { cout << "[NCCLBackend] finalize\n"; }
    void all_reduce() override { cout << "[NCCLBackend] Performing all reduce\n"; }
    void broadcast() override{
        cout<<"[NCCLBackend] Performing broadcast"<<endl;
    }
};

unique_ptr<Backend> create_backend(const string& name) {
    if (name == "hello") return make_unique<HelloBackend>();
    if (name == "mpi") return make_unique<MPIBackend>();
    if (name == "nccl") return make_unique<NCCLBackend>();
    throw runtime_error("Unknown backend: " + name);
}

class Comm {
private:
    unique_ptr<Backend> backend;

public:
    Comm(const string& backend_name = "hello") {
        backend = create_backend(backend_name);
    }

    void init() { backend->init(); }
    void finalize() { backend->finalize(); }

    void all_reduce() { backend->all_reduce(); }
    void all_to_all() { backend->all_to_all(); }
};
PYBIND11_MODULE(mcrdl, m) {
    py::class_<Comm>(m, "Comm")
        .def(py::init<const string&>(), py::arg("backend_name") = "mpi")
        .def("init", &Comm::init)
        .def("finalize", &Comm::finalize)
        .def("all_reduce", &Comm::all_reduce)
        .def("all_to_all", &Comm::all_to_all);
}

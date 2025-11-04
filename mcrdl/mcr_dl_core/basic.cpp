#include <bits/stdc++.h>
#include <memory>
#include <cstddef>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace std;

class Request{
public:
    virtual void test() = 0;
    virtual void wait() = 0;

    virtual ~Request() {}
};

class Backend{
public:
    virtual void init() = 0;
    virtual void say_hello() = 0;
    virtual void finalize() = 0;
    virtual ~Backend() {}
};
class HelloBackend : public Backend {
public:
    void init() override {
        cout << "[HelloBackend] Initialized!\n";
    }
    void finalize() override {
        cout << "[HelloBackend] Finalized!\n";
    }
    void say_hello() override {
        cout << "[HelloBackend] Hello, world from backend!\n";
    }
};
class Comm {
private:
    unique_ptr<Backend> backend;

public:
    Comm() {
        backend = make_unique<HelloBackend>();  
    }

    void init() {
        backend->init();
    }

    void say_hello() {
        backend->say_hello();
    }

    void finalize() {
        backend->finalize();
    }
};


PYBIND11_MODULE(mcrdl, m) {
    py::class_<Comm>(m, "Comm")
        .def(py::init<>())
        .def("init", &Comm::init)
        .def("say_hello", &Comm::say_hello)
        .def("finalize", &Comm::finalize);
}

int main() {
    Comm comm;
    comm.init();
    comm.say_hello();
    comm.finalize();
    return 0;
}
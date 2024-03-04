#include <Python.h>
#include "includes.h"

static PyObject* chfem_run(PyObject* self, PyObject *args) {
    char *nf;
    char *raw;

    // Parse the Python arguments to get the two strings
    if (!PyArg_ParseTuple(args, "ss", &nf, &raw)) {
        return NULL; // Return NULL to indicate a failure in parsing arguments
    }

    // Prepare arguments for run_chfem
    char *argv[] = {"", nf, raw};
    int argc = sizeof(argv) / sizeof(argv[0]); // This will be 2

    // Call your C function with these arguments
    int result = run_chfem(argc, argv);

    // Depending on what you want to do with the result, you can return it
    // For simplicity, here we're just returning None, indicating success.
    // You might want to return the actual result or handle errors accordingly.
    Py_RETURN_NONE;
}

static PyMethodDef ChfemTestMethods[] = {
    {"run", chfem_run, METH_VARARGS, "Execute the CHFEM test."},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef chfemtestmodule = {
    PyModuleDef_HEAD_INIT,
    "wrapper", // name of module
    NULL, // module documentation, may be NULL
    -1,   // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    ChfemTestMethods
};

PyMODINIT_FUNC PyInit_wrapper(void) {
    return PyModule_Create(&chfemtestmodule);
}

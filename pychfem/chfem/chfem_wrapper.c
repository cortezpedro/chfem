#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdlib.h> // For malloc and free
#include "includes.h"

static PyObject* chfem_run(PyObject* self, PyObject *args) {
    char *nf;
    PyArrayObject *in_nparray = NULL;

    // Parse the Python arguments to get a string and a numpy array
    if (!PyArg_ParseTuple(args, "sO!", &nf, &PyArray_Type, &in_nparray)) {
        return NULL; // Return NULL to indicate a failure in parsing arguments
    }

    chfemgpuInput_t * user_input = (chfemgpuInput_t *) malloc(sizeof(chfemgpuInput_t));
    initDefaultInput(user_input);

    // Ensure the numpy array is of the expected type (uint8) and 3D
    if (PyArray_TYPE(in_nparray) != NPY_UINT8 || PyArray_NDIM(in_nparray) != 3) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type uint8 and 3D.");
        return NULL;
    }

    // Initialize FEM model for homogenization
    if (!hmgInit(nf, NULL, NULL, PyArray_DATA(in_nparray))){
        printf("Failed to properly read input files.\nProcess aborted.\n");
        printf("#######################################################\n");
        free(user_input);
        return NULL;
    }
    if (runAnalysis(user_input) != 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef PychfemMethods[] = {
    {"run", chfem_run, METH_VARARGS, "Execute the CHFEM test."},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef pychfemModule = {
    PyModuleDef_HEAD_INIT,
    "wrapper", // name of module
    NULL, // module documentation, may be NULL
    -1,   // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    PychfemMethods
};

PyMODINIT_FUNC PyInit_wrapper(void) {
    import_array();  // Necessary for NumPy initialization
    return PyModule_Create(&pychfemModule);
}

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h> // For malloc and free
#include "includes.h"

static PyObject* chfem_run(PyObject* self, PyObject *args) {
    char *nf;
    char *raw;
    PyArrayObject *input_array = NULL;

    // Parse the Python arguments to get a string and a numpy array
    if (!PyArg_ParseTuple(args, "ssO!", &nf, &raw, &PyArray_Type, &input_array)) {
        return NULL; // Return NULL to indicate a failure in parsing arguments
    }

    char *argv[] = {"", nf, raw};
    int argc = sizeof(argv) / sizeof(argv[0]);

    chfemgpuInput_t * user_input = (chfemgpuInput_t *) malloc(sizeof(chfemgpuInput_t));
    initDefaultInput(user_input);

    // Check input
    unsigned char readInput_flag = readInput(argv,argc,user_input);
    if (!readInput_flag){
        free(user_input);
        printf("Failed to parse arguments.\nProcess aborted.\n");
        return 0;
    } else if (readInput_flag==2){ // found -h
        free(user_input);
        return 0;
    }

    // Ensure the numpy array is of the expected type (uint8) and 3D
    if (PyArray_TYPE(input_array) != NPY_UINT8 || PyArray_NDIM(input_array) != 3) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type uint8 and 3D.");
        return NULL;
    }

    uint8_t* data = (uint8_t*)PyArray_DATA(input_array);

    // Initialize FEM model for homogenization
    if (!hmgInit(user_input->neutral_file,user_input->raw_file,user_input->sdf_bin_file, data)){
        printf("Failed to properly read input files.\nProcess aborted.\n");
        printf("#######################################################\n");
        free(user_input);
        return NULL;
    }

    if (run_analysis(user_input) != 0) {
        return NULL;
    }
}

static PyMethodDef ChfemMethods[] = {
    {"run", chfem_run, METH_VARARGS, "Execute the CHFEM test."},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef chfemModule = {
    PyModuleDef_HEAD_INIT,
    "wrapper", // name of module
    NULL, // module documentation, may be NULL
    -1,   // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    ChfemMethods
};

PyMODINIT_FUNC PyInit_wrapper(void) {
    import_array();  // Necessary for NumPy initialization
    return PyModule_Create(&chfemModule);
}

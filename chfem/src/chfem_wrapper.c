#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include <stdlib.h> // For malloc and free
#include "includes.h"

static PyObject* chfem_run(PyObject* self, PyObject *args) {
    char *tmp_nf_file;
    int analysis_type, direction, solver, precondition, output_fields;
    PyArrayObject *in_nparray = NULL;

    // Parse the Python arguments to get a string and a numpy array
    if (!PyArg_ParseTuple(args, "O!siiiii", &PyArray_Type, &in_nparray, &tmp_nf_file, &analysis_type, &direction, &solver, &precondition, &output_fields)) {
        return NULL; // Return NULL to indicate a failure in parsing arguments
    }

    chfemgpuInput_t * user_input = (chfemgpuInput_t *) malloc(sizeof(chfemgpuInput_t));
    initDefaultInput(user_input);

    // passing user_input
    user_input->hmg_direction_flag = direction;
    user_input->solver_flag = solver;
    user_input->preconditioner_flag = precondition;
    user_input->exportFields_flag = output_fields;

    // Initialize FEM model for homogenization
    if (!hmgInit(tmp_nf_file, NULL, NULL, PyArray_DATA(in_nparray))){
        printf("Failed to properly read input files.\nProcess aborted.\n");
        printf("#######################################################\n");
        free(user_input);
        return NULL;
    }
    if (runAnalysis(user_input) != 0) {
        free(user_input);
        return NULL;
    }

    // Create a Python list to return the effective coefficients
    unsigned int n;
    if (analysis_type == 0 || analysis_type == 2) n = 3; // thermal or fluid
    else n = 6;  // elastic
    
    // for thermal expansion coefficients
    unsigned int alpha_n = user_input->num_of_thermal_expansion_coeffs;

    // will store constitutive matrix + thermal expansion in list to be returned
    PyObject* list = PyList_New(n + (alpha_n>0) );
    if (!list) return NULL;
    PyObject* row_list = NULL;
    for (unsigned int i = 0; i < n; i++) {
        row_list = PyList_New(n);
        if (!row_list) {
            Py_DECREF(list);
            return NULL;
        }
        for (unsigned int j = 0; j < n; j++) {
            PyObject* num = PyFloat_FromDouble(user_input->eff_coeff[i*n + j]);
            if (!num) {
                Py_DECREF(row_list);
                Py_DECREF(list);
                return NULL;
            }
            PyList_SetItem(row_list, j, num);
        }
        PyList_SetItem(list, i, row_list);
    }
    
    // add thermal expansion coefficients to the last (n-th) row
    if (alpha_n > 0){
      row_list = PyList_New(alpha_n);
      if (!row_list) {
          Py_DECREF(list);
          return NULL;
      }
      for (unsigned int j = 0; j < alpha_n; j++) {
          PyObject* num = PyFloat_FromDouble(user_input->eff_coeff[n*n + j]);
          if (!num) {
              Py_DECREF(row_list);
              Py_DECREF(list);
              return NULL;
          }
          PyList_SetItem(row_list, j, num);
      }
      PyList_SetItem(list, n, row_list);
    }

    free(user_input);
    return list;
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

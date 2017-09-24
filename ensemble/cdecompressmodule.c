#include <Python.h>

static PyObject *cdecompress_cdecompress(PyObject *self, PyObject *args){
    PyObject *compressed;
    PyObject *probs;

    if (!PyArg_ParseTuple(args, "O&", &compressed, PyArray_DescrConverter, )){
        return NULL;
    }

    return Py_BuildValue("i", *compressed);
}

static PyMethodDef DecompressMethods[] = {
    {"cdecompress", cdecompress_cdecompress, METH_VARARGS,
    "Copy values to numpy data buffer."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcdecompress(void){
    (void) Py_InitModule("cdecompress", DecompressMethods);
}

int main(int argc, char *argv[]){
    Py_SetProgramName(argv[0]);
    Py_Initialize();
    initcdecompress();
}
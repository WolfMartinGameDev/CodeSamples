// this file defines the entry point for the DLL application.
#include "pch.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// wrapper class that holds a Python object representing the Element class
class ElemWrapper : public Element
{
public:
	PyObject* pythonElement = NULL;

	ElemWrapper(PyObject* _pythonElement)
	{
		pythonElement = _pythonElement;
		// increases the reference count so that the garbage collector of Python does not free the memory, if the class is still used in C++
		Py_INCREF(pythonElement);
	}

	~ElemWrapper()
	{
		// decreases the reference count so that the garbage collector of Python knows, that the class is not used anymore in C++
		Py_DECREF(pythonElement);
	}

	// calls the getValue method of the python object and casts the result into a C++ string that is returned
	std::string getValue() override
	{
		PyObject* returnPyString = PyObject_CallMethod(pythonElement, "getValue", "");
		PyObject* returnPyBytes = PyUnicode_AsEncodedString(returnPyString, "utf-8", "~E~");
		std::string returnCString = PyBytes_AS_STRING(returnPyBytes);
		return returnCString;
	}
};

// struct holding a Twice C++ object and its Python wrapper
struct TwiceObject
{
	PyObject_HEAD
	Twice* cppTwice;
	ElemWrapper* parentWrapper;
};

// method that is called from Python to initialize a Twice C++ object and its Python wrapper (reference counter is increased)
static int Twice_init(TwiceObject* self, PyObject* args, PyObject* kwds)
{
	PyObject* parent = NULL;
	
	if (!PyArg_ParseTuple(args, "O", &parent))
	{
		return -1;
	}

	self->parentWrapper = new ElemWrapper(parent);
	self->cppTwice = new Twice(self->parentWrapper);
	return 0;
}

// method that is called from Python to free the memory for a Twice object (reference counter is decreased)
static void Twice_dealloc(TwiceObject* self)
{
	delete self->cppTwice;
	delete self->parentWrapper;
	Py_TYPE(self)->tp_free((PyObject*)self);
}

// method that is called from Python to invoke the C++ Method getValue of the Twice object and return the resulting string as UTF-8 encoded bytes
static PyObject* Twice_getValue(TwiceObject* self, PyObject* Py_UNUSED(ignored))
{
	std::string returnString = self->cppTwice->getValue();
	return PyUnicode_FromString(returnString.c_str());
}

// contains the method definitions of Twice Python objects
static PyMethodDef Twice_methods[] = {
	{"getValue", (PyCFunction)Twice_getValue, METH_NOARGS,
	 "Return the value of the parent twice"
	},
	{NULL} // sentinel value
};

// visitor pattern -> make it possible for Python to visit Twice objects to find out, if there are circular dependencies
static int Twice_traverse(TwiceObject* self, visitproc visit, void* arg)
{
	Py_VISIT(self->parentWrapper->pythonElement);
	return 0;
}

// tells python, that memory can be freed in case of a circular dependency
static int Twice_clear(TwiceObject* self)
{
	Py_CLEAR(self->parentWrapper->pythonElement);
	return 0;
}

// defines the Python Twice class
static PyTypeObject TwiceType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"Twice.Twice",
	sizeof(TwiceObject),
	0,

	// methods to implement standard operations
	(destructor)Twice_dealloc,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
	"Twice objects",
	(traverseproc)Twice_traverse,
	(inquiry)Twice_clear,
	NULL,
	NULL,
	NULL,
	NULL,
	Twice_methods,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	(initproc)Twice_init,
	NULL,
	PyType_GenericNew,
	NULL,
};

BOOL APIENTRY DllMain(HMODULE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	
	return TRUE;
}

// defines the methods of the Python module
static PyMethodDef ScriptBindingMethods[] = {
	{NULL, NULL, 0, NULL}
};

// defines the Python module
static struct PyModuleDef ScriptBindingmodule = {
	PyModuleDef_HEAD_INIT,
	"ScriptBinding", // name of module
	NULL, // module documentation, may be NULL
	-1, // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
	ScriptBindingMethods
};

// defines the initialization of the Python module (includes the Twice class)
PyMODINIT_FUNC
PyInit_ScriptBinding(void)
{
	PyObject* m;
	
	if (PyType_Ready(&TwiceType) < 0)
	{
		return NULL;
	}

	m = PyModule_Create(&ScriptBindingmodule);
	
	if (m == NULL)
	{
		return NULL;
	}

	Py_INCREF(&TwiceType);
	
	if (PyModule_AddObject(m, "Twice", (PyObject*)&TwiceType) < 0)
	{
		Py_DECREF(&TwiceType);
		Py_DECREF(m);
		return NULL;
	}
	
	return m;
}
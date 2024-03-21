import ScriptBinding
import gc

class MyParent(object):
	def getValue(self):
		return "hello world"

parent = MyParent() # instantiate the Python class
twice = ScriptBinding.Twice(parent) # instantiate the native class
parent.twice = twice # circular reference
print(twice.getValue()) # should print "hello world hello world"
twice = None
print(parent.twice.getValue()) # because of the reference count the garbage collection was not triggered and this still works
print(parent.getValue()) # should print "hello world"
parent = None # garbage collection is now possible
gc.collect() # force garbage collection
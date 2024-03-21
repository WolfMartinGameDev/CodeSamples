This folder contains the code for a memory allocator in C++ as well as a short test scenario.
Memory allocators are used to store memory blocks within the preallocated memory range so that they lie close by each other.
This can be used to streamline coding guidelines for memory usage and to minimize allocation times and memory fragmentation.
Processing efficiency can also be increased through the correct use of memory allocators by preventing cache misses
and adjusting memory allocation pools for certain program parts to dedicated hardware specifications (e.g. cache sizes of used consoles or mobile devices).
Altough the code comments propose ways to implement memory allignment (the memory starts with a memory address according to certain criteria -> e.g. divisible by 8)
and padding (the size of the memory blocks is always increased to fit certain criteria -> e.g. divisible by 8) this implementation does not support these features.
Exception handling is also not implemented, but code comments point out potential points of misuse.
#pragma once
#include <vector>
#include <list>
#include <iostream>

struct Block
{
	Block* prev; // circular linked list connecting the previous free memory block in the same size category -> nullptr means allocated
	Block* next; // linked list connecting the next free memory block in the same size category (not circular)
	size_t size; // free memory size without header and footer
};

class Allocator
{
public:
	// initialize allocator and provide a continuous memory where the allocations take place
	Allocator(const size_t availableMemorySize, void* availableMemory);

	// allocate a memory block of size
	// return null if not enough memory is available
	void* alloc(size_t size);

	// free a memory block allocated using alloc
	void free(void* data);

	// writes the block sizes of all blocks and whether they are free or not
	void writeBlocks();

private:
	// m_listHeads[k] contains free memory blocks with 2^k <= size < 2^(k + 1)
	Block* m_listHeads[64];

	// first block of address space
	Block* m_begin;

	// last block of address space
	Block* m_end;
	
	// returns the list index of the list that contains blocks of the specified size
	size_t listIndexOfSize(size_t size);
};
#include "Allocator.h"

// tests the Allocator class by allocating and freeing multiple differently sized memory blocks
int main()
{
	char* memoryBlock[1024];
	Allocator memoryManager(1024, memoryBlock);
	memoryManager.writeBlocks();
	void* allocatedBlock = memoryManager.alloc(128);
	memoryManager.writeBlocks();
	memoryManager.free(allocatedBlock);
	allocatedBlock = nullptr;
	memoryManager.writeBlocks();
	void* allocatedBlock1 = memoryManager.alloc(496);
	memoryManager.writeBlocks();
	void* allocatedBlock2 = memoryManager.alloc(240);
	memoryManager.writeBlocks();
	void* allocatedBlock3 = memoryManager.alloc(112);
	memoryManager.writeBlocks();
	void* allocatedBlock4 = memoryManager.alloc(48);
	memoryManager.writeBlocks();
	void* allocatedBlock5 = memoryManager.alloc(16);
	memoryManager.writeBlocks();
	memoryManager.free(allocatedBlock4);
	allocatedBlock4 = nullptr;
	memoryManager.writeBlocks();
	memoryManager.free(allocatedBlock2);
	allocatedBlock2 = nullptr;
	memoryManager.writeBlocks();
	memoryManager.free(allocatedBlock1);
	allocatedBlock1 = nullptr;
	memoryManager.writeBlocks();
	memoryManager.free(allocatedBlock5);
	allocatedBlock5 = nullptr;
	memoryManager.writeBlocks();
	memoryManager.free(allocatedBlock3);
	allocatedBlock3 = nullptr;
	memoryManager.writeBlocks();
}
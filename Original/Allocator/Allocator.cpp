#include "Allocator.h"

Allocator::Allocator(size_t availableMemorySize, void* availableMemory)
{
	// the available memory is what remains after taking the memory space needed to safe the memory block definition
	// at the beginning of the block and the block size at the end of the block into account
	availableMemorySize -= sizeof(Block);
	availableMemorySize -= sizeof(size_t);

	// for memory alignment more memory size from the memory block may have to be used to make it start from
	// an aligned address (e.g. one that is divisible by 8)
	// for padding more memory size from the memory block may have to be used to allow only a block size of multiples of the padding size
	// a check for a positive remaining memory size and the according exception handling can be added here

	for (int i = 0; i < 64; ++i)
	{
		m_listHeads[i] = nullptr;
	}

	// set the beginning and the end of the allocators memory range to the only block that is currently defined
	m_begin = (Block*)availableMemory;
	m_end = (Block*)availableMemory;

	// put the block into the correct list
	size_t newIndex = listIndexOfSize(availableMemorySize);

	m_listHeads[newIndex] = m_begin;
	m_listHeads[newIndex]->prev = m_listHeads[newIndex];
	m_listHeads[newIndex]->next = nullptr;
	m_listHeads[newIndex]->size = availableMemorySize;

	// write the memory size of the block after the end of the usable memory
	char* memory = reinterpret_cast<char*>(m_begin + 1);
	size_t* newBlockEnd = reinterpret_cast<size_t*>(memory + availableMemorySize);
	*newBlockEnd = availableMemorySize;
}

void* Allocator::alloc(size_t size)
{
	// if a block needs to be divided into two blocks (the remaining free and the allocated memory block), the memory space needed to safe
	// the memory block definition at the beginning of a block and the block size at the end of a block has to be taken into account
	// for padding the size would also be rounded up to a multiple of the padding size
	size_t realSize = size + sizeof(Block) + sizeof(size_t);

	// search for an appropriate memory block to split
	size_t k = listIndexOfSize(2 * size - 1);

	for (; k < 64; ++k)
	{
		if (m_listHeads[k])
		{
			break;
		}
	}

	if (k >= 64)
	{
		return nullptr;
	}
	else
	{
		Block* oldBlock = m_listHeads[k];
		
		// if the found block cannot be split anymore the whole block is returned
		if (oldBlock->size < realSize)
		{
			m_listHeads[k] = oldBlock->next;

			if (m_listHeads[k])
			{
				m_listHeads[k]->prev = oldBlock->prev;
			}

			oldBlock->next = nullptr;
			oldBlock->prev = nullptr;
		}
		else
		{
			// split the block into two blocks where the first one has the needed memory size 
			char* oldMemory = reinterpret_cast<char*>(oldBlock + 1);
			Block* newBlock = reinterpret_cast<Block*>(oldMemory + size + sizeof(size_t));
			char* newMemory = reinterpret_cast<char*>(newBlock + 1);

			newBlock->size = oldBlock->size - realSize;
			oldBlock->size = size;

			// write the memory size of the block after the end of the usable memory
			size_t* oldBlockEnd = reinterpret_cast<size_t*>(oldMemory + oldBlock->size);
			*oldBlockEnd = oldBlock->size;
			size_t* newBlockEnd = reinterpret_cast<size_t*>(newMemory + newBlock->size);
			*newBlockEnd = newBlock->size;

			// if the used block was the last block of the allocators memory the new block becomes the new last block of the allocators memory
			if (m_end == oldBlock)
			{
				m_end = newBlock;
			}

			// take the block out of the list
			m_listHeads[k] = oldBlock->next;

			if (m_listHeads[k])
			{
				m_listHeads[k]->prev = oldBlock->prev;
			}

			oldBlock->next = nullptr;
			oldBlock->prev = nullptr;

			// Put the new free block into the appropriate list
			size_t newIndex = listIndexOfSize(newBlock->size);
			newBlock->next = m_listHeads[newIndex];

			if (m_listHeads[newIndex])
			{
				newBlock->prev = m_listHeads[newIndex]->prev;
				m_listHeads[newIndex]->prev = newBlock;
			}
			else
			{
				newBlock->prev = newBlock;
			}

			m_listHeads[newIndex] = newBlock;
		}

		return oldBlock + 1;
	}
}

void Allocator::free(void* data)
{
	Block* freeBlock = (Block*)data - 1;

	// look for a block in the allocator memory right before the block that should be freed and merge them if the found block is free
	if (m_begin != freeBlock)
	{
		size_t* prevBlockSize = reinterpret_cast<size_t*>(freeBlock) - 1;
		char* prevMemory = reinterpret_cast<char*>(prevBlockSize) - *prevBlockSize;
		Block* prevBlock = reinterpret_cast<Block*>(prevMemory) - 1;
		
		if (prevBlock->prev)
		{
			size_t prevIndex = listIndexOfSize(prevBlock->size);

			prevBlock->size += freeBlock->size + sizeof(Block) + sizeof(size_t);

			if (m_listHeads[prevIndex] == prevBlock)
			{
				m_listHeads[prevIndex] = prevBlock->next;
			}
			
			if (prevBlock->next)
			{
				prevBlock->next->prev = prevBlock->prev;
			}
			else if (m_listHeads[prevIndex])
			{
				m_listHeads[prevIndex]->prev = prevBlock->prev;
			}

			prevBlock->prev->next = prevBlock->next;

			freeBlock = prevBlock;
		}
	}

	// look for a block in the allocator memory right after the block that should be freed and merge them, if the found block is free
	if (m_end != freeBlock)
	{
		char* nextMemory = reinterpret_cast<char*>(freeBlock + 2) + freeBlock->size + sizeof(size_t);
		Block* nextBlock = reinterpret_cast<Block*>(nextMemory) - 1;
		
		if (nextBlock->prev)
		{
			if (m_end == nextBlock)
			{
				m_end = freeBlock;
			}

			size_t nextIndex = listIndexOfSize(nextBlock->size);

			freeBlock->size += nextBlock->size + sizeof(Block) + sizeof(size_t);

			if (m_listHeads[nextIndex] == nextBlock)
			{
				m_listHeads[nextIndex] = nextBlock->next;
			}

			if (nextBlock->next)
			{
				nextBlock->next->prev = nextBlock->prev;
			}
			else if (m_listHeads[nextIndex])
			{
				m_listHeads[nextIndex]->prev = nextBlock->prev;
			}

			nextBlock->prev->next = nextBlock->next;
		}
	}

	// write the memory size of the block after the end of the usable memory
	char* freeMemory = reinterpret_cast<char*>(freeBlock + 1);
	size_t* freeBlockEnd = reinterpret_cast<size_t*>(freeMemory + freeBlock->size);
	*freeBlockEnd = freeBlock->size;

	// put the new free block into the appropriate list
	size_t newIndex = listIndexOfSize(freeBlock->size);

	if (m_listHeads[newIndex])
	{
		freeBlock->prev = m_listHeads[newIndex]->prev;
		freeBlock->next = m_listHeads[newIndex];
	}
	else
	{
		freeBlock->prev = freeBlock;
		freeBlock->next = nullptr;
	}

	m_listHeads[newIndex] = freeBlock;
}

void Allocator::writeBlocks()
{
	std::cout << "Start writeBlocks" << std::endl;
	Block* current = m_begin;
	size_t sum = 0;

	while (true)
	{
		if (current->prev)
		{
			std::cout << "Free block: " << current->size << std::endl;
		}
		else
		{
			std::cout << "Used block: " << current->size << std::endl;
		}

		std::cout << "Header and footer size: " << sizeof(Block) + sizeof(size_t) << std::endl;

		sum += current->size + sizeof(Block) + sizeof(size_t);

		if (current == m_end)
		{
			break;
		}

		char* currentMemory = reinterpret_cast<char*>(current + 1);
		current = reinterpret_cast<Block*>(currentMemory + current->size + sizeof(size_t));
	}

	std::cout << "Sum:" << sum << std::endl;
	std::cout << "End writeBlocks" << std::endl;
}

size_t Allocator::listIndexOfSize(size_t size)
{
	// get the rounded up log2 through bit shifting (better performance)
	if (size == 0)
	{
		return UINT_MAX; // the log2 of 0 is undefined -> exception handling can be added here
	}
	
	size_t log = 0;
	
	while (size > 1)
	{
		size >>= 1;
		log++;
	}
	
	return log;
}
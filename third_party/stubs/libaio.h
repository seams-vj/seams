// Stand-in for libaio, so that DiskANN's in-memory index builds without it: scratch.h includes
// aligned_file_reader.h, which needs the type io_context_t. Only the disk index (linux_aligned_file_reader.cpp,
// not built here) calls libaio.
#pragma once

typedef struct io_context* io_context_t;

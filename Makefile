# Builds the four programs with the compiler flags of the paper's binaries (GCC 11.4).
#   make         bin/build_seams, bin/join, bin/updates, bin/vamana
#   make clean   removes bin/ and build/ (third_party/DiskANN stays)
# bin/updates and bin/vamana link DiskANN: the first build downloads it into third_party/DiskANN and applies
# third_party/diskann.patch (third_party/fetch_diskann.sh, needs git and network access).

DISKANN = third_party/DiskANN

PROGRAM_FLAGS = -std=gnu++17 -O3 -march=native
DISKANN_FLAGS = -std=c++17 -DNDEBUG -Ofast -march=native -mtune=native -mavx2 -mfma -msse2 -ftree-vectorize \
                -fopenmp -fopenmp-simd -funroll-loops -DUSE_AVX2 -DMKL_ILP64 -w -I$(DISKANN)/include -Ithird_party/stubs
UPDATES_FLAGS = $(PROGRAM_FLAGS) -fopenmp -DNDEBUG -DUSE_AVX2 -DMKL_ILP64 -w -I$(DISKANN)/include -Ithird_party/stubs \
                -Isrc

DISKANN_SOURCES = abstract_data_store abstract_index ann_exception distance filter_utils in_mem_data_store \
                  in_mem_graph_store index index_factory logger math_utils memory_mapper natural_number_map \
                  natural_number_set partition pq pq_data_store pq_l2_distance scratch utils
DISKANN_OBJECTS = $(patsubst %,build/diskann/%.o,$(DISKANN_SOURCES))

all: bin/build_seams bin/join bin/updates bin/vamana

bin/build_seams: src/build_seams.cpp src/seams.h
	@mkdir -p bin
	$(CXX) $(PROGRAM_FLAGS) -Isrc src/build_seams.cpp -o $@ -pthread

bin/join: src/join.cpp src/io.h src/order.h
	@mkdir -p bin
	$(CXX) $(PROGRAM_FLAGS) -Isrc src/join.cpp -o $@

bin/updates: src/updates.cpp src/io.h src/order.h src/seams.h $(DISKANN_OBJECTS)
	@mkdir -p bin
	$(CXX) $(UPDATES_FLAGS) src/updates.cpp $(DISKANN_OBJECTS) -o $@ -lgomp -lpthread

bin/vamana: src/vamana.cpp $(DISKANN_OBJECTS)
	@mkdir -p bin
	$(CXX) $(DISKANN_FLAGS) src/vamana.cpp $(DISKANN_OBJECTS) -o $@ -lgomp -lpthread

build/diskann/%.o: build/diskann/fetched
	$(CXX) $(DISKANN_FLAGS) -c $(DISKANN)/src/$*.cpp -o $@

# written once fetch_diskann.sh has checked the patched checkout
build/diskann/fetched: third_party/fetch_diskann.sh third_party/diskann.patch
	sh third_party/fetch_diskann.sh
	@mkdir -p build/diskann
	touch $@

clean:
	rm -rf bin build

.PHONY: all clean

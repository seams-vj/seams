// Builds a Vamana graph over rows of an .fvecs file with DiskANN's dynamic in-memory index, and exports the saved
// index as an NSG graph file in which node r is row r of the .fvecs file.
//
//   vamana build  --data Y.fvecs --ids rows.txt --capacity N --R 70 --L 100 --C 500 --alpha 1.2 --threads T --out G
//   vamana export --index G --points ROWS --out G.nsg
//
// build indexes the rows listed in rows.txt (one row number per line) in an index of capacity N with one frozen start
// point and saves it as DiskANN's files G, G.tags and G.data. export writes all ROWS rows of the .fvecs file; rows that
// are not in the index get an empty list.
#include "index.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using DynamicIndex = diskann::Index<float, uint32_t, uint32_t>;

[[noreturn]] void die(const std::string& message) {
    std::cerr << "vamana: " << message << std::endl;
    std::exit(1);
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Args {
    std::string command;
    std::map<std::string, std::string> values;
    std::set<std::string> used;

    Args(int argc, char** argv) {
        if (argc < 2) {
            die("usage: vamana build|export --key value ...");
        }
        command = argv[1];
        for (int position = 2; position < argc; position += 2) {
            const std::string key = argv[position];
            if (key.rfind("--", 0) != 0 || position + 1 >= argc) {
                die("expected --key value, got " + key);
            }
            values[key.substr(2)] = argv[position + 1];
        }
    }

    std::string text(const std::string& key, const char* fallback = nullptr) {
        used.insert(key);
        if (values.count(key) != 0) {
            return values.at(key);
        }
        if (fallback == nullptr) {
            die("missing --" + key);
        }
        return fallback;
    }

    long long whole(const std::string& key, const char* fallback = nullptr) {
        return std::stoll(text(key, fallback));
    }

    double number(const std::string& key, const char* fallback = nullptr) {
        return std::stod(text(key, fallback));
    }

    void check() const {
        for (const std::pair<const std::string, std::string>& entry : values) {
            if (used.count(entry.first) == 0) {
                die("unknown --" + entry.first);
            }
        }
    }
};

// all rows of an .fvecs file, row-major
struct Rows {
    size_t count = 0;
    size_t dim = 0;
    std::vector<float> values;

    const float* row(size_t index) const {
        return values.data() + index * dim;
    }
};

Rows read_fvecs(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        die("cannot open " + path);
    }
    int32_t dim = 0;
    file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!file || dim <= 0) {
        die(path + ": bad fvecs header");
    }
    file.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(file.tellg());
    const size_t record = sizeof(int32_t) + sizeof(float) * static_cast<size_t>(dim);
    if (bytes % record != 0) {
        die(path + ": size is not a whole number of records");
    }
    Rows rows;
    rows.dim = static_cast<size_t>(dim);
    rows.count = bytes / record;
    rows.values.resize(rows.count * rows.dim);
    file.seekg(0, std::ios::beg);
    for (size_t index = 0; index < rows.count; index++) {
        int32_t record_dim = 0;
        file.read(reinterpret_cast<char*>(&record_dim), sizeof(record_dim));
        if (record_dim != dim) {
            die(path + ": record " + std::to_string(index) + " has another dimension");
        }
        file.read(reinterpret_cast<char*>(rows.values.data() + index * rows.dim), sizeof(float) * rows.dim);
    }
    if (!file) {
        die(path + ": short read");
    }
    return rows;
}

std::vector<uint32_t> read_ids(const std::string& path, size_t row_count) {
    std::ifstream file(path);
    if (!file) {
        die("cannot open " + path);
    }
    std::vector<uint32_t> ids;
    long long row = 0;
    while (file >> row) {
        if (row < 0 || static_cast<size_t>(row) >= row_count) {
            die(path + ": row " + std::to_string(row) + " out of range");
        }
        ids.push_back(static_cast<uint32_t>(row));
    }
    std::vector<uint32_t> sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        die(path + ": a row is listed twice");
    }
    return ids;
}

struct Recipe {
    uint32_t degree = 70;
    uint32_t list = 100;
    uint32_t candidates = 500;
    float alpha = 1.2f;
    uint32_t threads = 1;
    size_t capacity = 0;
};

Recipe read_recipe(Args& args) {
    Recipe recipe;
    recipe.degree = static_cast<uint32_t>(args.whole("R", "70"));
    recipe.list = static_cast<uint32_t>(args.whole("L", "100"));
    recipe.candidates = static_cast<uint32_t>(args.whole("C", "500"));
    recipe.alpha = static_cast<float>(args.number("alpha", "1.2"));
    recipe.threads = static_cast<uint32_t>(args.whole("threads", "1"));
    recipe.capacity = static_cast<size_t>(args.whole("capacity"));
    return recipe;
}

diskann::IndexWriteParameters write_parameters(const Recipe& recipe) {
    return diskann::IndexWriteParametersBuilder(recipe.list, recipe.degree)
        .with_max_occlusion_size(recipe.candidates)
        .with_alpha(recipe.alpha)
        .with_saturate_graph(false)
        .with_num_threads(recipe.threads)
        .build();
}

std::unique_ptr<DynamicIndex> make_index(const Recipe& recipe, size_t dim) {
    std::shared_ptr<diskann::IndexWriteParameters> writing =
        std::make_shared<diskann::IndexWriteParameters>(write_parameters(recipe));
    std::shared_ptr<diskann::IndexSearchParams> searching =
        std::make_shared<diskann::IndexSearchParams>(recipe.list, recipe.threads);
    const size_t frozen_points = 1;
    const bool dynamic_index = true;
    const bool enable_tags = true;
    return std::make_unique<DynamicIndex>(diskann::Metric::L2, dim, recipe.capacity, writing, searching, frozen_points,
                                          dynamic_index, enable_tags);
}

int build(Args& args) {
    const Rows rows = read_fvecs(args.text("data"));
    const std::vector<uint32_t> ids = read_ids(args.text("ids"), rows.count);
    const Recipe recipe = read_recipe(args);
    const std::string out = args.text("out");
    args.check();
    if (ids.size() > recipe.capacity) {
        die("more rows than --capacity");
    }

    // DiskANN reserves tag 0, so the tag of a row is its number + 1
    std::vector<float> block(ids.size() * rows.dim);
    std::vector<uint32_t> tags(ids.size());
    for (size_t position = 0; position < ids.size(); position++) {
        std::copy(rows.row(ids[position]), rows.row(ids[position]) + rows.dim, block.data() + position * rows.dim);
        tags[position] = ids[position] + 1;
    }
    std::unique_ptr<DynamicIndex> index = make_index(recipe, rows.dim);
    const Clock::time_point started = Clock::now();
    index->build(block.data(), ids.size(), tags);
    const double build_seconds = seconds_since(started);
    const bool compact_before_save = true;
    index->save(out.c_str(), compact_before_save);
    std::cout << "built " << ids.size() << " points in " << build_seconds << " s" << std::endl;
    return 0;
}

// the files Index::save writes
struct Saved {
    uint32_t start = 0;
    uint64_t frozen = 0;
    std::vector<std::vector<uint32_t>> lists;   // by location
    std::vector<uint32_t> tags;                 // by location, 0 = the frozen point
    size_t dim = 0;
    std::vector<float> data;                    // by location
};

template <typename Value>
void read_value(std::ifstream& file, Value& value, const std::string& path) {
    file.read(reinterpret_cast<char*>(&value), sizeof(Value));
    if (!file) {
        die(path + ": short read");
    }
}

Saved read_saved(const std::string& prefix) {
    Saved saved;
    {
        std::ifstream file(prefix, std::ios::binary);
        if (!file) {
            die("cannot open " + prefix);
        }
        uint64_t file_size = 0;
        uint32_t max_degree = 0;
        read_value(file, file_size, prefix);
        read_value(file, max_degree, prefix);
        read_value(file, saved.start, prefix);
        read_value(file, saved.frozen, prefix);
        uint32_t degree = 0;
        while (file.read(reinterpret_cast<char*>(&degree), sizeof(degree))) {
            std::vector<uint32_t> list(degree);
            file.read(reinterpret_cast<char*>(list.data()), sizeof(uint32_t) * degree);
            if (!file) {
                die(prefix + ": short adjacency list");
            }
            saved.lists.push_back(std::move(list));
        }
    }
    {
        const std::string path = prefix + ".tags";
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            die("cannot open " + path);
        }
        int32_t count = 0;
        int32_t width = 0;
        read_value(file, count, path);
        read_value(file, width, path);
        if (width != 1 || static_cast<size_t>(count) != saved.lists.size()) {
            die(path + ": does not match the graph");
        }
        saved.tags.resize(static_cast<size_t>(count));
        file.read(reinterpret_cast<char*>(saved.tags.data()), sizeof(uint32_t) * saved.tags.size());
        if (!file) {
            die(path + ": short read");
        }
    }
    {
        const std::string path = prefix + ".data";
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            die("cannot open " + path);
        }
        int32_t count = 0;
        int32_t dim = 0;
        read_value(file, count, path);
        read_value(file, dim, path);
        if (static_cast<size_t>(count) != saved.lists.size()) {
            die(path + ": does not match the graph");
        }
        saved.dim = static_cast<size_t>(dim);
        saved.data.resize(saved.lists.size() * saved.dim);
        file.read(reinterpret_cast<char*>(saved.data.data()), sizeof(float) * saved.data.size());
        if (!file) {
            die(path + ": short read");
        }
    }
    if (saved.frozen != 1) {
        die(prefix + ": expected one frozen point, found " + std::to_string(saved.frozen));
    }
    if (saved.start >= saved.lists.size() || saved.tags[saved.start] != 0) {
        die(prefix + ": the start is not frozen");
    }
    return saved;
}

double squared_distance(const float* left, const float* right, size_t dim) {
    double sum = 0.0;
    for (size_t coordinate = 0; coordinate < dim; coordinate++) {
        const double difference = static_cast<double>(left[coordinate]) - static_cast<double>(right[coordinate]);
        sum += difference * difference;
    }
    return sum;
}

// DiskANN's start is a frozen copy of the medoid: it is dropped with every edge into it, and the entry point of the
// NSG file is the indexed row nearest to it.
int export_nsg(Args& args) {
    const std::string prefix = args.text("index");
    const size_t points = static_cast<size_t>(args.whole("points"));
    const std::string out = args.text("out");
    args.check();
    const Saved saved = read_saved(prefix);

    const float* frozen_vector = saved.data.data() + saved.start * saved.dim;
    double nearest = std::numeric_limits<double>::max();
    uint32_t entry = 0;
    for (size_t location = 0; location < saved.lists.size(); location++) {
        if (location == saved.start) {
            continue;
        }
        const double distance = squared_distance(frozen_vector, saved.data.data() + location * saved.dim, saved.dim);
        if (distance < nearest) {
            nearest = distance;
            entry = saved.tags[location] - 1;
        }
    }

    std::vector<std::vector<uint32_t>> lists(points);
    std::vector<char> indexed(points, 0);
    for (size_t location = 0; location < saved.lists.size(); location++) {
        if (location == saved.start) {
            continue;
        }
        const uint32_t tag = saved.tags[location];
        if (tag == 0 || tag > points) {
            die(prefix + ": location " + std::to_string(location) + " has tag " + std::to_string(tag));
        }
        const uint32_t row = tag - 1;
        if (indexed[row]) {
            die(prefix + ": row " + std::to_string(row) + " appears twice");
        }
        indexed[row] = 1;
        for (uint32_t neighbour : saved.lists[location]) {
            if (neighbour >= saved.lists.size()) {
                die(prefix + ": edge to a missing location");
            }
            if (neighbour == saved.start) {
                continue;
            }
            lists[row].push_back(saved.tags[neighbour] - 1);
        }
    }

    size_t max_degree = 0;
    size_t edges = 0;
    size_t indexed_count = 0;
    for (size_t row = 0; row < points; row++) {
        max_degree = std::max(max_degree, lists[row].size());
        edges += lists[row].size();
        indexed_count += indexed[row];
    }
    {
        std::ofstream file(out, std::ios::binary);
        if (!file) {
            die("cannot write " + out);
        }
        const uint32_t max_degree_field = static_cast<uint32_t>(max_degree);
        file.write(reinterpret_cast<const char*>(&max_degree_field), sizeof(max_degree_field));
        file.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
        for (size_t row = 0; row < points; row++) {
            const uint32_t degree = static_cast<uint32_t>(lists[row].size());
            file.write(reinterpret_cast<const char*>(&degree), sizeof(degree));
            file.write(reinterpret_cast<const char*>(lists[row].data()), sizeof(uint32_t) * degree);
        }
    }
    {
        std::ifstream file(out, std::ios::binary);
        uint32_t max_degree_back = 0;
        uint32_t entry_back = 0;
        read_value(file, max_degree_back, out);
        read_value(file, entry_back, out);
        size_t rows_back = 0;
        size_t edges_back = 0;
        uint32_t degree = 0;
        while (file.read(reinterpret_cast<char*>(&degree), sizeof(degree))) {
            file.seekg(static_cast<std::streamoff>(sizeof(uint32_t) * degree), std::ios::cur);
            rows_back++;
            edges_back += degree;
        }
        if (rows_back != points || edges_back != edges || entry_back != entry) {
            die(out + ": read back differs");
        }
    }
    std::cout << "exported " << indexed_count << " of " << points << " rows, entry " << entry << ", mean degree "
              << (indexed_count ? static_cast<double>(edges) / static_cast<double>(indexed_count) : 0.0) << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args args(argc, argv);
    if (args.command == "build") {
        return build(args);
    }
    if (args.command == "export") {
        return export_nsg(args);
    }
    die("unknown command " + args.command);
}

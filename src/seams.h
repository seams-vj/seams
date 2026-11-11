// seams.h - Build of the seam table, shared by build_seams and updates. Build runs one top-k search per query on the
// graph index of the data (PoolSearch: its candidate list, the "pool"), Profile over the pool (Reach: the in-range
// regions at every rank) and Select (LargestRegions: the births to keep as seams); BuildWorker runs both over the pool
// of one query. Also here: the readers of .nsg and .fvecs, the loop over the queries on threads and a small JSON
// writer for the run summaries.
// Header only, C++17: #include "seams.h".
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace seams {

// seconds on a steady clock
inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[noreturn]] inline void die(const std::string& message) {
    std::cerr << "error: " << message << "\n";
    std::exit(1);
}

inline double peak_rss_mib() {
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return double(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return double(usage.ru_maxrss) / 1024.0;
#endif
}

// ----------------------------------------------------------------------------------------------------- threads ----

// One worker (the scratch memory of one thread) per thread. A thread makes its own worker the first time it needs it,
// so the worker lives in that thread's memory and two threads never write to one cache line.
template <class Worker>
class Workers {
public:
    // every worker is made as Worker(arguments...); the arguments are kept by reference
    template <class... Arguments>
    explicit Workers(int threads, const Arguments&... arguments) : slots_(size_t(threads)) {
        make_ = [kept = std::tuple<const Arguments&...>(arguments...)]() {
            return std::apply([](const Arguments&... values) { return std::make_unique<Worker>(values...); }, kept);
        };
    }

    size_t size() const {
        return slots_.size();
    }

    // only thread `index` calls this
    Worker& own(size_t index) {
        if (!slots_[index]) {
            slots_[index] = make_();
        }
        return *slots_[index];
    }

    // the workers made so far
    std::vector<const Worker*> made() const {
        std::vector<const Worker*> workers;
        for (const std::unique_ptr<Worker>& slot : slots_) {
            if (slot) {
                workers.push_back(slot.get());
            }
        }
        return workers;
    }

private:
    std::vector<std::unique_ptr<Worker>> slots_;
    std::function<std::unique_ptr<Worker>()> make_;
};

// body(worker, query) for every query, one thread per worker. A thread takes the next query when it is done with one,
// since the work per query varies a lot. body writes only its worker and the outputs of its query, so the outputs do
// not depend on the number of threads.
template <class Worker, class Body>
void for_each_query(Workers<Worker>& workers, size_t queries, Body body) {
    std::atomic<size_t> next_query{0};
    std::vector<std::thread> threads;
    for (size_t index = 0; index < workers.size(); ++index) {
        threads.emplace_back([&workers, &next_query, &body, queries, index]() {
            Worker& worker = workers.own(index);
            for (size_t query = next_query++; query < queries; query = next_query++) {
                body(worker, query);
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
}

// ------------------------------------------------------------------------------------------------------ inputs ----

// .nsg: [width u32][entry u32], then per node [count u32][count x u32]; read into compressed rows
struct Graph {
    uint32_t width = 0;
    uint32_t entry = 0;
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> neighbours;

    size_t size() const {
        return offsets.size() - 1;
    }
    const uint32_t* begin(size_t node) const {
        return neighbours.data() + offsets[node];
    }
    const uint32_t* end(size_t node) const {
        return neighbours.data() + offsets[node + 1];
    }
};

inline Graph read_nsg(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        die("cannot open nsg: " + path);
    }
    Graph graph;
    file.read(reinterpret_cast<char*>(&graph.width), 4);
    file.read(reinterpret_cast<char*>(&graph.entry), 4);
    if (!file) {
        die("bad nsg header: " + path);
    }
    graph.offsets.push_back(0);
    uint32_t count = 0;
    while (file.read(reinterpret_cast<char*>(&count), 4)) {
        const size_t first = graph.neighbours.size();
        graph.neighbours.resize(first + count);
        if (count != 0 &&
            !file.read(reinterpret_cast<char*>(graph.neighbours.data() + first), std::streamsize(count) * 4)) {
            die("truncated nsg: " + path);
        }
        graph.offsets.push_back(graph.neighbours.size());
    }
    return graph;
}

// The candidate list ("pool") of every query, in rank order (rank 0 = nearest), packed: the pool of query q is
// ids[offsets[q] .. offsets[q + 1]) with the distances at the same positions.
struct Pools {
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> ids;
    std::vector<double> dist;

    size_t queries() const {
        return offsets.size() - 1;
    }
    size_t length(size_t query) const {
        return offsets[query + 1] - offsets[query];
    }
    const uint32_t* ids_of(size_t query) const {
        return ids.data() + offsets[query];
    }
    const double* dist_of(size_t query) const {
        return dist.data() + offsets[query];
    }
    // the longest pool
    size_t widest() const {
        size_t widest_length = 0;
        for (size_t query = 0; query < queries(); ++query) {
            widest_length = std::max(widest_length, length(query));
        }
        return widest_length;
    }
};

// .fvecs: per row [dim i32][dim x f32]; read into one row-major array
struct Vectors {
    size_t count = 0;
    size_t dim = 0;
    std::vector<float> values;

    const float* row(size_t index) const {
        return values.data() + index * dim;
    }
};

// expected_count > 0: the file must hold exactly that many rows (the nodes of the graph)
inline Vectors read_fvecs(const std::string& path, size_t expected_count = 0) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        die("cannot open fvecs: " + path);
    }
    Vectors vectors;
    int32_t dimension = 0;
    if (!file.read(reinterpret_cast<char*>(&dimension), 4) || dimension <= 0) {
        die("bad fvecs: " + path);
    }
    vectors.dim = size_t(dimension);
    file.seekg(0, std::ios::end);
    const size_t bytes = size_t(file.tellg());
    const size_t row_bytes = 4 + 4 * vectors.dim;
    if (bytes % row_bytes != 0) {
        die("fvecs size is not a multiple of the row: " + path);
    }
    vectors.count = bytes / row_bytes;
    if (expected_count != 0 && vectors.count != expected_count) {
        die("fvecs rows " + std::to_string(vectors.count) + " != graph nodes " + std::to_string(expected_count));
    }
    vectors.values.resize(vectors.count * vectors.dim);
    file.seekg(0);
    for (size_t index = 0; index < vectors.count; ++index) {
        if (!file.read(reinterpret_cast<char*>(&dimension), 4) || size_t(dimension) != vectors.dim) {
            die("fvecs: bad row " + std::to_string(index));
        }
        char* row = reinterpret_cast<char*>(vectors.values.data() + index * vectors.dim);
        if (!file.read(row, std::streamsize(4 * vectors.dim))) {
            die("fvecs: truncated row " + std::to_string(index));
        }
    }
    return vectors;
}

// the distance of Build: L2 over float rows, each difference and the sum in double
inline double l2_distance(const float* left, const float* right, size_t dim) {
    double sum = 0.0;
    for (size_t index = 0; index < dim; ++index) {
        const double difference = double(left[index]) - double(right[index]);
        sum += difference * difference;
    }
    return std::sqrt(sum);
}

// a distance as the seam table prints it (12 significant digits) and reads it back
inline double as_printed(double dist) {
    char text[40];
    std::snprintf(text, sizeof text, "%.12g", dist);
    return std::strtod(text, nullptr);
}

// ------------------------------------------------------------------------------------------- the top-k search ----

// The candidate list of one query: the beam search of the graph index from its entry point. It keeps the `width`
// closest vectors seen, in distance order (a new vector goes before the equal distances already there), expands the
// closest unexpanded one and stops when none is left. A worker of one thread: the visited marks are its own.
class PoolSearch {
public:
    struct Candidate {
        double dist;
        uint32_t id;
        bool unexpanded;
    };

    explicit PoolSearch(const Graph& graph, const Vectors& base)
        : graph_(graph), base_(base), visited_in_(graph.size(), 0) {}

    const std::vector<Candidate>& run(const float* query, size_t width) {
        next_epoch();
        pool_.clear();
        pool_.reserve(width + 1);
        const uint32_t entry = graph_.entry;
        if (entry >= graph_.size()) {
            return pool_;
        }
        first_visit(entry);
        pool_.push_back({l2_distance(query, base_.row(entry), base_.dim), entry, true});
        ++distances_;
        size_t position = 0;
        while (position < pool_.size()) {
            if (!pool_[position].unexpanded) {
                ++position;
                continue;
            }
            pool_[position].unexpanded = false;
            const uint32_t node = pool_[position].id;
            size_t next_position = position + 1;
            for (const uint32_t* edge = graph_.begin(node); edge != graph_.end(node); ++edge) {
                const uint32_t neighbour = *edge;
                if (neighbour >= graph_.size()) {
                    continue;
                }
                if (!first_visit(neighbour)) {
                    continue;
                }
                const double dist = l2_distance(query, base_.row(neighbour), base_.dim);
                ++distances_;
                if (pool_.size() >= width && dist >= pool_.back().dist) {
                    continue;
                }
                const Candidate candidate{dist, neighbour, true};
                std::vector<Candidate>::iterator slot = std::lower_bound(
                    pool_.begin(), pool_.end(), candidate,
                    [](const Candidate& left, const Candidate& right) { return left.dist < right.dist; });
                const size_t inserted_at = size_t(slot - pool_.begin());
                pool_.insert(slot, candidate);
                if (pool_.size() > width) {
                    pool_.pop_back();
                }
                if (inserted_at < next_position) {
                    next_position = inserted_at;
                }
            }
            position = next_position;
        }
        return pool_;
    }

    // distances computed by this worker so far
    uint64_t distances() const {
        return distances_;
    }

private:
    // marks the node; false when it was marked already in this search
    bool first_visit(uint32_t node) {
        if (visited_in_[node] == epoch_) {
            return false;
        }
        visited_in_[node] = epoch_;
        return true;
    }

    void next_epoch() {
        if (++epoch_ == 0) {
            std::fill(visited_in_.begin(), visited_in_.end(), 0);
            epoch_ = 1;
        }
    }

    const Graph& graph_;
    const Vectors& base_;
    std::vector<uint32_t> visited_in_;   // per node: the epoch of the last search that visited it
    uint32_t epoch_ = 0;
    uint64_t distances_ = 0;
    std::vector<Candidate> pool_;
};

// The candidate lists of all queries, searched with width `width` on `threads` threads. The distances are kept as the
// seam table prints them (as_printed), so every later comparison is the one a reader of the table makes.
// `distances` = the distances computed.
inline Pools search_pools(const Graph& graph, const Vectors& base, const Vectors& queries, size_t width, int threads,
                          uint64_t& distances) {
    if (queries.dim != base.dim) {
        die("queries and data have different dimensionality");
    }
    const size_t count = queries.count;
    std::vector<uint32_t> ids(count * width);
    std::vector<double> dist(count * width);
    std::vector<size_t> lengths(count, 0);
    Workers<PoolSearch> workers(threads, graph, base);
    for_each_query(workers, count, [&](PoolSearch& search, size_t query) {
        const std::vector<PoolSearch::Candidate>& pool = search.run(queries.row(query), width);
        lengths[query] = pool.size();
        for (size_t rank = 0; rank < pool.size(); ++rank) {
            ids[query * width + rank] = pool[rank].id;
            dist[query * width + rank] = as_printed(pool[rank].dist);
        }
    });
    distances = 0;
    for (const PoolSearch* search : workers.made()) {
        distances += search->distances();
    }

    // the pools sit in slots of `width` per query; pack them
    Pools pools;
    pools.offsets.assign(1, 0);
    size_t packed = 0;
    for (size_t query = 0; query < count; ++query) {
        const size_t from = query * width;
        if (packed != from) {
            std::copy(ids.begin() + std::ptrdiff_t(from), ids.begin() + std::ptrdiff_t(from + lengths[query]),
                      ids.begin() + std::ptrdiff_t(packed));
            std::copy(dist.begin() + std::ptrdiff_t(from), dist.begin() + std::ptrdiff_t(from + lengths[query]),
                      dist.begin() + std::ptrdiff_t(packed));
        }
        packed += lengths[query];
        pools.offsets.push_back(packed);
    }
    ids.resize(packed);
    dist.resize(packed);
    pools.ids = std::move(ids);
    pools.dist = std::move(dist);
    return pools;
}

// --------------------------------------------------------------------------------------------- the guarantee ----

// how many of the `seen` points the seams must reach so that at most a share delta of them is missed:
// 1 - reached / seen <= delta  <=>  reached >= ceil((1 - delta) seen), at least 1
inline int points_to_reach(int seen, double delta) {
    const int keep = int(std::ceil(double(seen) * (1.0 - delta) - 1e-9));
    return std::max(1, std::min(keep, seen));
}

// The thresholds a seam choice serves: theta is unknown offline, so every theta_j = d(x, p_j) of the query's own pool
// is checked while the pool still shows the whole answer, i.e. theta_j < d(x, p_L) = r_L.
inline bool inside_pool(const double* dist, size_t length, size_t rank) {
    return dist[rank] < dist[length - 1];
}

// ----------------------------------------------------------------------------------------- Profile (Reach) ----

// Profile over the pool of one query: the points are added in rank order, and label[p] = 1 + the rank of the closest
// point that reaches p along out-edges of the data graph through the points added so far. A point is a birth when no
// closer point has an out-edge into it: its label is then its own 1 + rank. The labels in use are births, and they
// split the points added so far into in-range regions, one per birth.
class Reach {
public:
    static constexpr int NONE = std::numeric_limits<int>::max();

    explicit Reach(const Graph& graph) : graph_(graph), rank_of_(graph.size(), -1) {}

    // the pool of query `query`: ids[0 .. length)
    void begin(const uint32_t* ids, size_t length, size_t query) {
        ids_ = ids;
        length_ = length;
        for (size_t rank = 0; rank < length_; ++rank) {
            if (rank_of_[ids_[rank]] != -1) {
                die("duplicate point in the pool of x = " + std::to_string(query));
            }
            rank_of_[ids_[rank]] = int32_t(rank);
        }
        if (closer_.size() < length_) {
            closer_.resize(length_);
        }
        for (size_t rank = 0; rank < length_; ++rank) {
            closer_[rank].clear();
        }
        label_.assign(length_, NONE);
    }

    // Adds p_rank and lowers the labels it reaches; on_label(old, value) runs for every label change, old = 0 when the
    // point had no label. Returns true when p_rank is a birth.
    template <class OnLabel>
    bool add(size_t rank, OnLabel&& on_label) {
        int own = int(rank) + 1;
        for (int32_t earlier : closer_[rank]) {
            own = std::min(own, label_[size_t(earlier)]);
        }
        set_label(rank, own, on_label);
        for (const uint32_t* edge = graph_.begin(ids_[rank]); edge != graph_.end(ids_[rank]); ++edge) {
            const int32_t target = rank_of_[*edge];
            if (target < 0) {
                continue;
            }
            if (size_t(target) > rank) {
                closer_[size_t(target)].push_back(int32_t(rank));
                continue;
            }
            if (size_t(target) == rank) {
                continue;
            }
            // relabel: push the label `own` along out-edges through the points added so far
            stack_.push_back({target, own});
            while (!stack_.empty()) {
                const std::pair<int32_t, int> top = stack_.back();
                stack_.pop_back();
                if (label_[size_t(top.first)] <= top.second) {
                    continue;
                }
                set_label(size_t(top.first), top.second, on_label);
                const uint32_t node = ids_[size_t(top.first)];
                for (const uint32_t* next = graph_.begin(node); next != graph_.end(node); ++next) {
                    const int32_t next_rank = rank_of_[*next];
                    if (next_rank >= 0 && size_t(next_rank) <= rank) {
                        stack_.push_back({next_rank, top.second});
                    }
                }
            }
        }
        return own == int(rank) + 1;
    }

    void end() {
        for (size_t rank = 0; rank < length_; ++rank) {
            rank_of_[ids_[rank]] = -1;
        }
        length_ = 0;
    }

    // label changes of points that had a label already
    uint64_t relabels() const {
        return relabels_;
    }

private:
    template <class OnLabel>
    void set_label(size_t rank, int value, OnLabel& on_label) {
        int old = label_[rank];
        if (old == NONE) {
            old = 0;
        } else {
            ++relabels_;
        }
        label_[rank] = value;
        on_label(old, value);
    }

    const Graph& graph_;
    std::vector<int32_t> rank_of_;               // per node of the graph: its rank in the current pool, or -1
    const uint32_t* ids_ = nullptr;
    size_t length_ = 0;
    std::vector<int> label_;                     // per rank
    std::vector<std::vector<int32_t>> closer_;   // per rank: the lower ranks with an edge into it
    std::vector<std::pair<int32_t, int>> stack_;
    uint64_t relabels_ = 0;
};

// ----------------------------------------------------------------------------------------------- Select ----

// Region sizes: count[b] = the points labelled b; `ordered` holds (count, -b), smallest region first and, among equal
// sizes, the farther birth first.
class Regions {
public:
    void begin(size_t length) {
        count_.assign(length + 1, 0);
        ordered_.clear();
    }

    void change(int birth, int amount) {
        if (count_[size_t(birth)] > 0) {
            ordered_.erase({count_[size_t(birth)], -birth});
        }
        count_[size_t(birth)] += amount;
        if (count_[size_t(birth)] > 0) {
            ordered_.insert({count_[size_t(birth)], -birth});
        }
    }

    int count(int birth) const {
        return count_[size_t(birth)];
    }

    const std::set<std::pair<int, int>>& ordered() const {
        return ordered_;
    }

private:
    std::vector<int> count_;
    std::set<std::pair<int, int>> ordered_;
};

// Select: at every checked threshold keep the largest regions until ceil((1 - delta) n) of the n points seen are
// covered - that is, drop the smallest regions while their total fits in the n - ceil((1 - delta) n) points that may be
// missed - and keep their births. The seams are the union over the checked thresholds, plus rank 0 (early pruning
// reads its distance). A region holds at most the points its seam reaches, so the guarantee holds.
class LargestRegions {
public:
    void begin(size_t length) {
        regions_.begin(length);
        chosen_at_.assign(length + 1, NOT_CHOSEN);
        waiting_flag_.assign(length + 1, 0);
        waiting_.clear();
    }

    // follows a label change of Reach::add
    void on_label(int old, int value) {
        if (old) {
            regions_.change(old, -1);
            if (regions_.count(old) == 0 && waiting_flag_[size_t(old)]) {
                waiting_flag_[size_t(old)] = 0;
                waiting_.erase(std::find(waiting_.begin(), waiting_.end(), old));
            }
        }
        regions_.change(value, +1);
    }

    // a birth of Reach::add, with its label (1 + rank)
    void on_birth(int birth) {
        waiting_.push_back(birth);
        waiting_flag_[size_t(birth)] = 1;
    }

    // the choice at the threshold of rank `rank`, n = rank + 1 points seen; returns the number of regions kept (need_j)
    int choose(int rank, double delta) {
        const int seen = rank + 1;
        int budget = seen - points_to_reach(seen, delta);
        int dropped = 0;
        std::pair<int, int> last_dropped{0, 0};   // below every real key
        for (std::set<std::pair<int, int>>::const_iterator region = regions_.ordered().begin();
             region != regions_.ordered().end() && region->first <= budget; ++region) {
            budget -= region->first;
            last_dropped = *region;
            ++dropped;
        }
        size_t index = 0;
        while (index < waiting_.size()) {
            const int birth = waiting_[index];
            if (std::make_pair(regions_.count(birth), -birth) > last_dropped) {
                chosen_at_[size_t(birth)] = rank;
                waiting_flag_[size_t(birth)] = 0;
                waiting_[index] = waiting_.back();
                waiting_.pop_back();
            } else {
                ++index;
            }
        }
        return int(regions_.ordered().size()) - dropped;
    }

    // the ranks of the seams, ascending: rank 0 and every birth kept at some threshold
    std::vector<int32_t> seams(size_t length) const {
        std::vector<int32_t> ranks;
        for (size_t birth = 1; birth <= length; ++birth) {
            if (birth == 1 || chosen_at_[birth] < NOT_CHOSEN) {
                ranks.push_back(int32_t(birth - 1));
            }
        }
        return ranks;
    }

private:
    static constexpr int NOT_CHOSEN = std::numeric_limits<int>::max();
    Regions regions_;
    std::vector<int> chosen_at_;      // per birth: the first threshold rank at which it was kept
    std::vector<char> waiting_flag_;  // per birth: 1 while it is in `waiting_`
    std::vector<int> waiting_;        // births in use, not kept yet
};

// ------------------------------------------------------------------------------------ Profile and Select ----

// The memory of one Build thread, reused query after query.
struct BuildWorker {
    Reach reach;
    LargestRegions choice;

    explicit BuildWorker(const Graph& graph) : reach(graph) {}

    // Profile and Select over the candidate list of one query: the ranks of its seams, ascending; none for an empty
    // list. Select runs at every rank whose threshold the list serves (inside_pool).
    std::vector<int32_t> select(const Pools& pools, size_t query, double delta) {
        std::vector<int32_t> seam_ranks;
        const size_t length = pools.length(query);
        if (length == 0) {
            return seam_ranks;
        }
        const double* distances = pools.dist_of(query);
        auto track_regions = [this](int old_label, int new_label) {
            choice.on_label(old_label, new_label);
        };
        reach.begin(pools.ids_of(query), length, query);
        choice.begin(length);
        for (size_t rank = 0; rank < length; ++rank) {
            if (reach.add(rank, track_regions)) {
                choice.on_birth(int(rank) + 1);
            }
            if (inside_pool(distances, length, rank)) {
                choice.choose(int(rank), delta);
            }
        }
        seam_ranks = choice.seams(length);
        reach.end();
        return seam_ranks;
    }
};

// ---------------------------------------------------------------------------------------------- run summaries ----

// A small JSON object: the keys in the order they were added, the values already formatted.
class Json {
public:
    Json& text(const std::string& key, const std::string& value) {
        return raw(key, "\"" + value + "\"");
    }

    template <class Number>
    Json& number(const std::string& key, Number value) {
        std::ostringstream formatted;
        formatted << std::setprecision(10) << value;
        return raw(key, formatted.str());
    }

    Json& raw(const std::string& key, const std::string& value) {
        fields_.push_back({key, value});
        return *this;
    }

    // the object as text; `indent` = the indent of its fields
    std::string str(int indent = 2) const {
        std::string pad(size_t(indent), ' ');
        std::string result = "{\n";
        for (size_t index = 0; index < fields_.size(); ++index) {
            result += pad + "\"" + fields_[index].first + "\": " + fields_[index].second;
            result += index + 1 < fields_.size() ? ",\n" : "\n";
        }
        result += std::string(size_t(std::max(0, indent - 2)), ' ') + "}";
        return result;
    }

    void write(const std::string& path) const {
        std::ofstream file(path);
        if (!file) {
            die("cannot write " + path);
        }
        file << str() << "\n";
    }

private:
    std::vector<std::pair<std::string, std::string>> fields_;
};

}  // namespace seams

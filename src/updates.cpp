// updates.cpp - the join of the queries X with the data Y while Y changes and new queries arrive, with the seam table
// maintained in the background (async), synchronously (sync: the probes wait for every round) or not at all (none).
// Three kinds of threads run at once. Probe threads probe one query at a time on the live graph index of Y, cycling
// over X in its MST order: probe i is due at i / query-rate, and its latency is finish - due. Update threads apply the
// lines of the update file at a fixed rate: "i ROW" inserts and "d ROW" deletes row ROW of Y in the dynamic Vamana
// index (DiskANN), "q X" is the arrival of query X, whose row of the seam table is dropped at the start. A maintenance
// thread starts a round every --maintain-every applied lines: DiskANN consolidates the deletes, and in async and sync
// Build runs for the rows the updates touched and for the new queries that have arrived, and the new table is
// published. Until a round builds its row, a new query borrows from its MST parent, and after its first search it also
// starts from its provisional seam, the closest data vector its last probe computed.
//
//   updates --index G_Y --capacity C --data Y.fvecs --x-fvecs X.fvecs --x-nsg G_X.nsg --seams SEAM_TABLE
//           --radius RADIUS_FILE --eps THRESHOLD --mode async|sync|none [--updates FILE] --update-rate PER_SECOND
//           --query-rate PER_SECOND --duration-s SECONDS --join-threads J --update-threads U --maintain-threads M
//           --maintain-every LINES --out DIR [--L 256 --delta 0.05 --R 70 --vamana-L 100 --C 500 --alpha 1.2]
//
// G_Y is the DiskANN index written by vamana.cpp, SEAM_TABLE and RADIUS_FILE are written by build_seams, --L and
// --delta are those of build_seams, --R --vamana-L --C --alpha those of the index. A rate of 0 runs back to back;
// --duration-s 0 runs one pass over X and every update line. Writes to DIR: probes.tsv (one line per probe; state 0 =
// a row of Build, 1 = a new query before its first search, 2 = a new query with a provisional seam), updates.tsv,
// maintenance.tsv (one line per round) and summary.json.
#include "index.h"
#include "io.h"
#include "order.h"
#include "seams.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using DynamicIndex = diskann::Index<float, uint32_t, uint32_t>;

// The candidate queue of the greedy phase keeps its CANDIDATE_QUEUE_SIZE nearest candidates, and the greedy phase stops
// once its nearest candidate has not come closer for STALL_LIMIT steps in a row (as in join.cpp).
constexpr size_t CANDIDATE_QUEUE_SIZE = 256;
constexpr int STALL_LIMIT = 10;

// The seam table prints distances rounded; a comparison of a printed distance with eps allows for that.
constexpr double ROUNDING_TOLERANCE = 1e-9;

constexpr int32_t NO_LOCATION = -1;
constexpr int32_t NO_ROW = -1;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// `count` atomics, each holding `value`
template <class Value>
std::vector<std::atomic<Value>> atomic_vector(size_t count, Value value) {
    std::vector<std::atomic<Value>> slots(count);
    for (std::atomic<Value>& slot : slots) {
        slot.store(value, std::memory_order_relaxed);
    }
    return slots;
}

// DiskANN reserves tag 0, so the tag of a row of Y is its number + 1 (as vamana.cpp builds the index)
uint32_t tag_of(uint32_t row) {
    return row + 1;
}

// the distance of the join (l2_distance of io.h) on raw rows: each difference in float, the sum in double
double join_distance(const float* left, const float* right, size_t dimension) {
    double sum = 0.0;
    for (size_t component = 0; component < dimension; ++component) {
        const double difference = left[component] - right[component];
        sum += difference * difference;
    }
    return std::sqrt(sum);
}

// ----------------------------------------------------------------------------------------------- the seam table ----

// The row of one query: its seams in rank order, and the length and the radius r_L of the candidate list Build
// searched for it. pool_length 0: no candidate list, i.e. a new query that no round has built yet.
struct Row {
    std::vector<uint32_t> points;    // the seams, rows of Y
    std::vector<double> distances;   // their distances to the query, as the table prints them
    size_t pool_length = 0;
    double radius = std::nan("");
    bool patched = false;            // a round entered an inserted point into the seams: Build reruns for the row
};

struct SeamTable {
    std::vector<Row> rows;
};

// The seam table of build_seams (lines "x_id y_id dist rank", y_id -1 for a query without seams) and its radius file
// (lines "x pool_len r_L").
std::shared_ptr<SeamTable> read_seam_table(const std::string& seams_path, const std::string& radius_path,
                                           size_t points) {
    std::shared_ptr<SeamTable> table = std::make_shared<SeamTable>();
    std::ifstream seams_file(seams_path);
    if (!seams_file) {
        die("cannot open " + seams_path);
    }
    std::string line;
    if (!std::getline(seams_file, line) || line.rfind("x_id", 0) != 0) {
        die(seams_path + ": no header");
    }
    while (std::getline(seams_file, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream fields(line);
        long long query = -1;
        std::string point;
        std::string distance;
        long long rank = -1;
        if (!(fields >> query >> point >> distance >> rank)) {
            die(seams_path + ": bad line '" + line + "'");
        }
        if (query < 0 || size_t(query) > table->rows.size()) {
            die(seams_path + ": query ids not consecutive");
        }
        if (size_t(query) == table->rows.size()) {
            table->rows.emplace_back();
        } else if (size_t(query) + 1 != table->rows.size()) {
            die(seams_path + ": the lines of a query are not together");
        }
        if (point[0] == '-') {
            continue;
        }
        const uint32_t id = uint32_t(std::stoul(point));
        if (id >= points) {
            die(seams_path + ": y id out of range");
        }
        table->rows.back().points.push_back(id);
        table->rows.back().distances.push_back(std::stod(distance));
    }

    std::ifstream radius_file(radius_path);
    if (!radius_file) {
        die("cannot open " + radius_path);
    }
    if (!std::getline(radius_file, line) || line.rfind("x\tpool_len\tr_L", 0) != 0) {
        die(radius_path + ": no header");
    }
    size_t query = 0;
    while (std::getline(radius_file, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream fields(line);
        long long id = -1;
        long long length = -1;
        std::string value;
        if (!(fields >> id >> length >> value)) {
            die(radius_path + ": bad line '" + line + "'");
        }
        if (size_t(id) != query || query >= table->rows.size()) {
            die(radius_path + ": x not consecutive or beyond the seam table");
        }
        table->rows[query].pool_length = size_t(length);
        if (length > 0) {
            table->rows[query].radius = std::strtod(value.c_str(), nullptr);
        } else {
            table->rows[query].radius = std::nan("");
        }
        query++;
    }
    if (query != table->rows.size()) {
        die("the radius file and the seam table differ in size");
    }
    return table;
}

// --------------------------------------------------------------------------------------------- the live index ----

// the parameters of the dynamic Vamana index: degree R, list size L, candidates C, alpha, and the capacity
struct IndexRecipe {
    uint32_t degree = 70;
    uint32_t list_size = 100;
    uint32_t candidates = 500;
    float alpha = 1.2f;
    size_t capacity = 0;
};

diskann::IndexWriteParameters write_parameters(const IndexRecipe& recipe, uint32_t threads) {
    return diskann::IndexWriteParametersBuilder(recipe.list_size, recipe.degree)
        .with_max_occlusion_size(recipe.candidates)
        .with_alpha(recipe.alpha)
        .with_saturate_graph(false)
        .with_num_threads(threads)
        .build();
}

// an empty dynamic index with tags, one frozen start point and concurrent consolidation
std::unique_ptr<DynamicIndex> make_index(const IndexRecipe& recipe, size_t dimension, uint32_t scratch_threads) {
    std::shared_ptr<diskann::IndexWriteParameters> writing =
        std::make_shared<diskann::IndexWriteParameters>(write_parameters(recipe, scratch_threads));
    std::shared_ptr<diskann::IndexSearchParams> searching =
        std::make_shared<diskann::IndexSearchParams>(recipe.list_size, scratch_threads);
    const size_t frozen_points = 1;
    const bool dynamic_index = true;
    const bool enable_tags = true;
    const bool concurrent_consolidate = true;
    return std::make_unique<DynamicIndex>(diskann::Metric::L2, dimension, recipe.capacity, writing, searching,
                                          frozen_points, dynamic_index, enable_tags, concurrent_consolidate);
}

// ------------------------------------------------------------------------------------------- the update file ----

struct Update {
    char kind = 'i';   // 'i' insert or 'd' delete of a row of Y, 'q' arrival of a query
    uint32_t id = 0;   // the row of Y or the query
};

std::vector<Update> read_updates(const std::string& path, size_t points, size_t queries) {
    std::ifstream file(path);
    if (!file) {
        die("cannot open " + path);
    }
    std::vector<Update> updates;
    std::string kind;
    long long id = 0;
    while (file >> kind >> id) {
        size_t limit = points;
        if (kind == "q") {
            limit = queries;
        } else if (kind != "i" && kind != "d") {
            die(path + ": bad update line");
        }
        if (id < 0 || size_t(id) >= limit) {
            die(path + ": bad update line");
        }
        updates.push_back({kind[0], uint32_t(id)});
    }
    return updates;
}

// the rows of Y inserted and deleted by the update file, each in issue order
struct UpdatedRows {
    std::vector<uint32_t> inserted;
    std::vector<uint32_t> deleted;
};

UpdatedRows updated_rows(const std::vector<Update>& updates) {
    UpdatedRows rows;
    for (const Update& update : updates) {
        if (update.kind == 'i') {
            rows.inserted.push_back(update.id);
        }
        if (update.kind == 'd') {
            rows.deleted.push_back(update.id);
        }
    }
    return rows;
}

// ---------------------------------------------------------------------------------------------- the stream ----

enum Mode { ASYNC, SYNC, NONE };

// What every thread shares: the inputs, the live index of Y with the map between its locations and the rows of Y,
// the published seam table, the counters of the streams and the state of the new queries.
struct Stream {
    seams::Vectors data;                         // all rows of Y
    seams::Vectors queries;                      // all queries
    seams::Graph query_graph;                    // G_X
    std::vector<std::pair<int, int>> order;      // (parent, query) in probe order
    double eps = 0;
    Mode mode = ASYNC;
    size_t pool_width = 256;                     // --L of Build
    double delta = 0.05;
    IndexRecipe recipe;

    std::unique_ptr<DynamicIndex> index;
    uint32_t frozen = 0;                         // DiskANN's start: a frozen point that is not a row of Y
    std::atomic<int32_t> twin_row{NO_ROW};       // the row nearest to the frozen point: the entry point of the probes
    std::vector<std::atomic<int32_t>> row_at;              // location -> row, NO_ROW for the frozen point and free slots
    std::vector<std::atomic<int32_t>> location_of_row;     // row -> location, NO_LOCATION when not in the graph
    std::vector<std::atomic<uint8_t>> tombstone;           // row -> 1 once its delete is applied
    std::shared_ptr<SeamTable> table;            // read with std::atomic_load, replaced with std::atomic_store

    std::vector<Update> updates;
    double query_rate = 0;                       // probes per second, 0 = back to back
    double update_rate = 0;                      // update lines per second, 0 = back to back
    int64_t duration_us = 0;                     // 0 = one pass over X and every update line
    size_t maintain_every = 0;                   // applied lines per maintenance round, 0 = no rounds
    Clock::time_point start_time;
    std::atomic<size_t> next_probe{0};
    std::atomic<size_t> next_update{0};
    std::atomic<size_t> inserts_applied{0};
    std::atomic<size_t> deletes_applied{0};
    std::atomic<size_t> queries_applied{0};
    std::atomic<size_t> updates_applied{0};
    std::atomic<size_t> skipped_absent{0};       // probes not run: their query had not arrived yet
    std::atomic<bool> probes_done{false};
    std::atomic<bool> updates_done{false};

    // soft work sharing: per query, the closest row its probe computed, as (cycle << 32) | row; -1 = none yet
    std::vector<std::atomic<int64_t>> closest_visited;
    // the new queries: is_new for the queries of the "q" lines, arrived once the line is applied (the other queries
    // from the start), provisional = the provisional seam (a row of Y) or NO_ROW
    std::vector<uint8_t> is_new;
    std::vector<std::atomic<uint8_t>> arrived;
    std::vector<std::atomic<int32_t>> provisional;

    // sync: a round keeps the probes paused
    std::mutex pause_mutex;
    std::condition_variable pause_changed;
    bool probes_paused = false;
    int active_probes = 0;

    // per update line: when it was due and when it was applied, -1 until then
    std::vector<std::atomic<int64_t>> update_due_us;
    std::vector<std::atomic<int64_t>> update_done_us;
};

int64_t now_us(const Stream& stream) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - stream.start_time).count();
}

void wait_until_us(const Stream& stream, int64_t due) {
    while (true) {
        const int64_t now = now_us(stream);
        if (now >= due) {
            return;
        }
        const int64_t left = due - now;
        if (left > 200) {
            std::this_thread::sleep_for(std::chrono::microseconds(left - 100));
        }
    }
}

// the entry point of a probe without a live start: the twin, or the frozen point when the twin has left the graph
int32_t entry_location(const Stream& stream) {
    const int32_t twin = stream.twin_row.load(std::memory_order_acquire);
    if (twin != NO_ROW) {
        const int32_t location = stream.location_of_row[size_t(twin)].load(std::memory_order_acquire);
        if (location != NO_LOCATION) {
            return location;
        }
    }
    return int32_t(stream.frozen);
}

// The twin: the row nearest to the frozen point among the rows in the graph that are not deleted, the first in location
// order on ties. It is the entry point of the probes and of the snapshot that Build searches.
void refresh_twin(Stream& stream) {
    const float* frozen_vector = stream.index->vector_of(stream.frozen);
    double nearest = std::numeric_limits<double>::max();
    int32_t twin = NO_ROW;
    for (size_t location = 0; location < stream.row_at.size(); ++location) {
        const int32_t row = stream.row_at[location].load(std::memory_order_acquire);
        if (row == NO_ROW) {
            continue;
        }
        if (stream.location_of_row[size_t(row)].load(std::memory_order_acquire) != int32_t(location)) {
            continue;
        }
        if (stream.tombstone[size_t(row)].load(std::memory_order_acquire)) {
            continue;
        }
        const double distance = seams::l2_distance(frozen_vector, stream.data.row(size_t(row)), stream.data.dim);
        if (distance < nearest) {
            nearest = distance;
            twin = row;
        }
    }
    stream.twin_row.store(twin, std::memory_order_release);
}

// -------------------------------------------------------------------------------------------------- probes ----

enum RowState { BUILT = 0, COLD = 1, PROVISIONAL = 2 };

struct ProbeRecord {
    uint32_t query = 0;
    int32_t parent = -1;
    uint32_t cycle = 0;              // the pass over X
    int64_t due_us = 0;
    int64_t start_us = 0;
    int64_t finish_us = 0;
    uint32_t distances[3] = {};      // distance computations per phase (Phase of io.h)
    uint8_t dropped = 0;             // 1: the empty answer of early pruning
    uint8_t state = BUILT;
    uint32_t results_at = 0;         // the answer: results[results_at, results_at + results_count) of its worker
    uint32_t results_count = 0;
};

// the memory of one probe thread
struct JoinWorker {
    std::vector<uint32_t> stamp;     // per location: the epoch of the last probe that visited it
    uint32_t epoch = 0;
    std::vector<uint32_t> list;      // a list of neighbours
    std::vector<ProbeRecord> records;
    std::vector<uint32_t> results;   // the answers of its probes, one after the other

    explicit JoinWorker(size_t locations) : stamp(locations, 0) {}

    // the next probe: every location counts as not visited
    void next_epoch() {
        epoch++;
        if (epoch == 0) {
            std::fill(stamp.begin(), stamp.end(), 0);
            epoch = 1;
        }
    }
};

// One probe on the live graph: the visited locations, the candidate queue of the greedy phase, the locations found
// within eps and the closest row computed so far.
struct ProbeWalk {
    Stream& stream;
    JoinWorker& worker;
    ProbeRecord& record;
    const float* query_vector;
    std::multiset<std::pair<double, uint32_t>> candidates;   // (distance, location), nearest first
    std::vector<uint32_t> found;                              // the locations within eps
    int32_t closest_row = NO_ROW;
    double closest_distance = std::numeric_limits<double>::max();

    ProbeWalk(Stream& shared, JoinWorker& own_worker, ProbeRecord& own_record, const float* vector)
        : stream(shared), worker(own_worker), record(own_record), query_vector(vector) {}

    bool visited(uint32_t location) const {
        return worker.stamp[location] == worker.epoch;
    }

    // marks the location visited, computes its distance to the query, counts it in its phase and keeps the closest row
    double visit(uint32_t location, Phase phase) {
        worker.stamp[location] = worker.epoch;
        const double distance = join_distance(query_vector, stream.index->vector_of(location), stream.queries.dim);
        record.distances[phase]++;
        const int32_t row = stream.row_at[location].load(std::memory_order_relaxed);
        if (distance < closest_distance) {
            closest_distance = distance;
            closest_row = row;
        }
        return distance;
    }

    // visits the location and makes it a candidate of the greedy phase and, within eps, a start of the expansion
    double add_candidate(uint32_t location, Phase phase) {
        const double distance = visit(location, phase);
        candidates.emplace(distance, location);
        if (distance <= stream.eps) {
            found.push_back(location);
        }
        return distance;
    }

    // a row as a start: added unless it has left the graph or is visited already
    void add_row(int32_t row) {
        const int32_t location = stream.location_of_row[size_t(row)].load(std::memory_order_acquire);
        if (location != NO_LOCATION && !visited(uint32_t(location))) {
            add_candidate(uint32_t(location), SEED);
        }
    }
};

// soft work sharing: the closest row the probe of `query` computed in this pass over X, NO_ROW if none
int32_t closest_row_passed(const Stream& stream, int query, uint32_t cycle) {
    const int64_t packed = stream.closest_visited[size_t(query)].load(std::memory_order_acquire);
    if (packed < 0 || uint32_t(packed >> 32) != cycle) {
        return NO_ROW;
    }
    return int32_t(packed & 0xffffffff);
}

// the rank of the nearest seam of a row that is still in the graph, or the length of the row when none is
size_t first_live_seam(const Stream& stream, const Row& row) {
    size_t rank = 0;
    while (rank < row.points.size() &&
           stream.location_of_row[row.points[rank]].load(std::memory_order_acquire) == NO_LOCATION) {
        rank++;
    }
    return rank;
}

// A location offered as a start by the parent of a new query: visited once; within eps it starts the expansion, and
// the closest one offered is kept as the start of the greedy phase.
void offer_start(ProbeWalk& walk, uint32_t location, int32_t& start, double& start_distance) {
    if (walk.visited(location)) {
        return;
    }
    const double distance = walk.visit(location, SEED);
    if (distance <= walk.stream.eps) {
        walk.found.push_back(location);
    }
    if (distance < start_distance) {
        start_distance = distance;
        start = int32_t(location);
    }
}

// A new query borrows from its MST parent: the parent's row, or the provisional seam of a parent that is new as well;
// false when the parent has neither. With the parent's row: when the parent's nearest live seam lies beyond
// eps + d(query, parent), the answer is empty (record.dropped). Otherwise the parent's closest row of this pass, the
// parent's seams that can lie within eps of the query, the own provisional seam and the entry point are visited: the
// ones within eps start the expansion, the closest one starts the greedy phase. d(query, parent) and the distance of a
// provisional seam to the parent are not counted.
bool borrow_from_parent(ProbeWalk& walk, const Row& parent_row, int parent, int32_t provisional_seam) {
    Stream& stream = walk.stream;
    const bool parent_built = !parent_row.points.empty();
    int32_t parent_seam = NO_ROW;
    if (!parent_built) {
        parent_seam = stream.provisional[size_t(parent)].load(std::memory_order_acquire);
        if (parent_seam == NO_ROW) {
            return false;
        }
    }
    const float* parent_vector = stream.queries.row(size_t(parent));
    const size_t dimension = stream.queries.dim;
    const double gap = join_distance(walk.query_vector, parent_vector, dimension);
    const int32_t passed_row = closest_row_passed(stream, parent, walk.record.cycle);
    const double eps = stream.eps;

    if (parent_built) {
        const size_t anchor = first_live_seam(stream, parent_row);
        if (anchor < parent_row.points.size() && parent_row.distances[anchor] - gap > eps + ROUNDING_TOLERANCE) {
            walk.closest_row = int32_t(parent_row.points[anchor]);
            if (passed_row != NO_ROW) {
                walk.closest_row = passed_row;
            }
            walk.record.dropped = 1;
            return true;
        }
    }

    std::vector<int32_t> start_rows;
    if (passed_row != NO_ROW) {
        start_rows.push_back(passed_row);
    }
    if (parent_built) {
        for (size_t rank = 0; rank < parent_row.points.size(); ++rank) {
            if (parent_row.distances[rank] <= eps + gap + ROUNDING_TOLERANCE) {
                start_rows.push_back(int32_t(parent_row.points[rank]));
            }
        }
    } else {
        const double seam_distance = join_distance(parent_vector, stream.data.row(size_t(parent_seam)), dimension);
        if (seam_distance <= eps + gap + ROUNDING_TOLERANCE) {
            start_rows.push_back(parent_seam);
        }
    }
    if (provisional_seam != NO_ROW) {
        start_rows.push_back(provisional_seam);
    }
    int32_t start = NO_LOCATION;
    double start_distance = std::numeric_limits<double>::max();
    for (const int32_t row : start_rows) {
        const int32_t location = stream.location_of_row[size_t(row)].load(std::memory_order_acquire);
        if (location != NO_LOCATION) {
            offer_start(walk, uint32_t(location), start, start_distance);
        }
    }
    offer_start(walk, uint32_t(entry_location(stream)), start, start_distance);
    if (start != NO_LOCATION) {
        walk.candidates.emplace(start_distance, uint32_t(start));
    }
    return true;
}

// The greedy phase: expands the nearest candidate until it lies within eps or has not come closer for STALL_LIMIT
// steps. Every unvisited neighbour becomes a candidate; the frozen point is never a neighbour.
void greedy_phase(ProbeWalk& walk) {
    Stream& stream = walk.stream;
    double nearest_so_far = std::numeric_limits<double>::max();
    int steps_without_progress = 0;
    while (!walk.candidates.empty()) {
        const std::pair<double, uint32_t> nearest = *walk.candidates.begin();
        if (nearest.first <= stream.eps) {
            break;
        }
        if (nearest.first < nearest_so_far) {
            nearest_so_far = nearest.first;
            steps_without_progress = 0;
        } else {
            steps_without_progress++;
            if (steps_without_progress >= STALL_LIMIT) {
                break;
            }
        }
        walk.candidates.erase(walk.candidates.begin());
        stream.index->neighbours_of(nearest.second, walk.worker.list);
        for (const uint32_t neighbour : walk.worker.list) {
            if (neighbour == stream.frozen || walk.visited(neighbour)) {
                continue;
            }
            walk.add_candidate(neighbour, GREEDY);
        }
        while (walk.candidates.size() > CANDIDATE_QUEUE_SIZE) {
            walk.candidates.erase(std::prev(walk.candidates.end()));
        }
    }
}

// the expansion phase: breadth first from the locations found within eps, through the locations within eps
void expansion_phase(ProbeWalk& walk) {
    Stream& stream = walk.stream;
    std::queue<uint32_t> frontier;
    for (const uint32_t location : walk.found) {
        frontier.push(location);
    }
    while (!frontier.empty()) {
        const uint32_t location = frontier.front();
        frontier.pop();
        stream.index->neighbours_of(location, walk.worker.list);
        for (const uint32_t neighbour : walk.worker.list) {
            if (neighbour == stream.frozen || walk.visited(neighbour)) {
                continue;
            }
            const double distance = walk.visit(neighbour, EXPAND);
            if (distance <= stream.eps) {
                frontier.push(neighbour);
                walk.found.push_back(neighbour);
            }
        }
    }
}

// The end of a probe: its closest row goes to its tree children (soft work sharing) and, for a new query that searched,
// becomes its provisional seam; the probe is logged.
void finish_probe(Stream& stream, JoinWorker& worker, ProbeRecord& record, int32_t closest_row, bool keeps_provisional) {
    if (closest_row != NO_ROW) {
        const int64_t packed = (int64_t(record.cycle) << 32) | int64_t(uint32_t(closest_row));
        stream.closest_visited[record.query].store(packed, std::memory_order_release);
        if (keeps_provisional) {
            stream.provisional[record.query].store(closest_row, std::memory_order_release);
        }
    }
    record.finish_us = now_us(stream);
    worker.records.push_back(record);
}

// Probe `index` of the stream: query order[index mod |X|]. A query with seams starts from its seams that are still in
// the graph, in rank order up to the first one beyond eps once one within eps is found, and gets the empty answer when
// its nearest live seam lies beyond eps. A query without seams starts from the entry point. Then the closest row of
// the parent's probe, the entry point when no start is left, the greedy phase and the expansion phase. The answer:
// the rows within eps that are not deleted.
void run_probe(Stream& stream, JoinWorker& worker, size_t index, int64_t due) {
    const size_t query_count = stream.order.size();
    const int parent = stream.order[index % query_count].first;
    const int query = stream.order[index % query_count].second;
    if (!stream.arrived[size_t(query)].load(std::memory_order_acquire)) {
        stream.skipped_absent.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    ProbeRecord record;
    record.query = uint32_t(query);
    record.parent = parent;
    record.cycle = uint32_t(index / query_count);
    record.due_us = due;
    record.start_us = now_us(stream);
    record.results_at = uint32_t(worker.results.size());

    const std::shared_ptr<SeamTable> table = std::atomic_load(&stream.table);
    const Row& row = table->rows[size_t(query)];
    const bool has_seams = !row.points.empty();
    const bool new_row = stream.is_new[size_t(query)] && row.pool_length == 0;
    int32_t provisional_seam = NO_ROW;
    if (new_row) {
        provisional_seam = stream.provisional[size_t(query)].load(std::memory_order_acquire);
        if (provisional_seam == NO_ROW) {
            record.state = COLD;
        } else {
            record.state = PROVISIONAL;
        }
    }
    worker.next_epoch();
    ProbeWalk walk(stream, worker, record, stream.queries.row(size_t(query)));
    const double eps = stream.eps;

    bool borrowed = false;
    if (new_row && parent >= 0) {
        borrowed = borrow_from_parent(walk, table->rows[size_t(parent)], parent, provisional_seam);
        if (record.dropped) {
            finish_probe(stream, worker, record, walk.closest_row, false);
            return;
        }
    }

    if (has_seams) {
        const size_t anchor = first_live_seam(stream, row);
        if (anchor < row.points.size() && row.distances[anchor] > eps + ROUNDING_TOLERANCE) {
            record.dropped = 1;
            finish_probe(stream, worker, record, int32_t(row.points[anchor]), false);
            return;
        }
    }

    if (!borrowed) {
        if (has_seams) {
            for (size_t rank = 0; rank < row.points.size(); ++rank) {
                const int32_t location = stream.location_of_row[row.points[rank]].load(std::memory_order_acquire);
                if (location == NO_LOCATION || walk.visited(uint32_t(location))) {
                    continue;
                }
                const double distance = walk.add_candidate(uint32_t(location), SEED);
                if (distance > eps && !walk.found.empty()) {
                    break;
                }
            }
        } else {
            walk.add_candidate(uint32_t(entry_location(stream)), SEED);
            if (provisional_seam != NO_ROW) {
                walk.add_row(provisional_seam);
            }
        }
        if (parent >= 0) {
            const int32_t passed_row = closest_row_passed(stream, parent, record.cycle);
            if (passed_row != NO_ROW) {
                walk.add_row(passed_row);
            }
        }
    }

    if (walk.candidates.empty() && walk.found.empty()) {
        const uint32_t entry = uint32_t(entry_location(stream));
        if (!walk.visited(entry)) {
            walk.add_candidate(entry, SEED);
        }
    }

    greedy_phase(walk);
    expansion_phase(walk);

    for (const uint32_t location : walk.found) {
        const int32_t point = stream.row_at[location].load(std::memory_order_acquire);
        if (point == NO_ROW || stream.tombstone[size_t(point)].load(std::memory_order_acquire)) {
            continue;
        }
        worker.results.push_back(uint32_t(point));
    }
    record.results_count = uint32_t(worker.results.size()) - record.results_at;
    finish_probe(stream, worker, record, walk.closest_row, new_row);
}

void join_thread(Stream& stream, JoinWorker& worker) {
    const bool synchronous = stream.mode == SYNC;
    while (true) {
        const size_t index = stream.next_probe.fetch_add(1);
        int64_t due = 0;
        if (stream.query_rate > 0) {
            due = int64_t(double(index) * 1e6 / stream.query_rate);
        } else {
            due = now_us(stream);
        }
        bool over = index >= stream.order.size();
        if (stream.duration_us > 0) {
            over = due >= stream.duration_us;
        }
        if (over) {
            break;
        }
        if (stream.query_rate > 0) {
            wait_until_us(stream, due);
        }
        if (synchronous) {
            std::unique_lock<std::mutex> lock(stream.pause_mutex);
            while (stream.probes_paused) {
                stream.pause_changed.wait(lock);
            }
            stream.active_probes++;
        }
        run_probe(stream, worker, index, due);
        if (synchronous) {
            std::lock_guard<std::mutex> lock(stream.pause_mutex);
            stream.active_probes--;
            if (stream.active_probes == 0) {
                stream.pause_changed.notify_all();
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------- updates ----

void update_thread(Stream& stream) {
    while (true) {
        const size_t index = stream.next_update.fetch_add(1);
        if (index >= stream.updates.size()) {
            break;
        }
        int64_t due = 0;
        if (stream.update_rate > 0) {
            due = int64_t(double(index) * 1e6 / stream.update_rate);
            if (stream.duration_us > 0 && due >= stream.duration_us) {
                break;
            }
            wait_until_us(stream, due);
        } else {
            due = now_us(stream);
        }
        stream.update_due_us[index].store(due, std::memory_order_relaxed);
        const Update update = stream.updates[index];
        if (update.kind == 'i') {
            if (stream.index->insert_point(stream.data.row(update.id), tag_of(update.id)) != 0) {
                die("insert of row " + std::to_string(update.id) + " failed");
            }
            const int64_t location = stream.index->location_of(tag_of(update.id));
            if (location < 0) {
                die("row " + std::to_string(update.id) + " has no location after its insert");
            }
            stream.row_at[size_t(location)].store(int32_t(update.id), std::memory_order_release);
            stream.location_of_row[update.id].store(int32_t(location), std::memory_order_release);
            stream.inserts_applied.fetch_add(1, std::memory_order_release);
        } else if (update.kind == 'q') {
            stream.arrived[update.id].store(1, std::memory_order_release);
            stream.queries_applied.fetch_add(1, std::memory_order_release);
        } else {
            stream.tombstone[update.id].store(1, std::memory_order_release);
            if (stream.index->lazy_delete(tag_of(update.id)) != 0) {
                die("delete of row " + std::to_string(update.id) + ": not in the index");
            }
            stream.deletes_applied.fetch_add(1, std::memory_order_release);
        }
        stream.update_done_us[index].store(now_us(stream), std::memory_order_release);
        stream.updates_applied.fetch_add(1, std::memory_order_release);
    }
}

// ---------------------------------------------------------------------------------------------- maintenance ----

struct RoundRecord {
    int round = 0;
    int64_t start_us = 0;
    int64_t published_us = 0;
    size_t inserts_cut = 0;     // inserts and deletes applied when the round started
    size_t deletes_cut = 0;
    size_t consolidated = 0;    // deleted rows that left the graph in this round
    size_t stale = 0;           // rows an insert entered
    size_t hit = 0;             // rows whose candidate list or seams held a deleted row
    size_t built = 0;           // rows Build ran for: stale, hit and new
    size_t pending = 0;         // new queries that had arrived without a row of Build
};

// sync: waits until no probe runs and keeps the next probes waiting until resume_probes
void pause_probes(Stream& stream) {
    std::unique_lock<std::mutex> lock(stream.pause_mutex);
    stream.probes_paused = true;
    while (stream.active_probes != 0) {
        stream.pause_changed.wait(lock);
    }
}

void resume_probes(Stream& stream) {
    std::unique_lock<std::mutex> lock(stream.pause_mutex);
    stream.probes_paused = false;
    lock.unlock();
    stream.pause_changed.notify_all();
}

// the new queries that have arrived and have no row of Build yet
std::vector<uint32_t> waiting_new_rows(const Stream& stream) {
    const std::shared_ptr<SeamTable> table = std::atomic_load(&stream.table);
    std::vector<uint32_t> rows;
    for (size_t query = 0; query < stream.queries.count; ++query) {
        if (!stream.is_new[query] || table->rows[query].pool_length != 0) {
            continue;
        }
        if (stream.arrived[query].load(std::memory_order_acquire)) {
            rows.push_back(uint32_t(query));
        }
    }
    return rows;
}

// DiskANN removes the first deletes_cut deleted rows from the graph; returns how many of them were still in it
size_t consolidate(Stream& stream, size_t deletes_cut, const UpdatedRows& updated, int threads) {
    if (deletes_cut > 0) {
        const diskann::consolidation_report report =
            stream.index->consolidate_deletes(write_parameters(stream.recipe, uint32_t(threads)));
        if (report._status != diskann::consolidation_report::status_code::SUCCESS) {
            die("consolidate_deletes failed");
        }
    }
    size_t consolidated = 0;
    for (size_t position = 0; position < deletes_cut; ++position) {
        const uint32_t row = updated.deleted[position];
        if (stream.location_of_row[row].load(std::memory_order_acquire) != NO_LOCATION) {
            stream.location_of_row[row].store(NO_LOCATION, std::memory_order_release);
            consolidated++;
        }
    }
    refresh_twin(stream);
    return consolidated;
}

// Near(y) of every point y: the queries with a candidate list and d(x, y) < r_L(x) (<= r_L(x) with including_radius),
// found by a top-k search of y on G_X, with their distances as the table prints them
struct NearQueries {
    std::vector<std::vector<uint32_t>> queries;     // per point
    std::vector<std::vector<double>> distances;     // per point, d(x, y) of each of its queries
};

NearQueries near_queries(const Stream& stream, const SeamTable& table, const std::vector<uint32_t>& points,
                         bool including_radius, int threads) {
    NearQueries near;
    near.queries.resize(points.size());
    near.distances.resize(points.size());
    seams::Workers<seams::PoolSearch> workers(threads, stream.query_graph, stream.queries);
    seams::for_each_query(workers, points.size(), [&](seams::PoolSearch& search, size_t position) {
        const std::vector<seams::PoolSearch::Candidate>& pool =
            search.run(stream.data.row(points[position]), stream.pool_width);
        for (const seams::PoolSearch::Candidate& candidate : pool) {
            const Row& row = table.rows[candidate.id];
            if (row.pool_length == 0) {
                continue;
            }
            const double distance = seams::as_printed(candidate.dist);
            bool inside = distance < row.radius;
            if (including_radius) {
                inside = distance <= row.radius;
            }
            if (!inside) {
                continue;
            }
            near.queries[position].push_back(candidate.id);
            near.distances[position].push_back(distance);
        }
    });
    return near;
}

// patch: every inserted point enters, by distance, the seams of the rows it falls in; these rows are stale
void patch_rows(const Stream& stream, SeamTable& table, const std::vector<uint32_t>& inserted, int threads) {
    const NearQueries near = near_queries(stream, table, inserted, false, threads);
    for (size_t position = 0; position < inserted.size(); ++position) {
        const uint32_t point = inserted[position];
        for (size_t member = 0; member < near.queries[position].size(); ++member) {
            Row& row = table.rows[near.queries[position][member]];
            const double distance = near.distances[position][member];
            if (std::find(row.points.begin(), row.points.end(), point) != row.points.end()) {
                continue;
            }
            const std::vector<double>::iterator slot =
                std::upper_bound(row.distances.begin(), row.distances.end(), distance);
            const std::ptrdiff_t rank = slot - row.distances.begin();
            row.points.insert(row.points.begin() + rank, point);
            row.distances.insert(row.distances.begin() + rank, distance);
            row.patched = true;
        }
    }
}

// hit: the rows whose candidate list or seams hold a deleted point, ascending
std::vector<uint32_t> hit_rows(const Stream& stream, const SeamTable& table, const std::vector<uint32_t>& deleted,
                               int threads) {
    std::unordered_map<uint32_t, std::vector<uint32_t>> owners;   // a seam -> the queries it is a seam of
    for (size_t query = 0; query < table.rows.size(); ++query) {
        for (const uint32_t point : table.rows[query].points) {
            owners[point].push_back(uint32_t(query));
        }
    }
    const NearQueries near = near_queries(stream, table, deleted, true, threads);
    std::vector<uint32_t> rows;
    for (size_t position = 0; position < deleted.size(); ++position) {
        const std::vector<uint32_t>& near_point = near.queries[position];
        rows.insert(rows.end(), near_point.begin(), near_point.end());
        const std::unordered_map<uint32_t, std::vector<uint32_t>>::const_iterator owner = owners.find(deleted[position]);
        if (owner == owners.end()) {
            continue;
        }
        for (const uint32_t query : owner->second) {
            if (std::find(near_point.begin(), near_point.end(), query) == near_point.end()) {
                rows.push_back(query);
            }
        }
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

// one thread of the snapshot: takes chunks of locations and copies their lists as lists of rows
void copy_lists(Stream& stream, std::vector<std::vector<uint32_t>>& lists, std::atomic<size_t>& next_location) {
    const size_t chunk = 4096;
    const size_t locations = stream.row_at.size();
    std::vector<uint32_t> list;
    while (true) {
        const size_t from = next_location.fetch_add(chunk);
        if (from >= locations) {
            return;
        }
        const size_t to = std::min(locations, from + chunk);
        for (size_t location = from; location < to; ++location) {
            const int32_t row = stream.row_at[location].load(std::memory_order_acquire);
            if (row == NO_ROW) {
                continue;
            }
            if (stream.location_of_row[size_t(row)].load(std::memory_order_acquire) != int32_t(location)) {
                continue;
            }
            stream.index->neighbours_of(uint32_t(location), list);
            std::vector<uint32_t>& target = lists[size_t(row)];
            target.clear();
            for (const uint32_t neighbour : list) {
                if (neighbour == stream.frozen) {
                    continue;
                }
                const int32_t other = stream.row_at[neighbour].load(std::memory_order_acquire);
                if (other != NO_ROW) {
                    target.push_back(uint32_t(other));
                }
            }
        }
    }
}

// The live graph over the rows of Y, for Build: every location that holds a row, its list copied under its lock,
// without the edges into the frozen point and into locations without a row; its entry is the row of the entry point.
seams::Graph snapshot_graph(Stream& stream, int threads) {
    const size_t points = stream.data.count;
    std::vector<std::vector<uint32_t>> lists(points);
    std::atomic<size_t> next_location{0};
    std::vector<std::thread> pool;
    for (int thread = 0; thread < threads; ++thread) {
        pool.emplace_back(copy_lists, std::ref(stream), std::ref(lists), std::ref(next_location));
    }
    for (std::thread& thread : pool) {
        thread.join();
    }
    seams::Graph graph;
    graph.offsets.reserve(points + 1);
    graph.offsets.push_back(0);
    for (size_t row = 0; row < points; ++row) {
        graph.neighbours.insert(graph.neighbours.end(), lists[row].begin(), lists[row].end());
        graph.offsets.push_back(graph.neighbours.size());
    }
    const int32_t entry_row = stream.row_at[size_t(entry_location(stream))].load(std::memory_order_acquire);
    graph.entry = 0;
    if (entry_row != NO_ROW) {
        graph.entry = uint32_t(entry_row);
    }
    return graph;
}

// Build for the given rows on the snapshot: a top-k search per query, then Profile and Select; the rows get their new
// seams, candidate-list length and r_L
void build_rows(const Stream& stream, SeamTable& table, const seams::Graph& graph, const std::vector<uint32_t>& rows,
                int threads) {
    const size_t dimension = stream.queries.dim;
    seams::Vectors row_queries;
    row_queries.count = rows.size();
    row_queries.dim = dimension;
    row_queries.values.resize(rows.size() * dimension);
    for (size_t position = 0; position < rows.size(); ++position) {
        const float* vector = stream.queries.row(rows[position]);
        std::copy(vector, vector + dimension, row_queries.values.begin() + std::ptrdiff_t(position * dimension));
    }
    uint64_t pool_distances = 0;
    const seams::Pools pools =
        seams::search_pools(graph, stream.data, row_queries, stream.pool_width, threads, pool_distances);
    std::vector<std::vector<int32_t>> seam_ranks(rows.size());
    seams::Workers<seams::BuildWorker> workers(threads, graph);
    seams::for_each_query(workers, rows.size(), [&](seams::BuildWorker& worker, size_t position) {
        seam_ranks[position] = worker.select(pools, position, stream.delta);
    });
    for (size_t position = 0; position < rows.size(); ++position) {
        Row& row = table.rows[rows[position]];
        const size_t length = pools.length(position);
        row.points.clear();
        row.distances.clear();
        row.patched = false;
        row.pool_length = length;
        row.radius = std::nan("");
        if (length > 0) {
            row.radius = pools.dist_of(position)[length - 1];
        }
        for (const int32_t rank : seam_ranks[position]) {
            row.points.push_back(pools.ids_of(position)[size_t(rank)]);
            row.distances.push_back(pools.dist_of(position)[size_t(rank)]);
        }
    }
}

// One round: consolidate; in async and sync also patch, hit, snapshot, Build of the stale, hit and new rows, publish.
// The inserts and deletes before inserts_handled / deletes_handled were handled by earlier rounds.
RoundRecord maintenance_round(Stream& stream, int round, size_t& inserts_handled, size_t& deletes_handled,
                              const UpdatedRows& updated, int threads) {
    RoundRecord record;
    record.round = round;
    record.start_us = now_us(stream);
    const bool maintained = stream.mode != NONE;
    if (stream.mode == SYNC) {
        pause_probes(stream);
    }
    record.inserts_cut = stream.inserts_applied.load(std::memory_order_acquire);
    record.deletes_cut = stream.deletes_applied.load(std::memory_order_acquire);
    std::vector<uint32_t> new_rows;
    if (maintained) {
        new_rows = waiting_new_rows(stream);
    }
    record.consolidated = consolidate(stream, record.deletes_cut, updated, threads);

    if (maintained) {
        const std::shared_ptr<SeamTable> current = std::atomic_load(&stream.table);
        std::shared_ptr<SeamTable> fresh = std::make_shared<SeamTable>(*current);
        const std::vector<uint32_t> inserted(updated.inserted.begin() + std::ptrdiff_t(inserts_handled),
                                             updated.inserted.begin() + std::ptrdiff_t(record.inserts_cut));
        patch_rows(stream, *fresh, inserted, threads);
        const std::vector<uint32_t> deleted(updated.deleted.begin() + std::ptrdiff_t(deletes_handled),
                                            updated.deleted.begin() + std::ptrdiff_t(record.deletes_cut));
        const std::vector<uint32_t> hit = hit_rows(stream, *fresh, deleted, threads);
        const seams::Graph graph = snapshot_graph(stream, threads);
        std::vector<uint32_t> stale;
        for (size_t query = 0; query < fresh->rows.size(); ++query) {
            if (fresh->rows[query].patched) {
                stale.push_back(uint32_t(query));
            }
        }
        std::vector<uint32_t> touched;
        std::set_union(stale.begin(), stale.end(), hit.begin(), hit.end(), std::back_inserter(touched));
        std::vector<uint32_t> rebuild;
        std::set_union(touched.begin(), touched.end(), new_rows.begin(), new_rows.end(), std::back_inserter(rebuild));
        if (!rebuild.empty()) {
            build_rows(stream, *fresh, graph, rebuild, threads);
        }
        record.stale = stale.size();
        record.hit = hit.size();
        record.pending = new_rows.size();
        record.built = rebuild.size();
        std::atomic_store(&stream.table, fresh);
        inserts_handled = record.inserts_cut;
        deletes_handled = record.deletes_cut;
    }
    record.published_us = now_us(stream);
    if (stream.mode == SYNC) {
        resume_probes(stream);
    }
    return record;
}

void maintenance_thread(Stream& stream, const UpdatedRows& updated, int threads, std::vector<RoundRecord>& rounds) {
    size_t inserts_handled = 0;
    size_t deletes_handled = 0;
    size_t last_cut = 0;
    int round = 0;
    while (true) {
        const bool over = stream.probes_done.load() && stream.updates_done.load();
        const size_t applied = stream.updates_applied.load(std::memory_order_acquire);
        if (stream.maintain_every > 0 && applied - last_cut >= stream.maintain_every) {
            last_cut = applied;
            round++;
            rounds.push_back(maintenance_round(stream, round, inserts_handled, deletes_handled, updated, threads));
            continue;
        }
        if (over) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

// -------------------------------------------------------------------------------------------------- outputs ----

// the MST order of X (order.h): G_X weighted by the distances of the join, the twin as the entry point
std::vector<std::pair<int, int>> probe_order(const Stream& stream, const std::string& queries_path,
                                             const std::string& query_graph_path) {
    const NsgLists query_lists = read_nsg_lists(query_graph_path);
    const Vectors queries = read_vectors(queries_path, query_lists.neighbours.size());
    const Graph query_graph = build_graph(query_lists.neighbours, queries, true);
    const float* twin = stream.data.row(size_t(stream.twin_row.load()));
    const std::vector<float> entry_vector(twin, twin + stream.data.dim);
    return mst_order(query_graph, queries, entry_vector);
}

// a logged probe and the results of the worker that ran it
struct LoggedProbe {
    const ProbeRecord* record;
    const std::vector<uint32_t>* results;
};

bool runs_earlier(const LoggedProbe& left, const LoggedProbe& right) {
    if (left.record->due_us != right.record->due_us) {
        return left.record->due_us < right.record->due_us;
    }
    return left.record->start_us < right.record->start_us;
}

void write_probes(const std::string& path, const std::vector<std::unique_ptr<JoinWorker>>& workers) {
    std::vector<LoggedProbe> probes;
    for (const std::unique_ptr<JoinWorker>& worker : workers) {
        for (const ProbeRecord& record : worker->records) {
            probes.push_back({&record, &worker->results});
        }
    }
    std::sort(probes.begin(), probes.end(), runs_earlier);
    std::ofstream file(path);
    file << "x\tparent\tcycle\tdue_us\tstart_us\tfinish_us\tdropped\tndc_seed\tndc_greedy\tndc_expand\tstate\tresults\n";
    for (const LoggedProbe& probe : probes) {
        const ProbeRecord& record = *probe.record;
        file << record.query << '\t' << record.parent << '\t' << record.cycle << '\t' << record.due_us << '\t'
             << record.start_us << '\t' << record.finish_us << '\t' << int(record.dropped) << '\t'
             << record.distances[SEED] << '\t' << record.distances[GREEDY] << '\t' << record.distances[EXPAND] << '\t'
             << int(record.state) << '\t';
        if (record.results_count == 0) {
            file << '-';
        }
        for (uint32_t position = 0; position < record.results_count; ++position) {
            if (position > 0) {
                file << ',';
            }
            file << (*probe.results)[record.results_at + position];
        }
        file << '\n';
    }
}

void write_updates(const std::string& path, const Stream& stream) {
    std::ofstream file(path);
    file << "kind\trow\tdue_us\tdone_us\n";
    for (size_t index = 0; index < stream.updates.size(); ++index) {
        const int64_t done = stream.update_done_us[index].load();
        if (done < 0) {
            break;
        }
        file << stream.updates[index].kind << '\t' << stream.updates[index].id << '\t'
             << stream.update_due_us[index].load() << '\t' << done << '\n';
    }
}

void write_rounds(const std::string& path, const std::vector<RoundRecord>& rounds) {
    std::ofstream file(path);
    file << "round\tstart_us\tpublished_us\tinserts_cut\tdeletes_cut\tconsolidated\tstale\thit\tbuilt\tpending\n";
    for (const RoundRecord& record : rounds) {
        file << record.round << '\t' << record.start_us << '\t' << record.published_us << '\t' << record.inserts_cut
             << '\t' << record.deletes_cut << '\t' << record.consolidated << '\t' << record.stale << '\t' << record.hit
             << '\t' << record.built << '\t' << record.pending << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args args(argc, argv);
    Stream stream;
    const std::string index_prefix = args.text("index");
    const std::string data_path = args.text("data");
    const std::string queries_path = args.text("x-fvecs");
    const std::string query_graph_path = args.text("x-nsg");
    const std::string seams_path = args.text("seams");
    const std::string radius_path = args.text("radius");
    stream.eps = args.number("eps");
    stream.pool_width = size_t(args.number("L", "256"));
    stream.delta = args.number("delta", "0.05");
    stream.recipe.degree = uint32_t(args.number("R", "70"));
    stream.recipe.list_size = uint32_t(args.number("vamana-L", "100"));
    stream.recipe.candidates = uint32_t(args.number("C", "500"));
    stream.recipe.alpha = float(args.number("alpha", "1.2"));
    stream.recipe.capacity = size_t(args.number("capacity"));
    const std::string mode = args.text("mode");
    if (mode == "async") {
        stream.mode = ASYNC;
    } else if (mode == "sync") {
        stream.mode = SYNC;
    } else if (mode == "none") {
        stream.mode = NONE;
    } else {
        die("--mode must be async, sync or none");
    }
    const std::string updates_path = args.text("updates", "");
    stream.update_rate = args.number("update-rate", "0");
    stream.query_rate = args.number("query-rate", "0");
    stream.duration_us = int64_t(args.number("duration-s", "0") * 1e6);
    const int join_threads = int(args.number("join-threads", "1"));
    const int update_threads = int(args.number("update-threads", "1"));
    const int maintain_threads = int(args.number("maintain-threads", "1"));
    stream.maintain_every = size_t(args.number("maintain-every", "0"));
    const std::string out = args.text("out");
    args.check();
    if (join_threads < 1 || update_threads < 1 || maintain_threads < 1) {
        die("thread counts must be positive");
    }
    std::filesystem::create_directories(out);

    const Clock::time_point load_started = Clock::now();
    stream.data = seams::read_fvecs(data_path);
    stream.queries = seams::read_fvecs(queries_path);
    stream.query_graph = seams::read_nsg(query_graph_path);
    if (stream.query_graph.size() != stream.queries.count) {
        die("x-fvecs and x-nsg differ in size");
    }
    if (stream.queries.dim != stream.data.dim) {
        die("queries and data have different dimensionality");
    }
    const size_t points = stream.data.count;
    stream.table = read_seam_table(seams_path, radius_path, points);
    if (stream.table->rows.size() != stream.queries.count) {
        die("the seam table and X differ in size");
    }
    if (!updates_path.empty()) {
        stream.updates = read_updates(updates_path, points, stream.queries.count);
    }
    const UpdatedRows updated = updated_rows(stream.updates);

    // the queries of the "q" lines arrive when their line is applied; their rows of the seam table are dropped
    stream.is_new.assign(stream.queries.count, 0);
    stream.arrived = atomic_vector<uint8_t>(stream.queries.count, 1);
    stream.provisional = atomic_vector<int32_t>(stream.queries.count, NO_ROW);
    size_t arriving = 0;
    for (const Update& update : stream.updates) {
        if (update.kind != 'q') {
            continue;
        }
        if (stream.is_new[update.id]) {
            die("query " + std::to_string(update.id) + " arrives twice");
        }
        stream.is_new[update.id] = 1;
        stream.arrived[update.id].store(0, std::memory_order_relaxed);
        stream.table->rows[update.id] = Row();
        arriving++;
    }

    const uint32_t scratch_threads = uint32_t(update_threads + maintain_threads + 2);
    stream.index = make_index(stream.recipe, stream.data.dim, scratch_threads);
    stream.index->load(index_prefix.c_str(), scratch_threads, stream.recipe.list_size);
    if (stream.index->get_num_points() + updated.inserted.size() >= stream.recipe.capacity) {
        die("--capacity must exceed the points of the index plus the inserts (DiskANN needs a free slot)");
    }
    stream.frozen = stream.index->start_location();
    const size_t locations = stream.recipe.capacity + 1;
    if (stream.frozen >= locations) {
        die("the start location is beyond the capacity");
    }
    stream.row_at = atomic_vector<int32_t>(locations, NO_ROW);
    stream.location_of_row = atomic_vector<int32_t>(points, NO_LOCATION);
    stream.tombstone = atomic_vector<uint8_t>(points, 0);
    size_t indexed = 0;
    for (size_t row = 0; row < points; ++row) {
        const int64_t location = stream.index->location_of(tag_of(uint32_t(row)));
        if (location < 0) {
            continue;
        }
        if (size_t(location) >= locations) {
            die("a location beyond the capacity");
        }
        stream.row_at[size_t(location)].store(int32_t(row), std::memory_order_relaxed);
        stream.location_of_row[row].store(int32_t(location), std::memory_order_relaxed);
        indexed++;
    }
    refresh_twin(stream);
    stream.order = probe_order(stream, queries_path, query_graph_path);
    if (stream.order.size() != stream.queries.count) {
        die("the order does not cover X");
    }
    stream.closest_visited = atomic_vector<int64_t>(stream.queries.count, -1);
    stream.update_due_us = atomic_vector<int64_t>(stream.updates.size(), -1);
    stream.update_done_us = atomic_vector<int64_t>(stream.updates.size(), -1);
    const double load_seconds = seconds_since(load_started);
    std::cerr << "updates: " << points << " rows of Y, " << indexed << " indexed, " << stream.queries.count
              << " queries (" << arriving << " arriving), " << stream.updates.size() << " update lines, mode " << mode
              << ", threads J " << join_threads << " U " << update_threads << " M " << maintain_threads
              << ", loaded in " << load_seconds << " s\n";

    std::vector<std::unique_ptr<JoinWorker>> workers;
    for (int thread = 0; thread < join_threads; ++thread) {
        workers.push_back(std::make_unique<JoinWorker>(locations));
    }
    std::vector<RoundRecord> rounds;
    stream.start_time = Clock::now();
    std::vector<std::thread> joins;
    for (int thread = 0; thread < join_threads; ++thread) {
        joins.emplace_back(join_thread, std::ref(stream), std::ref(*workers[size_t(thread)]));
    }
    std::vector<std::thread> updaters;
    for (int thread = 0; thread < update_threads; ++thread) {
        updaters.emplace_back(update_thread, std::ref(stream));
    }
    std::thread maintainer(maintenance_thread, std::ref(stream), std::cref(updated), maintain_threads,
                           std::ref(rounds));
    for (std::thread& thread : updaters) {
        thread.join();
    }
    stream.updates_done.store(true);
    for (std::thread& thread : joins) {
        thread.join();
    }
    stream.probes_done.store(true);
    maintainer.join();
    const double stream_seconds = seconds_since(stream.start_time);

    write_probes(out + "/probes.tsv", workers);
    write_updates(out + "/updates.tsv", stream);
    write_rounds(out + "/maintenance.tsv", rounds);
    size_t probes = 0;
    for (const std::unique_ptr<JoinWorker>& worker : workers) {
        probes += worker->records.size();
    }
    seams::Json summary;
    summary.text("program", "updates")
        .text("mode", mode)
        .text("index", index_prefix)
        .text("seams", seams_path)
        .text("update_file", updates_path)
        .number("points", points)
        .number("indexed_at_start", indexed)
        .number("queries", stream.queries.count)
        .number("arriving_queries", arriving)
        .number("updates", stream.updates.size())
        .number("updates_applied", stream.updates_applied.load())
        .number("inserts_applied", stream.inserts_applied.load())
        .number("deletes_applied", stream.deletes_applied.load())
        .number("queries_applied", stream.queries_applied.load())
        .number("skipped_absent", stream.skipped_absent.load())
        .number("probes", probes)
        .number("rounds", rounds.size())
        .number("join_threads", join_threads)
        .number("update_threads", update_threads)
        .number("maintain_threads", maintain_threads)
        .number("query_rate", stream.query_rate)
        .number("update_rate", stream.update_rate)
        .number("duration_s", double(stream.duration_us) / 1e6)
        .number("maintain_every", stream.maintain_every)
        .number("eps", stream.eps)
        .number("L", stream.pool_width)
        .number("delta", stream.delta)
        .number("R", stream.recipe.degree)
        .number("vamana_L", stream.recipe.list_size)
        .number("C", stream.recipe.candidates)
        .number("alpha", stream.recipe.alpha)
        .number("capacity", stream.recipe.capacity)
        .number("twin_row", stream.twin_row.load())
        .number("load_s", load_seconds)
        .number("stream_s", stream_seconds)
        .number("peak_rss_mib", seams::peak_rss_mib())
        .write(out + "/summary.json");
    std::cerr << "updates: " << probes << " probes, " << stream.updates_applied.load() << " update lines, "
              << rounds.size() << " maintenance rounds in " << stream_seconds << " s\n";
    return 0;
}

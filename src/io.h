// io.h - input and output of the join and of the update program: the command line (--key value), .fvecs vectors and
// .nsg graphs read into hash maps, the L2 distance of the join, the seam table and the exact answer sizes as the join
// reads them, and the report of a join (queries.tsv, summary.json).
// Header only, C++17: #include "io.h".
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using Vectors = std::unordered_map<int, std::vector<float>>;   // vector id -> its values

[[noreturn]] inline void die(const std::string& message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

// the distance of the join: L2 with the difference of each dimension in float and the sum in double
inline double l2_distance(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size()) {
        die("vectors of different dimensions");
    }
    double sum = 0.0;
    for (size_t index = 0; index < left.size(); index++) {
        const double difference = left[index] - right[index];
        sum += difference * difference;
    }
    return std::sqrt(sum);
}

// an out-edge: the neighbour and a weight (the distance between the two vectors, or 0)
struct Edge {
    int neighbour;
    double weight;
};

// A directed graph in a hash map: node -> its out-edges in the order they were added; a node is in the map once it
// has an out-edge. get_nodes() lists the nodes in the iteration order of the map, and the MST order of the queries
// follows it, so the map type and the order in which nodes are added are part of the join's behaviour.
class Graph {
public:
    void add_edge(int from, int to, double weight) {
        edges_[from].push_back({to, weight});
    }

    const std::vector<Edge>& get_neighbors(int node) const {
        const std::unordered_map<int, std::vector<Edge>>::const_iterator found = edges_.find(node);
        if (found == edges_.end()) {
            return no_edges_;
        }
        return found->second;
    }

    std::vector<int> get_nodes() const {
        std::vector<int> nodes;
        for (const std::pair<const int, std::vector<Edge>>& node : edges_) {
            nodes.push_back(node.first);
        }
        return nodes;
    }

private:
    std::unordered_map<int, std::vector<Edge>> edges_;
    std::vector<Edge> no_edges_;
};

// The first `limit` vectors of an .fvecs file, all of them by default: per vector [dimension int32][dimension x
// float32]. A vector is keyed by its position in the file; one cut short by the end of the file is not read.
inline Vectors read_vectors(const std::string& path, size_t limit = std::numeric_limits<size_t>::max()) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        die("cannot open " + path);
    }
    Vectors vectors;
    for (size_t id = 0; id < limit; id++) {
        int32_t dimension = 0;
        if (!file.read(reinterpret_cast<char*>(&dimension), sizeof dimension)) {
            break;
        }
        std::vector<float> values(dimension);
        if (!file.read(reinterpret_cast<char*>(values.data()), dimension * sizeof(float))) {
            break;
        }
        vectors.emplace(int(id), std::move(values));
    }
    return vectors;
}

// The out-neighbours of every node of an .nsg file and its entry point: [width uint32][entry uint32], then per node
// [degree uint32][degree x uint32]. Nodes are numbered by their position in the file and added to the map in that
// order; a node cut short by the end of the file is not read.
struct NsgLists {
    int entry = -1;
    std::unordered_map<int, std::vector<int>> neighbours;
};

inline NsgLists read_nsg_lists(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        die("cannot open " + path);
    }
    uint32_t width = 0;
    uint32_t entry = 0;
    file.read(reinterpret_cast<char*>(&width), sizeof width);
    file.read(reinterpret_cast<char*>(&entry), sizeof entry);
    if (!file) {
        die("bad nsg header: " + path);
    }
    NsgLists lists;
    lists.entry = int(entry);
    int node = 0;
    uint32_t degree = 0;
    while (file.read(reinterpret_cast<char*>(&degree), sizeof degree)) {
        std::vector<uint32_t> ids(degree);
        if (!file.read(reinterpret_cast<char*>(ids.data()), degree * sizeof(uint32_t))) {
            break;
        }
        lists.neighbours.emplace(node, std::vector<int>(ids.begin(), ids.end()));
        node++;
    }
    return lists;
}

// The command line: --key value pairs. Every key a program reads is marked as used; check() rejects the others.
struct Args {
    std::map<std::string, std::string> values;
    std::set<std::string> used;

    Args(int argc, char** argv) {
        for (int position = 1; position < argc; position += 2) {
            const std::string key = argv[position];
            if (key.rfind("--", 0) != 0 || position + 1 >= argc) {
                die(std::string("usage: ") + argv[0] + " --key value ...");
            }
            values[key.substr(2)] = argv[position + 1];
        }
    }

    bool has(const std::string& key) const {
        return values.count(key) > 0;
    }

    // the value of --key; without one, the fallback, or exit when there is none
    std::string text(const std::string& key, const char* fallback = nullptr) {
        used.insert(key);
        if (has(key)) {
            return values.at(key);
        }
        if (!fallback) {
            die("missing --" + key);
        }
        return fallback;
    }

    double number(const std::string& key, const char* fallback = nullptr) {
        return std::stod(text(key, fallback));
    }

    void check() const {
        for (const std::pair<const std::string, std::string>& value : values) {
            if (!used.count(value.first)) {
                die("unknown --" + value.first);
            }
        }
    }
};

// What the join reads: the query vectors X and data vectors Y, their graphs, and the entry point of G_Y.
struct Data {
    Vectors queries;
    Vectors points;
    Graph query_graph;   // its edges carry d(u, v), the weights of the MST
    Graph point_graph;
    int entry_point = -1;
};

// The graph of the adjacency lists, read in the iteration order of their map; weighted: every edge carries the
// distance between its two vectors, else 0.
inline Graph build_graph(const std::unordered_map<int, std::vector<int>>& adjacency, const Vectors& vectors,
                         bool weighted) {
    Graph graph;
    for (const std::pair<const int, std::vector<int>>& node : adjacency) {
        for (int neighbour : node.second) {
            double weight = 0.0;
            if (weighted) {
                weight = l2_distance(vectors.at(node.first), vectors.at(neighbour));
            }
            graph.add_edge(node.first, neighbour, weight);
        }
    }
    return graph;
}

// --x-fvecs, --x-nsg, --y-nsg, --y-fvecs (the first |G_Y| vectors of it)
inline Data load(Args& args) {
    Data data;
    data.queries = read_vectors(args.text("x-fvecs"));
    const NsgLists query_lists = read_nsg_lists(args.text("x-nsg"));
    data.query_graph = build_graph(query_lists.neighbours, data.queries, true);
    const NsgLists point_lists = read_nsg_lists(args.text("y-nsg"));
    data.points = read_vectors(args.text("y-fvecs"), point_lists.neighbours.size());
    data.point_graph = build_graph(point_lists.neighbours, data.points, false);
    data.entry_point = point_lists.entry;
    return data;
}

// The seam table: lines "x_id y_id dist rank" (a header line is skipped; y_id -1 = a query without seams). Per query
// the seams of the lowest ranks, at most per_query of them, in rank order.
inline std::vector<std::vector<int>> read_seams(const std::string& path, int per_query) {
    std::ifstream file(path);
    if (!file) {
        die("cannot open " + path);
    }
    std::map<int, std::vector<std::pair<int, int>>> ranked;   // query -> (rank, point)
    std::string query;
    std::string point;
    std::string distance;
    std::string rank;
    while (file >> query >> point >> distance >> rank) {
        const bool header = !std::isdigit(static_cast<unsigned char>(query[0]));
        const bool no_seam = point[0] == '-';
        if (header || no_seam) {
            continue;
        }
        ranked[std::stoi(query)].emplace_back(std::stoi(rank), std::stoi(point));
    }
    std::vector<std::vector<int>> seams;
    if (!ranked.empty()) {
        seams.resize(ranked.rbegin()->first + 1);
    }
    for (std::pair<const int, std::vector<std::pair<int, int>>>& owner : ranked) {
        std::vector<std::pair<int, int>>& candidates = owner.second;
        std::sort(candidates.begin(), candidates.end());
        const size_t kept = std::min(candidates.size(), static_cast<size_t>(per_query));
        for (size_t position = 0; position < kept; position++) {
            seams[owner.first].push_back(candidates[position].second);
        }
    }
    return seams;
}

// the exact answer size of every query, one per line, line i = query i
inline std::vector<long long> read_counts(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        die("cannot open " + path);
    }
    std::vector<long long> counts;
    long long count;
    while (file >> count) {
        counts.push_back(count);
    }
    return counts;
}

// the phases of a probe: the seeds, the greedy phase, the expansion
enum Phase { SEED, GREEDY, EXPAND };

struct QueryStats {
    int query = -1;
    int parent = -1;
    long long results = 0;
    long long distances[3] = {};   // distance computations per phase
    long long redundant[3] = {};   // ... of points out of range
    double microseconds = 0;
};

// the latency below which a share of the queries finishes, from latencies sorted ascending
inline double percentile(const std::vector<double>& sorted_latencies, double share) {
    size_t position = share * sorted_latencies.size();
    if (position >= sorted_latencies.size()) {
        position = sorted_latencies.size() - 1;
    }
    return sorted_latencies[position];
}

// {"seed": ..., "greedy": ..., "expand": ..., "total": ...}
inline std::string by_phase(const long long* totals) {
    std::ostringstream json;
    json << "{\"seed\": " << totals[SEED] << ", \"greedy\": " << totals[GREEDY] << ", \"expand\": " << totals[EXPAND]
         << ", \"total\": " << totals[SEED] + totals[GREEDY] + totals[EXPAND] << "}";
    return json.str();
}

// Writes <directory>/queries.tsv (a line per query) and <directory>/summary.json, and a summary line to stdout.
// counts: the exact answer sizes (empty: no ground truth, recall -1).
inline void report(const std::string& directory, const Args& args, const std::vector<QueryStats>& stats,
                   const std::vector<long long>& counts, double order_seconds, double join_seconds) {
    std::filesystem::create_directories(directory);
    std::ofstream per_query(directory + "/queries.tsv");
    per_query << "x\tparent\tresults\tanswer\trecall\tndc_seed\tndc_greedy\tndc_expand"
                 "\tred_seed\tred_greedy\tred_expand\tus\n";
    long long pairs = 0;
    long long answer_pairs = 0;
    long long distances[3] = {};
    long long redundant[3] = {};
    std::vector<double> latencies_ms;
    for (const QueryStats& entry : stats) {
        long long answer = -1;   // -1: no ground truth given
        if (!counts.empty()) {
            answer = counts.at(entry.query);
        }
        double recall = -1.0;
        if (answer == 0) {
            recall = 1.0;
        }
        if (answer > 0) {
            recall = static_cast<double>(entry.results) / answer;
        }
        per_query << entry.query << '\t' << entry.parent << '\t' << entry.results << '\t' << answer << '\t' << recall;
        for (int phase = 0; phase < 3; phase++) {
            per_query << '\t' << entry.distances[phase];
        }
        for (int phase = 0; phase < 3; phase++) {
            per_query << '\t' << entry.redundant[phase];
        }
        per_query << '\t' << entry.microseconds << '\n';
        pairs += entry.results;
        if (answer > 0) {
            answer_pairs += answer;
        }
        for (int phase = 0; phase < 3; phase++) {
            distances[phase] += entry.distances[phase];
            redundant[phase] += entry.redundant[phase];
        }
        latencies_ms.push_back(entry.microseconds / 1000);
    }
    const double query_count = stats.size();
    const double mean_ms = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / query_count;
    std::sort(latencies_ms.begin(), latencies_ms.end());
    long long gt_pairs = -1;   // -1: no ground truth given
    double recall = -1.0;
    if (!counts.empty()) {
        gt_pairs = answer_pairs;
    }
    if (answer_pairs > 0) {
        recall = static_cast<double>(pairs) / answer_pairs;
    }

    std::ofstream summary(directory + "/summary.json");
    summary << std::setprecision(10) << "{\n  \"args\": {";
    std::string separator;
    for (const std::pair<const std::string, std::string>& value : args.values) {
        summary << separator << '"' << value.first << "\": \"" << value.second << '"';
        separator = ", ";
    }
    summary << "},\n  \"queries\": " << stats.size() << ",\n  \"pairs\": " << pairs
            << ",\n  \"gt_pairs\": " << gt_pairs << ",\n  \"recall\": " << recall
            << ",\n  \"order_s\": " << order_seconds << ",\n  \"join_s\": " << join_seconds
            << ",\n  \"ms_per_query\": {\"mean\": " << mean_ms << ", \"p50\": " << percentile(latencies_ms, 0.5)
            << ", \"p90\": " << percentile(latencies_ms, 0.9) << ", \"p99\": " << percentile(latencies_ms, 0.99) << "}"
            << ",\n  \"ndc\": " << by_phase(distances) << ",\n  \"redundant\": " << by_phase(redundant) << "\n}\n";
    std::cout << "queries " << stats.size() << "  pairs " << pairs << "  recall " << recall << "  ms/query " << mean_ms
              << "  ndc/query " << (distances[SEED] + distances[GREEDY] + distances[EXPAND]) / query_count
              << "  redundant/query " << (redundant[SEED] + redundant[GREEDY] + redundant[EXPAND]) / query_count
              << std::endl;
}

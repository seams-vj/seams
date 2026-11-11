// join.cpp - the threshold join of a query set X with a data set Y over their graph indexes G_X and G_Y: every pair
// (x, y) with d(x, y) <= the threshold, found by one probe of G_Y per query. The queries are probed in the depth-first
// order of a minimum spanning tree of G_X (order.h). A probe starts from the seams of its query in the seam table
// written by build_seams, and from the closest data vector that the probe of its tree parent computed a distance to
// (soft work sharing). A query whose nearest seam lies beyond the threshold gets the empty answer without a distance
// computation. Otherwise the greedy phase moves from the starts towards the query until it reaches a data vector
// within the threshold, and the expansion phase collects every data vector within the threshold that is reachable
// from the ones found through data vectors within the threshold. Without --seams a probe starts from the entry point
// of G_Y and the parent's closest vector: soft work sharing alone. One thread.
//
//   join --eps THRESHOLD --x-fvecs X.fvecs --x-nsg X.nsg --y-fvecs Y.fvecs --y-nsg Y.nsg
//        [--seams SEAM_TABLE [--m SEAMS_PER_QUERY]] [--gt-counts COUNTS] [--out DIRECTORY]
//
// --m M starts each probe from the M nearest seams of its query only (--m 1: the nearest seam); by default from all.
// --gt-counts: the exact answer size of each query, one per line, for the recall.
// Writes DIRECTORY/queries.tsv (a line per query, in the order of the probes) and DIRECTORY/summary.json (io.h);
// DIRECTORY is join_out by default. A query that got the empty answer has no distance computation in queries.tsv.
#include "io.h"
#include "order.h"

#include <cctype>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// The greedy phase keeps its CANDIDATE_QUEUE_SIZE nearest candidates. It stops when its nearest candidate lies within
// the threshold, or when its nearest candidate has not come closer for STALL_LIMIT steps in a row.
constexpr size_t CANDIDATE_QUEUE_SIZE = 256;
constexpr int STALL_LIMIT = 10;

// The seam table prints distances rounded, so a nearest seam printed less than this beyond the threshold may lie
// within it and does not give the empty answer.
constexpr double ROUNDING_TOLERANCE = 1e-9;

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

double microseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

// The distance of the nearest seam (rank 0) of every query in the seam table; NaN for a query without one.
std::vector<double> read_nearest_seam_distance(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        die("cannot open " + path);
    }
    std::vector<double> nearest_distances;
    std::string query;
    std::string point;
    std::string distance;
    std::string rank;
    while (file >> query >> point >> distance >> rank) {
        const bool header = !std::isdigit(static_cast<unsigned char>(query[0]));
        const bool no_seam = point[0] == '-';
        if (header || no_seam || rank != "0") {
            continue;
        }
        const size_t query_id = std::stoul(query);
        if (query_id >= nearest_distances.size()) {
            nearest_distances.resize(query_id + 1, std::numeric_limits<double>::quiet_NaN());
        }
        nearest_distances[query_id] = std::stod(distance);
    }
    return nearest_distances;
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    const double threshold = args.number("eps");
    const std::string out = args.text("out", "join_out");
    const Data data = load(args);
    std::vector<std::vector<int>> seams;
    std::vector<double> nearest_seam_distance;
    if (args.has("seams")) {
        int seams_per_query = std::numeric_limits<int>::max();
        if (args.has("m")) {
            seams_per_query = static_cast<int>(args.number("m"));
        }
        seams = read_seams(args.text("seams"), seams_per_query);
        nearest_seam_distance = read_nearest_seam_distance(args.text("seams"));
    }
    std::vector<long long> counts;
    if (args.has("gt-counts")) {
        counts = read_counts(args.text("gt-counts"));
    }
    args.check();

    const Clock::time_point order_start = Clock::now();
    const std::vector<std::pair<int, int>> order =
        mst_order(data.query_graph, data.queries, data.points.at(data.entry_point));
    const Clock::time_point join_start = Clock::now();

    std::unordered_map<int, int> closest_visited;   // query -> the closest data vector its probe computed a distance to
    std::vector<QueryStats> stats;
    for (const std::pair<int, int>& step : order) {
        const int parent = step.first;
        const int query = step.second;
        const Clock::time_point query_start = Clock::now();
        QueryStats query_stats;
        query_stats.query = query;
        query_stats.parent = parent;

        // The empty answer: the nearest seam lies beyond the threshold (a NaN distance never does). No distance is
        // computed, and the tree children receive the nearest seam.
        const bool has_seams = static_cast<size_t>(query) < seams.size() && !seams[query].empty();
        const bool empty_answer = has_seams && static_cast<size_t>(query) < nearest_seam_distance.size() &&
                                  nearest_seam_distance[query] > threshold + ROUNDING_TOLERANCE;
        if (empty_answer) {
            closest_visited[query] = seams[query].front();
            query_stats.microseconds = microseconds_since(query_start);
            stats.push_back(query_stats);
            continue;
        }

        const std::vector<float>& query_vector = data.queries.at(query);
        std::unordered_set<int> visited;
        std::vector<int> matches;                           // the data vectors found within the threshold
        std::multiset<std::pair<double, int>> candidates;   // (distance, data vector), nearest first
        int closest_point = -1;
        double closest_distance = std::numeric_limits<double>::max();
        // computes the distance of a data vector to the query, counts it in the phase and keeps the closest vector
        auto visit = [&](int point, Phase phase) -> double {
            visited.insert(point);
            const double distance = l2_distance(query_vector, data.points.at(point));
            query_stats.distances[phase]++;
            if (distance > threshold) {
                query_stats.redundant[phase]++;
            }
            if (distance < closest_distance) {
                closest_distance = distance;
                closest_point = point;
            }
            return distance;
        };

        // the starts: the seams of the query in rank order, or the entry point of G_Y for a query without seams
        std::vector<int> seeds = {data.entry_point};
        if (has_seams) {
            seeds = seams[query];
        }
        for (int seed : seeds) {
            if (visited.count(seed) > 0) {
                continue;
            }
            const double distance = visit(seed, SEED);
            candidates.emplace(distance, seed);
            if (distance <= threshold) {
                matches.push_back(seed);
            } else if (has_seams && !matches.empty()) {
                break;   // the seams come by distance to the query, so the remaining ones lie beyond the threshold too
            }
        }
        // soft work sharing: the closest data vector of the parent's probe
        const std::unordered_map<int, int>::iterator passed = closest_visited.find(parent);
        if (passed != closest_visited.end() && visited.count(passed->second) == 0) {
            const int passed_point = passed->second;
            const double distance = visit(passed_point, SEED);
            candidates.emplace(distance, passed_point);
            if (distance <= threshold) {
                matches.push_back(passed_point);
            }
        }

        // the greedy phase: every neighbour not visited yet becomes a candidate
        double nearest_so_far = std::numeric_limits<double>::max();
        int steps_without_improvement = 0;
        while (!candidates.empty()) {
            const double nearest_distance = candidates.begin()->first;
            const int nearest_point = candidates.begin()->second;
            if (nearest_distance <= threshold) {
                break;
            }
            if (nearest_distance < nearest_so_far) {
                nearest_so_far = nearest_distance;
                steps_without_improvement = 0;
            } else {
                steps_without_improvement++;
                if (steps_without_improvement >= STALL_LIMIT) {
                    break;
                }
            }
            candidates.erase(candidates.begin());
            for (const Edge& edge : data.point_graph.get_neighbors(nearest_point)) {
                if (visited.count(edge.neighbour) > 0) {
                    continue;
                }
                const double distance = visit(edge.neighbour, GREEDY);
                candidates.emplace(distance, edge.neighbour);
                if (distance <= threshold) {
                    matches.push_back(edge.neighbour);
                }
            }
            while (candidates.size() > CANDIDATE_QUEUE_SIZE) {
                candidates.erase(std::prev(candidates.end()));
            }
        }

        // the expansion phase: breadth first from the matches found so far, through the matches it finds
        std::queue<int> frontier;
        for (int point : matches) {
            frontier.push(point);
        }
        while (!frontier.empty()) {
            const int point = frontier.front();
            frontier.pop();
            for (const Edge& edge : data.point_graph.get_neighbors(point)) {
                if (visited.count(edge.neighbour) > 0) {
                    continue;
                }
                const double distance = visit(edge.neighbour, EXPAND);
                if (distance <= threshold) {
                    frontier.push(edge.neighbour);
                    matches.push_back(edge.neighbour);
                }
            }
        }

        if (closest_point != -1) {
            closest_visited[query] = closest_point;
        }
        query_stats.results = matches.size();
        query_stats.microseconds = microseconds_since(query_start);
        stats.push_back(query_stats);
    }
    const double join_seconds = seconds_between(join_start, Clock::now());
    report(out, args, stats, counts, seconds_between(order_start, join_start), join_seconds);
}

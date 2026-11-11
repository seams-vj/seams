// order.h - the order in which the join probes the queries: a minimum spanning tree over the query graph G_X and the
// entry point of G_Y, walked depth first, so that every query comes after its tree parent and soft work sharing can
// hand it the parent's closest vector.
// Header only, C++17: #include "order.h".
#pragma once
#include "io.h"

#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// the union-find root of a node, halving the path on the way
inline int find_root(std::unordered_map<int, int>& parent_of, int node) {
    while (parent_of[node] != node) {
        parent_of[node] = parent_of[parent_of[node]];
        node = parent_of[node];
    }
    return node;
}

// adds the edge first-second to the tree unless the two nodes are connected already
inline void join_trees(std::unordered_map<int, int>& parent_of, std::unordered_map<int, std::vector<int>>& tree,
                       int first, int second) {
    const int first_root = find_root(parent_of, first);
    const int second_root = find_root(parent_of, second);
    if (first_root == second_root) {
        return;
    }
    parent_of[first_root] = second_root;
    tree[first].push_back(second);
    tree[second].push_back(first);
}

// Kruskal over the edges (u, v), u < v, of the query graph, weighted by d(u, v), with the edges (entry point, x) of
// weight d(x, entry point) merged in by weight; then a depth-first walk of the tree from every query not reached yet,
// in the order of query_graph.get_nodes(). Returns (parent, query) in processing order; parent -1 (the entry point)
// starts a walk.
inline std::vector<std::pair<int, int>> mst_order(const Graph& query_graph, const Vectors& queries,
                                                  const std::vector<float>& entry_vector) {
    const int ENTRY = -1;
    std::vector<std::pair<double, int>> entry_edges;
    for (const std::pair<const int, std::vector<float>>& query : queries) {
        entry_edges.emplace_back(l2_distance(query.second, entry_vector), query.first);
    }
    std::sort(entry_edges.begin(), entry_edges.end());
    const std::vector<int> nodes = query_graph.get_nodes();
    std::vector<std::tuple<double, int, int>> edges;
    for (int node : nodes) {
        for (const Edge& edge : query_graph.get_neighbors(node)) {
            if (node < edge.neighbour) {
                edges.emplace_back(edge.weight, node, edge.neighbour);
            }
        }
    }
    std::sort(edges.begin(), edges.end());

    std::unordered_map<int, int> parent_of;
    parent_of[ENTRY] = ENTRY;
    for (int node : nodes) {
        parent_of[node] = node;
    }
    std::unordered_map<int, std::vector<int>> tree;
    size_t next_entry_edge = 0;
    for (const std::tuple<double, int, int>& edge : edges) {
        const double weight = std::get<0>(edge);
        while (next_entry_edge < entry_edges.size() && entry_edges[next_entry_edge].first <= weight) {
            join_trees(parent_of, tree, ENTRY, entry_edges[next_entry_edge].second);
            next_entry_edge++;
        }
        join_trees(parent_of, tree, std::get<1>(edge), std::get<2>(edge));
    }

    std::vector<std::pair<int, int>> order;
    std::unordered_set<int> done;
    for (int start : nodes) {
        std::vector<std::pair<int, int>> stack = {{ENTRY, start}};
        while (!stack.empty()) {
            const int parent = stack.back().first;
            const int query = stack.back().second;
            stack.pop_back();
            if (done.count(query) > 0) {
                continue;
            }
            done.insert(query);
            order.emplace_back(parent, query);
            for (int neighbour : tree[query]) {
                if (neighbour != ENTRY && done.count(neighbour) == 0) {
                    stack.emplace_back(query, neighbour);
                }
            }
        }
    }
    return order;
}

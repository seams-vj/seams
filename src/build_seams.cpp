// build_seams - Build of the seam table, offline. For every query: one top-k search on the graph index of the data
// (its candidate list of length L), Profile over the list and Select of the seams with recall tolerance delta.
// Writes under --out:
//   <dataset>.seams.tsv   the seam table, one row per seam: x_id, y_id, dist, rank, ranks ascending (rank 0, the
//                         nearest candidate, is always a seam); a query without candidates has one row with y_id -1
//   <dataset>.radius.tsv  the coverage radius of every query: x, pool_len (the length of its candidate list), r_L
//                         (the distance of its last candidate, nan without candidates)
//   <dataset>.run.json    the options, the counts and the seconds of each step
// The seam table and the radius file do not depend on --threads. Run (L 256 and delta 0.05 are the paper's settings):
//   build_seams --dataset sift --nsg sift_base.nsg --fvecs sift_base.fvecs --query-fvecs sift_query.fvecs
//               --L 256 --delta 0.05 --threads 24 --out seams
#include "seams.h"

using namespace seams;

namespace {

struct Options {
    std::string dataset;
    std::string nsg;           // the graph index of the data
    std::string fvecs;         // the data vectors
    std::string query_fvecs;   // the query vectors
    std::string out;           // the output folder
    size_t pool_width = 0;     // --L: the length of the candidate list of the top-k search
    double delta = 0.05;
    int threads = 1;
};

// the value of the option at argv[index], i.e. argv[index + 1]; index moves onto it
std::string option_value(int argc, char** argv, int& index, const std::string& key) {
    if (index + 1 >= argc) {
        die("missing value for " + key);
    }
    ++index;
    return argv[index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--dataset") {
            options.dataset = option_value(argc, argv, index, key);
        } else if (key == "--nsg") {
            options.nsg = option_value(argc, argv, index, key);
        } else if (key == "--fvecs") {
            options.fvecs = option_value(argc, argv, index, key);
        } else if (key == "--query-fvecs") {
            options.query_fvecs = option_value(argc, argv, index, key);
        } else if (key == "--out") {
            options.out = option_value(argc, argv, index, key);
        } else if (key == "--L") {
            options.pool_width = std::stoul(option_value(argc, argv, index, key));
        } else if (key == "--delta") {
            options.delta = std::stod(option_value(argc, argv, index, key));
        } else if (key == "--threads") {
            options.threads = std::max(1, std::stoi(option_value(argc, argv, index, key)));
        } else {
            die("unknown option " + key);
        }
    }
    const bool missing = options.dataset.empty() || options.nsg.empty() || options.fvecs.empty() ||
                         options.query_fvecs.empty() || options.out.empty() || options.pool_width == 0;
    if (missing) {
        die("usage: build_seams --dataset D --nsg F --fvecs F --query-fvecs F --L 256 --out DIR [--delta 0.05] "
            "[--threads 1]");
    }
    return options;
}

// the graph index, the data vectors and the candidate list of every query
struct Inputs {
    Graph graph;
    Vectors base;
    Pools pools;
    uint64_t pool_distances = 0;   // the distances the top-k searches computed
};

// Reads the inputs and runs the top-k search of every query; the time of each step goes into `seconds`.
// The query vectors live only in here: Profile and Select do not need them.
Inputs load_inputs(const Options& options, Json& seconds) {
    Inputs inputs;
    double started = now_s();
    inputs.graph = read_nsg(options.nsg);
    seconds.number("load_graph", now_s() - started);
    started = now_s();
    inputs.base = read_fvecs(options.fvecs, inputs.graph.size());
    const Vectors queries = read_fvecs(options.query_fvecs);
    seconds.number("load_vectors", now_s() - started);
    started = now_s();
    inputs.pools = search_pools(inputs.graph, inputs.base, queries, options.pool_width, options.threads,
                                inputs.pool_distances);
    seconds.number("pools", now_s() - started);
    return inputs;
}

void write_seam_table(const std::string& path, const Pools& pools,
                      const std::vector<std::vector<int32_t>>& seam_ranks) {
    std::ofstream file(path);
    if (!file) {
        die("cannot write " + path);
    }
    file << "x_id\ty_id\tdist\trank\n";
    file << std::setprecision(12);
    for (size_t query = 0; query < seam_ranks.size(); ++query) {
        if (pools.length(query) == 0) {
            file << query << "\t-1\tnan\t0\n";
            continue;
        }
        const uint32_t* ids = pools.ids_of(query);
        const double* distances = pools.dist_of(query);
        for (int32_t rank : seam_ranks[query]) {
            file << query << '\t' << ids[rank] << '\t' << distances[rank] << '\t' << rank << '\n';
        }
    }
}

void write_radius(const std::string& path, const Pools& pools) {
    std::ofstream file(path);
    if (!file) {
        die("cannot write " + path);
    }
    file << "x\tpool_len\tr_L\n";
    file << std::setprecision(12);
    for (size_t query = 0; query < pools.queries(); ++query) {
        const size_t length = pools.length(query);
        file << query << '\t' << length << '\t';
        if (length > 0) {
            file << pools.dist_of(query)[length - 1] << '\n';
        } else {
            file << "nan\n";
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    Json seconds;
    const Inputs inputs = load_inputs(options, seconds);
    const Pools& pools = inputs.pools;
    const size_t query_count = pools.queries();

    std::vector<std::vector<int32_t>> seam_ranks(query_count);
    Workers<BuildWorker> workers(options.threads, inputs.graph);
    double started = now_s();
    for_each_query(workers, query_count, [&seam_ranks, &pools, &options](BuildWorker& worker, size_t query) {
        seam_ranks[query] = worker.select(pools, query, options.delta);
    });
    const double pass_seconds = now_s() - started;

    started = now_s();
    const std::string prefix = options.out + "/" + options.dataset;
    uint64_t seam_count = 0;
    for (const std::vector<int32_t>& ranks : seam_ranks) {
        seam_count += ranks.size();
    }
    write_seam_table(prefix + ".seams.tsv", pools, seam_ranks);
    write_radius(prefix + ".radius.tsv", pools);
    const double write_seconds = now_s() - started;

    uint64_t relabels = 0;
    for (const BuildWorker* worker : workers.made()) {
        relabels += worker->reach.relabels();
    }
    seconds.number("pass", pass_seconds);
    seconds.number("write", write_seconds);
    Json run;
    run.text("program", "build_seams");
    run.text("dataset", options.dataset);
    run.text("nsg", options.nsg);
    run.text("fvecs", options.fvecs);
    run.text("query_fvecs", options.query_fvecs);
    run.number("L", options.pool_width);
    run.number("delta", options.delta);
    run.number("queries", query_count);
    run.number("threads", options.threads);
    run.number("pool_width", pools.widest());   // the longest candidate list, at most L
    run.number("pool_distances", inputs.pool_distances);
    run.number("seams", seam_count);
    run.number("relabels", relabels);
    run.raw("seconds", seconds.str(4));
    run.number("peak_rss_mib", peak_rss_mib());
    run.write(prefix + ".run.json");
    std::cerr << options.dataset << " build_seams: " << seam_count << " seams over " << query_count << " queries, pass "
              << pass_seconds << " s on " << options.threads << " threads\n";
    return 0;
}

# Seams

Code of our method in the paper *Find Offline, Share Online: A Join Index for Threshold-Based Vector Joins over Graph
Indexes*.

A threshold-based vector join returns every pair (x, y) of a query set X and a data set Y with d(x, y) ≤ θ, here over
a graph index per set (G_X and G_Y). Offline, Build runs one top-k search per query on G_Y and keeps a few seams per
query: data vectors from which the expansion phase reaches the in-range regions of the query's candidate list, with
recall tolerance δ. Online, the join probes the queries along a minimum spanning tree of G_X, starts each probe from
the seams of its query and from the closest vector its tree parent visited (soft work sharing), returns the empty
answer without a distance computation when θ is below the distance r₁ of the nearest seam, and maintenance rounds
rerun Build for the rows that inserts and deletes of Y made stale and for new queries.

## Build

Requirements: Linux on x86-64 with AVX2 and FMA, g++ with C++17 (the paper used GCC 11.4.0) and make.
`bin/updates` and `bin/vamana` also need OpenMP, git, the Boost headers (DiskANN includes
`boost/dynamic_bitset.hpp`) and network access on the first build, which downloads DiskANN into
`third_party/DiskANN` and applies `third_party/diskann.patch` (`third_party/fetch_diskann.sh`).

    make                              # bin/build_seams bin/join bin/updates bin/vamana
    make bin/build_seams bin/join     # the static join only, without DiskANN

The Makefile uses the compiler flags of the paper's binaries, including `-march=native`.

## Programs

| program | what it does | in the paper |
|---|---|---|
| `build_seams` | runs a top-k search per query, then Profile and Select, and writes the seam table and the radius file | the seam table of every join, Fig. 10 and the build cost and coverage in Sec. VII-D |
| `join` | probes the queries in MST order with soft work sharing, starts from the seams and returns the empty answer when θ < r₁, on one thread | Seams+SWS, Seams-s₁+SWS and SWS in Fig. 6–9 and 13 and in Tables IV and V |
| `vamana` | builds a dynamic Vamana index with DiskANN and exports it as `.nsg` | the indexes of the update experiments |
| `updates` | runs the join while rows of Y are inserted and deleted and new queries arrive, and maintains the seam table in the background (`async`), synchronously (`sync`) or not at all (`none`) | Fig. 11 and 12 |

Each source file starts with a comment on its options and outputs.

## Input formats

- `.fvecs` (vectors): per vector an int32 dimension d, then d float32 values; the i-th vector in the file is vector i.
  The join uses the L2 distance.
- `.nsg` (graphs), as written by the NSG code and by `vamana export`: uint32 width, uint32 entry point, then for every
  node in order a uint32 degree followed by that many uint32 neighbour ids.
- Seam table (`<dataset>.seams.tsv`, written by `build_seams`, read by `join --seams` and `updates --seams`): the header
  `x_id y_id dist rank`, then one tab-separated line per seam: the query, the data vector, their distance and the rank
  of the data vector in the candidate list of the query, ranks ascending (rank 0, the nearest candidate, is always a
  seam). A query without candidates has the single line `x -1 nan 0`.
- Radius file (`<dataset>.radius.tsv`, written by `build_seams`, read by `updates --radius`): the header
  `x pool_len r_L`, then one line per query: the length of its candidate list and r_L, the distance of its last
  candidate (`nan` for an empty list).
- Exact answer sizes (`join --gt-counts`, optional): one integer per line, line i for query i. They only enter the
  recall; this repository does not compute them (a brute-force scan does).
- Update file (`updates --updates`): one change per line, applied in file order at `--update-rate` lines per second.
  `i ROW` inserts and `d ROW` deletes row ROW of the `--data` file; `q X` is the arrival of query X, whose row of the
  seam table is dropped at the start and which is probed only once its line has been applied.
- Row list (`vamana build --ids`): one row number of the `.fvecs` file per line.

## Datasets and graphs

Each dataset is converted to `.fvecs`, with the train split of its HDF5 file as Y and the test split as X:

- [sift-128-euclidean](http://ann-benchmarks.com/sift-128-euclidean.hdf5) for SIFT
- [gist-960-euclidean](http://ann-benchmarks.com/gist-960-euclidean.hdf5) for GIST
- [glove-200-angular](http://ann-benchmarks.com/glove-200-angular.hdf5) for GloVe, with the vectors normalized to unit
  length
- [fashion-mnist-784-euclidean](http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5) for FMNIST
- [coco-nomic-768-normalized](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/coco-nomic-768-normalized.hdf5)
  for COCO
- [imagenet-align-640-normalized](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/imagenet-align-640-normalized.hdf5)
  for ImageNet
- [laion-clip-512-normalized](https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/laion-clip-512-normalized.hdf5)
  for LAION
- SIFT-E with 10⁶ queries and 10⁶ data vectors sampled from [SIFT1B](http://corpus-texmex.irisa.fr/)

`data/thresholds.json` gives θ for every dataset at the mean answer sizes k = 1, 5, 10, 20, 50 and 100.

G_X and G_Y of the static experiments are NSG graphs, built with the reference implementations of
[NNDescent](https://github.com/ZJULearning/efanna_graph) and [NSG](https://github.com/ZJULearning/nsg), for Y and
likewise for X:

    test_nndescent Y.fvecs Y.knn 400 400 12 15 100     # NNDescent: K L iter S R
    test_nsg_index Y.fvecs Y.knn 60 70 500 Y.nsg       # NSG: L R C

## Static joins (Fig. 6–10 and 13, Tables IV and V)

    bin/build_seams --dataset sift --nsg sift_base.nsg --fvecs sift_base.fvecs --query-fvecs sift_query.fvecs \
                    --L 256 --delta 0.05 --threads 24 --out seams
    bin/join --eps 128.62348152650821 --seams seams/sift.seams.tsv \
             --x-fvecs sift_query.fvecs --x-nsg sift_query.nsg --y-fvecs sift_base.fvecs --y-nsg sift_base.nsg \
             --gt-counts sift_k10.counts --out join_sift_k10

`--eps` is the threshold θ. The value above is θ of SIFT at k = 10 from `data/thresholds.json`.

`build_seams` writes three files to `seams/`:

- `sift.seams.tsv` is the seam table.
- `sift.radius.tsv` holds the radius r_L of every query.
- `sift.run.json` holds the counts and the time of each step.

The seam table and the radius file are the same for any number of threads.

`join` writes two files to `join_sift_k10/`:

- `queries.tsv` has one line per query in probe order with its answer size, recall, distance computations per phase
  and time in microseconds.
- `summary.json` has the totals over all queries: pairs, recall over pairs, latency per query and distance
  computations per phase.

- Seams-s₁+SWS (Fig. 9): add `--m 1`. SWS alone: leave out `--seams`.
- Fig. 7 varies δ (`--delta` from 0.3 to 0.01); Sec. VII-D varies L (`--L` 64, 128, 256, 1024) and compares r_L in
  the radius file with θ at k = 100.
- Fig. 13 runs both programs once per part of Y (N − 1 random parts). Any graph of Y in the `.nsg` format can replace
  the NSG graph (Table V: HNSW and CDMG+).

## Updates (Fig. 11 and 12)

    bin/vamana build --data sift_base.fvecs --ids y_init.txt --capacity 1010000 --R 70 --L 100 --C 500 --alpha 1.2 \
                     --threads 24 --out g_y
    bin/vamana export --index g_y --points 1000000 --out g_y.nsg
    bin/build_seams --dataset sift --nsg g_y.nsg --fvecs sift_base.fvecs --query-fvecs sift_query.fvecs \
                    --L 256 --delta 0.05 --threads 24 --out start
    bin/updates --index g_y --capacity 1010000 --data sift_base.fvecs --x-fvecs sift_query.fvecs --x-nsg g_x.nsg \
                --seams start/sift.seams.tsv --radius start/sift.radius.tsv --eps 128.62348152650821 --mode async \
                --updates updates.txt --update-rate 2000 --query-rate 2000 --duration-s 501 \
                --join-threads 12 --update-threads 4 --maintain-threads 8 --maintain-every 200000 --out async

The capacity is |Y| plus 1%. The defaults of `--L 256 --delta 0.05` (Build) and `--R 70 --vamana-L 100 --C 500
--alpha 1.2` (the index) are the paper's settings. `updates` writes `probes.tsv` (one line per probe: due, start and
finish time, empty answer, distance computations, answer), `updates.tsv`, `maintenance.tsv` (one line per maintenance
round) and `summary.json`. Recall over time needs the exact answers on the changing Y, which this repository does not
compute.

- Fig. 11 (as above): `y_init.txt` lists a random half of the rows of Y, and `updates.txt` alternates `d` of a row of
  that half and `i` of a row of the other half until every initial row is replaced (10⁶ lines on SIFT). A round starts
  every 200,000 lines, once 20% of Y has been inserted and 20% deleted. ImageNet: `--capacity 1293978`,
  `--maintain-every 256232`, `--duration-s 641.6`. The three lines are `--mode async`, `sync` and `none`.
- Fig. 12: G_Y holds all of Y, and `updates.txt` has a line `q X` for each query of a random 20% of X, arriving over
  100 s: `--update-rate 20` on SIFT and 2 on ImageNet, `--maintain-every 200` and 20 (a round every tenth of the
  arrivals), `--duration-s 150`, `--mode async`.
- G_X: a Vamana graph of X for SIFT (`vamana build` over all rows of X with capacity 10,100, then `vamana export`)
  and the NSG graph of X for ImageNet.

## License

Our code is under the MIT License (`LICENSE`). `third_party/diskann.patch` changes DiskANN and is under DiskANN's MIT
License (`third_party/DiskANN-LICENSE`, the same text as `LICENSE` in the DiskANN checkout). The headers in
`third_party/stubs` stand in for two libraries that DiskANN includes but its in-memory index does not use.

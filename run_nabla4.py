import argparse
import numpy as np

import nabla4_gtfn  # type: ignore [import-not-found]

import pickle

def print_median_runtimes(runtimes):
    for key in runtimes.keys():
        values = runtimes[key]
        print(
            "{} median runtime: {}".format(
                key,
                np.median(values),
            )
        )

def run_gtfn(repetitions, dry_runs, e2c2v, e2ecv, KDim, backend):
    import cupy as cp  # type: ignore [import-not-found]

    float_dtype = cp.float64
    int_dtype = cp.int32

    from gt4py.storage import zeros, from_array  # type: ignore [import-not-found]

    EdgeDim = len(e2c2v)

    z_nabla4_e2_wp_gtfn = zeros(
        shape=(EdgeDim, KDim), dtype=float_dtype, backend=backend
    )
    runtimes = nabla4_gtfn.calculate_nabla4_gpu(
        repetitions,
        dry_runs,
        (
            from_array(
                np.random.randn(EdgeDim, KDim), dtype=float_dtype, backend=backend
            ),
            (0, 0),
        ),
        (
            from_array(
                np.random.randn(EdgeDim, KDim), dtype=float_dtype, backend=backend
            ),
            (0, 0),
        ),
        (
            from_array(
                np.random.randn(EdgeDim),
                dtype=float_dtype,
                backend=backend,
            ),
            (0,),
        ),
        (
            from_array(
                np.random.randn(EdgeDim),
                dtype=float_dtype,
                backend=backend,
            ),
            (0,),
        ),
        (
            from_array(
                np.random.randn(EdgeDim, KDim),
                dtype=float_dtype,
                backend=backend,
            ),
            (0, 0),
        ),
        (
            from_array(
                np.random.randn(EdgeDim),
                dtype=float_dtype,
                backend=backend,
            ),
            (0,),
        ),
        (
            from_array(
                np.random.randn(EdgeDim),
                dtype=float_dtype,
                backend=backend,
            ),
            (0,),
        ),
        (z_nabla4_e2_wp_gtfn, (0, 0)),
        0,
        EdgeDim,
        0,
        KDim,
        (
            from_array(np.array(e2c2v), dtype=int_dtype, backend=backend),
            (0, 0),
        ),
        (
            from_array(np.array(e2ecv), dtype=int_dtype, backend=backend),
            (0, 0),
        ),
    )
    return runtimes


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--klevels", type=int, default=65, help="Number of k levels")
    parser.add_argument(
        "--repetitions", type=int, default=101, help="Number of repetitions"
    )
    parser.add_argument(
        "--dry-run", default=True, help="Do a dry run or not", action="store_true"
    )

    args = parser.parse_args()
    return args


def run_benchmarks():
    args = parse_arguments()

    repetitions = args.repetitions
    dry_runs = 10 if args.dry_run else 0

    num_edges = 902700

    filtered_e2c2v = pickle.load(open("torus_128_filtered_e2c2v.pickle", 'rb'))
    filtered_e2ecv = pickle.load(open("torus_128_filtered_e2ecv.pickle", 'rb'))

    assert len(filtered_e2c2v) == num_edges
    assert len(filtered_e2ecv) == num_edges

    runtimes = {}

    runtimes_gtfn = run_gtfn(
        repetitions,
        dry_runs,
        filtered_e2c2v,
        filtered_e2ecv,
        args.klevels,
        "gt:gpu",
    )
    runtimes["nabla4_benchmark_unstructured_gtfn_gpu"] = runtimes_gtfn

    print_median_runtimes(runtimes)

if __name__ == "__main__":
    run_benchmarks()

import numpy as np
import random
import argparse
import pinocchio as pin
import hppfcl
from pycolbench.utils import load_solvers_benchmark, get_distance_category, Results, ShapePair
from pycolbench.utils import FIELDS, FLAG_TIME, FLAG_NO_TIME, RESULTS_PATH
import pandas as pd
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance_category", dest="distance_category", default="close_proximity", type=str)
    parser.add_argument("--seed", dest="seed", default=0, type=int)
    parser.add_argument("--tolerance", dest="tolerance", default=1e-8, type=float)
    parser.add_argument("--num_poses", dest="num_poses", default=100, type=int)
    parser.add_argument("--num_pairs", dest="num_pairs", default=25, type=int)
    parser.add_argument("--measure_time", dest="measure_time", action="store_true")
    parser.add_argument("--outfile", dest="outfile", default="continuous_ellipsoids", type=str)

    args = parser.parse_args()

    print("\n--------------")
    print(f"Num pairs to test: {args.num_pairs}")
    print(f"Num poses to test: {args.num_poses}")
    print(f"Measure of execution time: {args.measure_time}")
    print("--------------")

    # -- SEED
    np.random.seed(args.seed)
    random.seed(args.seed)
    pin.seed(args.seed)

    # -- SCALES FOR THE ELLIPSOIDS
    scales = [1, 5, 10]

    # -- SOLVERS
    max_iterations = 1000
    cv_criterion = hppfcl.GJKConvergenceCriterion.DualityGap
    solvers = load_solvers_benchmark(max_iterations, args.tolerance, cv_criterion)

    # -- DISTANCES
    print(f"Distance category: {args.distance_category}")
    shape_distances = get_distance_category(args.distance_category)
    additional_key = str(args.distance_category) + "_" + str(args.seed)

    # -- CREATE RESULTS DICT
    P = args.num_pairs * args.num_poses * len(shape_distances)
    S = len(solvers)
    results = Results(num_problems=P, num_solvers=S)
    additional_key = str(args.seed)

    print("----------------")
    print(f"Total numbers of solvers to test: {S}")
    print(f"Total number of problem to solve: {P}")
    print("----------------")

    # -- MAIN LOOP
    npair = 0
    pbar = tqdm(total=args.num_pairs)
    while npair < args.num_pairs:
        try:
            # Choose r1 and r2 randomly (gaussian distrib)
            r1 = np.random.normal(0.1, 0.025, 3)
            r2 = np.random.normal(0.1, 0.025, 3)

            # Then we scale shape1 and shape2 randomly using the 'scales' list
            scale1 = random.choice(scales)
            r1 = scale1 * r1

            scale2 = random.choice(scales)
            r2 = scale2 * r2

            shape1 = hppfcl.Ellipsoid(r1[0], r1[1], r1[2])
            shape2 = hppfcl.Ellipsoid(r2[0], r2[1], r2[2])

            pair_id = npair
            shape_pair = ShapePair(shape1, shape2, pair_id)

            results.run_solvers_on_pair(shape_pair, solvers, args.num_poses, shape_distances,
                                        additional_key=additional_key,
                                        measure_time=args.measure_time)
            npair += 1
            pbar.update(1)
        except Exception as e:
            print(e)

    # SAVE CSV FILE
    print("Saving results...")
    df = pd.DataFrame(results.results, columns=[field.value for field in FIELDS])
    flag_time = FLAG_TIME if args.measure_time else FLAG_NO_TIME
    # Create results directory
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
    benchmark_path = os.path.join(RESULTS_PATH, "continuous_ellipsoids")
    if not os.path.exists(benchmark_path):
        os.mkdir(benchmark_path)
    # Save results
    output_path = os.path.join(benchmark_path, f"{args.outfile}_{flag_time}_{args.distance_category}_{args.tolerance}_{args.seed}.csv")
    df.to_csv(output_path, chunksize=1000, index=False)
    print("... Results saved.")

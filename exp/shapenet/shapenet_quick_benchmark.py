import os
import numpy as np
import pandas as pd
import random
import argparse
import pinocchio as pin
import hppfcl
from pycolbench.utils import collision_quick_benchmark, load_solvers
from pycolbench.utils import get_distance_category, load_convex_hull
from pycolbench.utils import SHAPENET_PATH


def get_list_of_shapes(num_pairs, min_num_points, max_num_points):
    shapenet_path = SHAPENET_PATH

    # Select thx to min/max number of points
    try:
        dddf = pd.read_csv(os.path.join(shapenet_path, "subshapenet.csv"))
    except Exception as e:
        print(e)
        print("exp/data/subshapenet.csv probably does not exist. To generate it, run `python exp/shapenet/generate_subshapenet.py`.")
        raise
    ddf = dddf[dddf["num_points"] >= min_num_points]
    df = ddf[ddf["num_points"] <= max_num_points]
    paths = df["path"].tolist()

    shapes0 = []
    shapes1 = []
    print("Loading shapes.")
    for i in range(num_pairs):
        # SHAPE0
        print(f"PAIR {i+1}/{num_pairs}")
        mesh0_path = random.choice(paths)
        mesh0_path = os.path.join(shapenet_path, mesh0_path)
        shape0 = load_convex_hull(mesh0_path)
        print(f"Shape0 num_points: {shape0.num_points}")
        shapes0.append(shape0)

        # SHAPE1
        mesh1_path = random.choice(paths)
        mesh1_path = os.path.join(shapenet_path, mesh1_path)
        shape1 = load_convex_hull(mesh1_path)
        print(f"Shape1 num_points: {shape1.num_points}")
        shapes1.append(shape1)
        print("---")
    return shapes0, shapes1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance_category", dest="distance_category", default="close_proximity", type=str)
    parser.add_argument("--seed", dest="seed", default=0, type=int)
    parser.add_argument("--tolerance", dest="tolerance", default=1e-8, type=float)
    parser.add_argument("--num_poses", dest="num_poses", default=100, type=int)
    parser.add_argument("--num_pairs", dest="num_pairs", default=25, type=int)
    parser.add_argument("--min_num_points", dest="min_num_points", default=100, type=int)
    parser.add_argument("--max_num_points", dest="max_num_points", default=1000, type=int)
    parser.add_argument("--measure_time", dest="measure_time", action="store_true")
    parser.add_argument("--python", dest="python", action="store_true")

    args = parser.parse_args()

    print("--------------")
    print(f"Num pairs to test: {args.num_pairs}")
    print(f"Num poses to test: {args.num_poses}")
    print(f"Measure of execution time: {args.measure_time}")
    print("--------------")

    # -- SEED
    np.random.seed(args.seed)
    random.seed(args.seed)
    pin.seed(args.seed)

    # -- SHAPES
    shapes1, shapes2 = get_list_of_shapes(args.num_pairs, args.min_num_points, args.max_num_points)

    # -- SOLVERS
    max_iterations = 1000
    cv_criterion = hppfcl.GJKConvergenceCriterion.DualityGap
    solvers = load_solvers(max_iterations, args.tolerance, cv_criterion, args.python)

    dists = get_distance_category(args.distance_category)
    print(f"Distances selected: {dists}")
    results = collision_quick_benchmark(solvers, shapes1, shapes2, dists, args.num_poses, measure_time=args.measure_time)

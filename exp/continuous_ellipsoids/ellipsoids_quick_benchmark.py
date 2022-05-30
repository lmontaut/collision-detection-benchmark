import numpy as np
import random
import argparse
import pinocchio as pin
import hppfcl
from pycolbench.utils import collision_quick_benchmark, load_solvers
from pycolbench.utils import get_distance_category

def get_list_of_shapes(num_pairs, scales=[1]):
    shapes0 = []
    shapes1 = []

    for _ in range(num_pairs):
        # Choose r0 and r1 randomly (gaussian distrib)
        r0 = np.random.normal(0.1, 0.025, 3)
        r1 = np.random.normal(0.1, 0.025, 3)

        # Then we scale shape0 and shape1 randomly using the 'scales' list
        scale0 = random.choice(scales)
        r0 = scale0 * r0

        scale1 = random.choice(scales)
        r1 = scale1 * r1

        shape0 = hppfcl.Ellipsoid(r0[0], r0[1], r0[2])
        shape1 = hppfcl.Ellipsoid(r1[0], r1[1], r1[2])

        shapes0.append(shape0)
        shapes1.append(shape1)
    return shapes0, shapes1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance_category", dest="distance_category", default="close_proximity", type=str)
    parser.add_argument("--seed", dest="seed", default=0, type=int)
    parser.add_argument("--tolerance", dest="tolerance", default=1e-8, type=float)
    parser.add_argument("--num_poses", dest="num_poses", default=100, type=int)
    parser.add_argument("--num_pairs", dest="num_pairs", default=25, type=int)
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

    # -- SCALES FOR THE ELLIPSOIDS
    scales = [1, 5, 10]

    # -- ELLIPSOIDS
    shapes1, shapes2 = get_list_of_shapes(args.num_pairs, scales)

    # -- SOLVERS
    max_iterations = 1000
    cv_criterion = hppfcl.GJKConvergenceCriterion.DualityGap
    solvers = load_solvers(max_iterations, args.tolerance, cv_criterion, args.python)

    # -- RUN
    dists = get_distance_category(args.distance_category)
    print(f"Distances selected: {dists}")
    results = collision_quick_benchmark(solvers, shapes1, shapes2, dists, args.num_poses, measure_time=args.measure_time)

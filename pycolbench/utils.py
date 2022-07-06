import numpy as np
import pinocchio as pin
import hppfcl
from hppfcl import GJKVariant, GJKConvergenceCriterion
from pycolbench.solvers import GJKSolverHPPFCL, GJKSolver
from typing import Any, List, Dict
import pandas as pd
from enum import Enum
import os


class FIELDS(Enum):
    """
    Contains all fields used to store information on a solver solving a collision problem.
    """
    # General info
    solver_name = "solver_name"
    problem_id = "problem_id"
    pose_id = "pose_id"
    pair_id = "pair_id"
    dist_shapes = "dist_shapes"
    scale1 = "scale1"
    scale2 = "scale2"
    dist_to_vanilla = "dist_to_vanilla"
    agree_with_vanilla = "agree_with_vanilla"
    normalize_dir = "normalize_dir"
    num_vertices_shape0 = "num_vertices_shape0"
    num_vertices_shape1 = "num_vertices_shape1"

    # Metrics
    numit = "numit"
    numit_rel = "numit_rel"
    numit_early = "numit_early"
    numit_early_rel = "numit_early_rel"

    num_call_support = "num_call_support"
    num_call_support_rel = "num_call_support_rel"
    num_call_support_early = "num_call_support_early"
    num_call_support_early_rel = "num_call_support_early_rel"

    cumulative_support_dotprod = "cumulative_support_dotprod"
    cumulative_support_dotprod_rel = "cumulative_support_dotprod_rel"
    cumulative_support_dotprod_early = "cumulative_support_dotprod_early"
    cumulative_support_dotprod_early_rel = "cumulative_support_dotprod_early_rel"

    mean_support_dotprod = "mean_support_dotprod"
    mean_support_dotprod_rel = "mean_support_dotprod_rel"
    mean_support_dotprod_early = "mean_support_dotprod_early"
    mean_support_dotprod_early_rel = "mean_support_dotprod_early_rel"

    num_call_projection = "num_call_projection"
    num_call_projection_rel = "num_call_projection_rel"
    num_call_projection_early = "num_call_projection_early"
    num_call_projection_early_rel = "num_call_projection_early_rel"

    execution_time = "execution_time"
    execution_time_rel = "execution_time_rel"
    execution_time_early = "execution_time_early"
    execution_time_early_rel = "execution_time_early_rel"


class SOLVER_NAMES(Enum):
    GJK = "GJK"
    Nesterov = "Nesterov"
    NesterovNormalized = "NesterovNormalized"


SOLVER_NAMES_PLOT: Dict[str, str] = {
    "GJK": "GJK",
    "Nesterov": "Nesterov",
    "NesterovNormalized": "Nesterov + normalization"
}

SOLVER_COLORS: Dict[str, str] = {
    "GJK": "green",
    "Nesterov": "orange",
    "NesterovNormalized": "red"
}


# Used for benchmarks, wether or not measure computation time
FLAG_TIME: str = "withtime"
FLAG_NO_TIME: str = "notime"

METRICS_WITHOUT_TIME: List[FIELDS] = [FIELDS.numit,
                                      FIELDS.numit_rel,
                                      FIELDS.numit_early,
                                      FIELDS.numit_early_rel,
                                      FIELDS.num_call_support,
                                      FIELDS.num_call_support_rel,
                                      FIELDS.num_call_support_early,
                                      FIELDS.num_call_support_early_rel,
                                      FIELDS.cumulative_support_dotprod,
                                      FIELDS.cumulative_support_dotprod_rel,
                                      FIELDS.cumulative_support_dotprod_early,
                                      FIELDS.cumulative_support_dotprod_early_rel,
                                      FIELDS.num_call_projection,
                                      FIELDS.num_call_projection_rel,
                                      FIELDS.num_call_projection_early,
                                      FIELDS.num_call_projection_early_rel,
                                      FIELDS.mean_support_dotprod,
                                      FIELDS.mean_support_dotprod_rel,
                                      FIELDS.mean_support_dotprod_early,
                                      FIELDS.mean_support_dotprod_early_rel]

METRICS_WITH_TIME: List[FIELDS] = [FIELDS.execution_time,
                                   FIELDS.execution_time_rel,
                                   FIELDS.execution_time_early,
                                   FIELDS.execution_time_early_rel]

# Path where results will be stored
RESULTS_PATH: str = "exp/results"
SHAPENET_PATH: str = "exp/shapenet/data"

def load_solvers_quick_benchmark(max_iterations: int, tolerance: float,
                 cv_criterion: GJKConvergenceCriterion, python: bool = False):
    solvers = []

    # Vanilla GJK
    solvers.append(GJKSolverHPPFCL(max_iterations, tolerance,
                                   gjk_variant = GJKVariant.DefaultGJK,
                                   cv_criterion = cv_criterion, name = "Default GJK"))

    if python:
        solvers.append(GJKSolver(max_iterations, tolerance,
                                 gjk_variant = GJKVariant.DefaultGJK,
                                 cv_criterion = cv_criterion,
                                 normalize_dir = False, name="Default GJK (python)"))

    # Nesterov accelerated GJK without normalization heuristic
    solvers.append(GJKSolverHPPFCL(max_iterations, tolerance,
                                   gjk_variant = GJKVariant.NesterovAcceleration,
                                   cv_criterion = cv_criterion,
                                   normalize_dir = False, name = "GJK + Nesterov"))

    if python:
        solvers.append(GJKSolver(max_iterations, tolerance,
                                 gjk_variant = GJKVariant.NesterovAcceleration,
                                 cv_criterion = cv_criterion,
                                 normalize_dir = False, name="GJK + Nesterov (python)"))

    # Nesterov accelerated GJK with normalization heuristic
    solvers.append(GJKSolverHPPFCL(max_iterations, tolerance,
                                   gjk_variant = GJKVariant.NesterovAcceleration,
                                   cv_criterion = cv_criterion,
                                   normalize_dir = True, name = "GJK + Nesterov + normalize"))

    if python:
        solvers.append(GJKSolver(max_iterations, tolerance,
                                 gjk_variant = GJKVariant.NesterovAcceleration,
                                 cv_criterion = cv_criterion,
                                 normalize_dir = True, name="GJK + Nesterov + normalize (python)"))
    return solvers

def load_solvers_benchmark(max_iterations: int, tolerance: float, cv_criterion: GJKConvergenceCriterion):
    solvers = []

    # Vanilla GJK
    solvers.append(GJKSolverHPPFCL(max_iterations, tolerance,
                                   gjk_variant = GJKVariant.DefaultGJK,
                                   cv_criterion = cv_criterion, name = SOLVER_NAMES.GJK.value))

    # Nesterov accelerated GJK without normalization heuristic
    solvers.append(GJKSolverHPPFCL(max_iterations, tolerance,
                                   gjk_variant = GJKVariant.NesterovAcceleration,
                                   cv_criterion = cv_criterion,
                                   normalize_dir = False, name = SOLVER_NAMES.Nesterov.value))

    # Nesterov accelerated GJK with normalization heuristic
    solvers.append(GJKSolverHPPFCL(max_iterations, tolerance,
                                   gjk_variant = GJKVariant.NesterovAcceleration,
                                   cv_criterion = cv_criterion,
                                   normalize_dir = True, name = SOLVER_NAMES.NesterovNormalized.value))
    return solvers


def create_results_dir() -> None:
    """
    Creates RESULTS_PATH directory if doesn't exist.
    """
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)


def load_and_concatenate_results(benchmark: str, distance_category: str, flag_time: str, additional_key: str = "") -> pd.DataFrame:
    """
    Load benchmark results for the selected distance category.
    flag_time is used to know wether to load `_notime_` or `_withtime_` cvs files.

    The idea is that `_notime_` files contain more collision problems than `_withtime_` files. 
    This is due to the fact that `_withtime_` benchmarks take a lot of time to run so we might want to compute time-related metrics
    on a lower amount of problems.
    """
    path_benchmark = os.path.join(RESULTS_PATH, benchmark)
    paths_ = os.listdir(path_benchmark)

    # -- Get the distance category only
    print(f"Loading results for benchmark: {benchmark}, distance category: {distance_category}...")
    paths = []
    for path in paths_:
        if path.count(distance_category) and path.count(flag_time) and path.count(additional_key):
            paths.append(os.path.join(path_benchmark, path))
    if len(paths) > 0:
        # -- Concat all the results. The metrics need to be computed on all problems for the distance category selected.
        dfs = [pd.read_csv(path) for path in paths]
        df = pd.concat([dfs[i] for i in range(len(dfs))])
        print("... Done.")
        return df
    else:
        print("No matching csv files found.")
        return None


def save_processed_results(df: pd.DataFrame, benchmark: str, additional_key: str = "") -> None:
    """
    Save a benchmark's processed results into csv files.
    """
    # -- Save these in csv file
    print("Saving metrics distribution...")
    processed_path = os.path.join(RESULTS_PATH, benchmark, "processed")
    if not os.path.exists(processed_path):
        os.mkdir(processed_path)
    output_path = os.path.join(processed_path, f"{benchmark}_{additional_key}.csv")
    df.to_csv(output_path, chunksize=1000, index=False)
    print("... Saved.")


class ShapePair:
    """
    Class to store info about a shape0/shape1 pair.
    """
    pair_id: Any
    shape0: hppfcl.ShapeBase
    shape1: hppfcl.ShapeBase
    scale0: float
    scale1: float

    def __init__(self, shape0: hppfcl.ShapeBase, shape1: hppfcl.ShapeBase, pair_id: Any,
                 scale1: float = 1, scale2: float = 1):
        self.shape0 = shape0
        self.shape1 = shape1
        self.pair_id = pair_id
        self.scale0 = scale1
        self.scale1 = scale2


class CollisionProblem:
    """
    A collision problem is defined by:
        - A pair of objects
        - A relative pose: a relative rotation and an axis (A) defined by line passing by
            the witness points of the two shapes for this relative pose.
        - A distance between the objects: shape1 can be translated along axis (A) in order
            to place it at desired distance to shape1.
            This allows to study the impact of distance on a given relative pose.
    """
    shape_pair: ShapePair
    pose_id: int
    dist_shapes: float
    problem_id: str

    def __init__(self, shape_pair: ShapePair, pose_id: int, dist_shapes: float,
                 additional_key: str = ""):
        self.shape_pair = shape_pair
        self.pose_id = pose_id
        self.dist_shapes = dist_shapes
        self.problem_id = str(shape_pair.pair_id) + "_" + str(self.pose_id) + "_" + str(dist_shapes) + "_" + str(additional_key)


class Results:
    """
    A class to store solver results on a collision problems.
    """
    results: Dict[str, List[Any]]
    num_problems: int
    num_solvers: int
    num_entries: int
    pose_id: int
    index: int
    num_errors: int

    def __init__(self, num_problems, num_solvers):
        self.num_problems = num_problems
        self.num_solvers = num_solvers
        self.num_entries = self.num_problems * self.num_solvers
        self.results = {}
        for metric in FIELDS:
            self.results[metric.value] = [None] * self.num_entries
        self.index = -1  # Used to access entries
        self.pose_id = -1  # Used to define unique collision problem ids
        self.num_diff_boolean_check = 0

    def run_solvers_on_pair(self, shape_pair: ShapePair, solvers: List[GJKSolverHPPFCL],
                            num_poses: int, shape_distances: List[float],
                            additional_key: str = "", measure_time: bool = False):
        """
        Runs all solvers on num_poses relative configurations * len(shape_distances) distances collision problems.

        The different benchmarks define pairs of shapes differently. In order to generate specific ids for each
        collision pair, we use:
            - The shapes pair id
            - The id of the pose
            - The distance between the shapes
            - An additional_key provided by the use, allows to run jobs in parallel.

        The option measure_time is optional. If selected each collision problem will be solved 100 times by a solver to get
        a mean execution time.
        """

        # Set shapes in solvers
        for solver in solvers:
            solver.set_shapes(shape_pair.shape0, shape_pair.shape1)

        # Main loop
        T1 = hppfcl.Transform3f(np.eye(3), np.zeros(3))
        npose = 0
        while npose < num_poses:
            T2 = generate_random_pose()
            res_relative_pose = solvers[0].evaluate(T1, T2)

            # Check if shapes are not in collision, otherwise we can't separate them
            # as separation_vector is [0, 0, 0]
            if not res_relative_pose:
                npose += 1
                self.pose_id += 1

                # Relative pose gives an axis on which we can translate the shapes
                separation_vector = solvers[0].ray
                n = separation_vector / np.linalg.norm(separation_vector)
                t = T2.getTranslation()
                for dist in shape_distances:
                    # In this loop, we are solving problem:
                    # `p = pair shape0-shape1 / pose_id / di`

                    # CURRENT COLLISION PROBLEM
                    # Get the shapes at dist from one another along axis n
                    dt = separation_vector - dist * n
                    newt = t + dt
                    T2 = hppfcl.Transform3f(T2.getRotation(), newt)
                    collision_problem = CollisionProblem(shape_pair, self.pose_id, dist, additional_key)

                    for nsolver in range(len(solvers)):
                        solver: GJKSolverHPPFCL = solvers[nsolver]
                        self.index += 1

                        if measure_time:
                            _ = solver.compute_execution_time(T1, T2)
                        else:
                            _ = solver.evaluate(T1, T2)

                        # Dump the metrics of this problem solved by this solver.
                        self.dump_results(solver, solvers[0], collision_problem)

    def dump_results(self, solver: GJKSolverHPPFCL, solver_vanilla: GJKSolverHPPFCL, collision_problem: CollisionProblem):
        """
        Dumps all the metrics (absolute and relative to vanilla GJK).
        """
        assert solver_vanilla.gjk_variant == hppfcl.GJKVariant.DefaultGJK  # Make sure we are comparing against vanilla GJK

        # Utils
        ndotprod = solver.support_cumulative_ndotprods / solver.numit if solver.numit > 0 else -1
        ndotprod_early = solver.support_cumulative_ndotprods_early / solver.numit_early if solver.numit_early > 0 else -1
        ndotprod_vanilla = solver_vanilla.support_cumulative_ndotprods / solver_vanilla.numit if solver_vanilla.numit > 0 else -1
        ndotprod_vanilla_early = solver_vanilla.support_cumulative_ndotprods_early / solver_vanilla.numit_early if solver_vanilla.numit_early > 0 else -1

        shape0 = collision_problem.shape_pair.shape0
        shape1 = collision_problem.shape_pair.shape1

        # Filling results
        # -- General data about solver and current problem
        self.results[FIELDS.solver_name.value][self.index] = solver.name
        self.results[FIELDS.problem_id.value][self.index] = collision_problem.problem_id
        self.results[FIELDS.pose_id.value][self.index] = self.pose_id
        self.results[FIELDS.pair_id.value][self.index] = collision_problem.shape_pair.pair_id
        self.results[FIELDS.dist_shapes.value][self.index] = collision_problem.dist_shapes
        self.results[FIELDS.normalize_dir.value][self.index] = solver.normalize_dir if isinstance(solver, GJKSolverHPPFCL) else False
        self.results[FIELDS.num_vertices_shape0.value][self.index] = shape0.num_points if isinstance(shape0, hppfcl.ConvexBase) else 0
        self.results[FIELDS.num_vertices_shape1.value][self.index] = shape1.num_points if isinstance(shape1, hppfcl.ConvexBase) else 0
        self.results[FIELDS.scale1.value][self.index] = collision_problem.shape_pair.scale0
        self.results[FIELDS.scale2.value][self.index] = collision_problem.shape_pair.scale1

        # -- Agreement with vanilla GJK
        # NOTE: if the distance btw shapes is close to GJK tolerance, it is expected to have discrepancy between Nesterov-accelerated and Vanilla GJK.
        # This is expected as both solvers only approach the exact solution up to sqrt(epsilon).
        self.results[FIELDS.dist_to_vanilla.value][self.index] = np.linalg.norm(solver.ray - solver_vanilla.ray)
        self.results[FIELDS.agree_with_vanilla.value][self.index] = 1 - (solver.res - solver_vanilla.res)

        # -- Number of iterations
        self.results[FIELDS.numit.value][self.index] = solver.numit
        self.results[FIELDS.numit_rel.value][self.index] = solver_vanilla.numit / solver.numit if solver.numit > 0 else -1
        self.results[FIELDS.numit_early.value][self.index] = solver.numit_early
        self.results[FIELDS.numit_early_rel.value][self.index] = solver_vanilla.numit_early / solver.numit_early if solver.numit_early > 0 else -1

        # -- Support related
        self.results[FIELDS.num_call_support.value][self.index] = solver.num_call_support
        self.results[FIELDS.num_call_support_rel.value][self.index] = solver_vanilla.num_call_support / solver.num_call_support if solver.num_call_support > 0 else -1
        self.results[FIELDS.num_call_support_early.value][self.index] = solver.num_call_support_early
        self.results[FIELDS.num_call_support_early_rel.value][self.index] = solver_vanilla.num_call_support_early / solver.num_call_support_early if solver.num_call_support_early > 0 else -1

        self.results[FIELDS.cumulative_support_dotprod.value][self.index] = solver.support_cumulative_ndotprods
        self.results[FIELDS.cumulative_support_dotprod_rel.value][self.index] = solver_vanilla.support_cumulative_ndotprods / solver.support_cumulative_ndotprods if solver.support_cumulative_ndotprods > 0 else -1
        self.results[FIELDS.cumulative_support_dotprod_early.value][self.index] = solver.support_cumulative_ndotprods_early
        self.results[FIELDS.cumulative_support_dotprod_early_rel.value][self.index] = solver_vanilla.support_cumulative_ndotprods_early / solver.support_cumulative_ndotprods_early if solver.support_cumulative_ndotprods_early > 0 else -1

        self.results[FIELDS.mean_support_dotprod.value][self.index] = ndotprod
        self.results[FIELDS.mean_support_dotprod_rel.value][self.index] = ndotprod_vanilla / ndotprod if ndotprod > 0 else -1
        self.results[FIELDS.mean_support_dotprod_early.value][self.index] = ndotprod_early
        self.results[FIELDS.mean_support_dotprod_early_rel.value][self.index] = ndotprod_vanilla_early / ndotprod_early if ndotprod_early > 0 else -1
        # ------------------------

        # -- Projection related
        self.results[FIELDS.num_call_projection.value][self.index] = solver.num_call_projection
        self.results[FIELDS.num_call_projection_rel.value][self.index] = solver_vanilla.num_call_projection / solver.num_call_projection if solver.num_call_projection > 0 else solver_vanilla.num_call_projection
        self.results[FIELDS.num_call_projection_early.value][self.index] = solver.num_call_projection_early
        self.results[FIELDS.num_call_projection_early_rel.value][self.index] = solver_vanilla.num_call_projection_early / solver.num_call_support_early if solver.num_call_projection_early > 0 else solver_vanilla.num_call_projection_early
        # ------------------------

        # -- Execution time related
        self.results[FIELDS.execution_time.value][self.index] = solver.mean_execution_time
        self.results[FIELDS.execution_time_rel.value][self.index] = solver_vanilla.mean_execution_time / solver.mean_execution_time if solver.mean_execution_time > 0 else -1
        self.results[FIELDS.execution_time_early.value][self.index] = solver.mean_execution_time_early
        self.results[FIELDS.execution_time_early_rel.value][self.index] = solver_vanilla.mean_execution_time_early / solver.mean_execution_time_early if solver.mean_execution_time_early > 0 else -1
        # ------------------------


allowed_distances = ["distant", "close_proximity", "overlapping"]


def get_distance_category(distance_category: str):
    """
    Distance categories: distant, close-proximity, overlapping.
    """
    assert distance_category in allowed_distances,\
        print(f"Allowed distances: {allowed_distances}. selected: {distance_category}")
    if distance_category == "distant":
        distances = [5., 2.5, 1., 0.5]
    if distance_category == "close_proximity":
        distances = [1e-1, 1e-2, 1e-3, 1e-4]
    if distance_category == "overlapping":
        distances = [-1e-1, -1e-2, -1e-3, -1e-4]
    return distances


def load_convex_hull(path: str):
    """
    Returns convex hull of mesh on path.
    """
    assert path is not None
    loader = hppfcl.MeshLoader()
    mesh: hppfcl.BVHModelOBB = loader.load(path)
    _ = mesh.buildConvexHull(True, "Qt")
    return mesh.convex


def generate_random_pose():
    """
    Generate a random rotation matrix. For the position, we generate a vector
    on the surface of the norm sphere. We scale it to ensure the shapes are not
    in collision.
    """
    R = pin.SE3.Random().rotation
    w = pin.log3(pin.SE3.Random().rotation)
    w = w / np.linalg.norm(w)
    t = 3 * w
    return hppfcl.Transform3f(R, t)


METRICS_TO_PRINT_NO_TIME: List[FIELDS] = [FIELDS.dist_to_vanilla,
    FIELDS.numit,
    FIELDS.numit_rel,
    FIELDS.numit_early,
    FIELDS.numit_early_rel]
                                

METRICS_TO_PRINT_WITH_TIME: List[FIELDS] = [FIELDS.execution_time,
    FIELDS.execution_time_rel,
    FIELDS.execution_time_early,
    FIELDS.execution_time_early_rel]

def collision_quick_benchmark(solvers: List[GJKSolverHPPFCL],
                                       shapes0: List[hppfcl.ShapeBase],
                                       shapes1: List[hppfcl.ShapeBase],
                                       dists: List[float], num_poses: int, seed=0,
                                       measure_time=False):
    """
    solvers must be a dict of CollisionSolverBase objects.
    shapes0 and shapes1 must be lists of shapes.
    """
    assert len(shapes0) == len(shapes1), "shapes0 and shapes1 must be of same length."

    P = num_poses * len(shapes0) * len(dists)
    S = len(solvers)
    results = Results(num_problems=P, num_solvers=S)
    print("----------------")
    print(f"Total numbers of solvers to test: {S}")
    print(f"Total number of problem to solve: {P}")
    print("----------------")

    for si in range(len(shapes0)):
        shape0 = shapes0[si]
        shape1 = shapes1[si]
        pair_id = si
        shape_pair = ShapePair(shape0, shape1, pair_id)

        additional_key = str(seed)
        results.run_solvers_on_pair(shape_pair, solvers, num_poses, dists, additional_key=additional_key, measure_time=measure_time)
    results = pd.DataFrame(results.results, columns=[field.value for field in FIELDS])

    print(f"Results computed on {P} problems.")
    for metric in FIELDS:
        if metric in METRICS_TO_PRINT_NO_TIME or (measure_time and metric in METRICS_TO_PRINT_WITH_TIME):
            print("----------------")
            print(metric.value)
            print("----------------")
            for solver in solvers:
                res = results[results[FIELDS.solver_name.value] == solver.name][metric.value]
                print(f"--> {solver.name}:")
                if metric != FIELDS.dist_to_vanilla:
                    printMeanStd(res, round_res=True)
                    printMinMaxMedian(res, round_res=True)
                else:
                    printMeanStd(res, round_res=False)
                    printMinMaxMedian(res, round_res=False)
            print("")
    return results


def printMeanStd(arr: pd.DataFrame, round_res=False):
    if len(arr) > 0:
        mean = arr.mean()
        std = arr.std()
        if round_res:
            mean = round(mean, 2)
            std = round(std, 2)
            print(f"\tMean: {mean} +- {std}")
        else:
            print("\tMean: {:.2e} +- {:.2e}".format(mean, std))
    else:
        print(f"\tCan't compute mean/std. Array is likely empty. len of array: {len(arr)}")


def printMinMaxMedian(arr: pd.DataFrame, round_res=False):
    if len(arr) > 0:
        min_arr = arr.min()
        max_arr = arr.max()
        median = arr.quantile(0.5)
        quartile1 = arr.quantile(0.25)
        quartile2 = arr.quantile(0.75)
        decile1 = arr.quantile(0.10)
        decile2 = arr.quantile(0.90)
        if round_res:
            min_arr = round(min_arr, 2)
            max_arr = round(max_arr, 2)
            median = round(median, 2)
            quartile1 = round(quartile1, 2)
            quartile2 = round(quartile2, 2)
            decile1 = round(decile1, 2)
            decile2 = round(decile2, 2)
            print(f"\tMin: {min_arr} / Max: {max_arr}")
            print(f"\tMedian: {median} / Q1: {quartile1} / Q3: {quartile2} / D1: {decile1} / D10: {decile2}")
        else:
            print("\tMin: {:.2e} / Max: {:.2e}".format(min_arr, max_arr))
            print("\tMedian: {:.2e} / Q1: {:.2e} / Q3: {:2e} / D1: {:2e} / D10: {:2e}".format(median, quartile1, quartile2, decile1, decile2))
    else:
        print(f"\tCan't compute min/max/median. Array is likely empty. len of array: {len(arr)}")

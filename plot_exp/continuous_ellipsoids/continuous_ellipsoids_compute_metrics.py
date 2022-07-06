import pandas as pd
from pycolbench.utils import FIELDS, FLAG_NO_TIME
from pycolbench.utils import load_and_concatenate_results, save_processed_results
from typing import Dict, Any

def compute_mean_std_numit(solver_results: Dict[str, Any], distance_category) -> pd.DataFrame:
    distances = solver_results[FIELDS.dist_shapes.value].unique()
    if distance_category == "overlapping":
        distances = distances[::-1]
    mean = [0 for _ in range(len(distances))]
    mean_early = [0 for _ in range(len(distances))]
    std = [0 for _ in range(len(distances))]
    std_early = [0 for _ in range(len(distances))]
    for di, distance in enumerate(distances):
        mean[di] = solver_results[solver_results[FIELDS.dist_shapes.value] == distance][FIELDS.numit.value].mean()
        mean_early[di] = solver_results[solver_results[FIELDS.dist_shapes.value] == distance][FIELDS.numit_early.value].mean()

        std[di] = solver_results[solver_results[FIELDS.dist_shapes.value] == distance][FIELDS.numit.value].std()
        std_early[di] = solver_results[solver_results[FIELDS.dist_shapes.value] == distance][FIELDS.numit_early.value].std()

    results = {}
    results[FIELDS.dist_shapes.value] = distances
    results["mean"] = mean
    results["mean_early"] = mean_early
    results["std"] = std
    results["std_early"] = std_early
    columns = [FIELDS.dist_shapes.value, "mean", "mean_early", "std", "std_early"]
    df = pd.DataFrame(results, columns=columns)
    return df


if __name__ == '__main__':
    dist_categories = ["close_proximity", "overlapping" , "distant"]

    benchmark = "continuous_ellipsoids"
    flag_time = FLAG_NO_TIME
    for dist_category in dist_categories:
        df = load_and_concatenate_results(benchmark, dist_category, flag_time)

        # -- Get solvers results for numit
        if df is not None:
            solver_names = df[FIELDS.solver_name.value].unique()
            print(f"Number of line in df: {len(df)}")
            print(f"Number of solvers: {len(solver_names)}")
            for solver_name in solver_names:
                print(solver_name)
            problems = df[FIELDS.problem_id.value].unique()
            print(f"Number of collision problems: {len(problems)}")

            for solver_name in solver_names:
                # -- Compute metrics distributions for each solver
                print(f"Computing metrics distributions on solver {solver_name}...")
                df_metric = compute_mean_std_numit(df[df[FIELDS.solver_name.value] == solver_name], dist_category)
                print("Done.")

                additional_key = solver_name + "_" + dist_category + "_numit"
                save_processed_results(df_metric, benchmark, additional_key=additional_key)

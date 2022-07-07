# Collision Detection Benchmarks [WIP]
This repo contains the benchmarks for the paper "Collision Detection Accelerated: An Optimization Perspective" published at RSS 2022.
You can find the paper [here](https://arxiv.org/abs/2205.09663) and the project page [here](https://lmontaut.github.io/nesterov-gjk.github.io/).
There are two main benchmarks: the **ellipsoid** benchmark (strictly-convex shapes) and the **convex mesh** benchmark (non-strictly convex shapes), which are intended to compare the GJK algorithm and our method: Nesterov accelerated GJK.

These benchmarks call the HPPFCL C++ library in which both GJK and Nesterov-accelerated GJK are implemented.

For prototyping, we have also reimplemented GJK and Nesterov-accelerated GJK in Python.

# Installation
To make the install easy, we recommend using conda to isolate the required packages needed to run the benchmarks from your system.
- Clone this repo: `git clone --recursive https://github.com/lmontaut/collision-detection-benchmark.git && cd collision-detection-benchmark`
- Install conda and a new conda environment: `conda create -n collision_benchmark python=3.8 && conda activate collision_benchmark`
- Install dependencies: `conda install cmake pinocchio pandas tqdm qhull`. For `pinocchio`, add the `conda-forge` channel `conda config --add channels conda-forge`.
- Re-activate the conda env `conda activate collision_benchmark` for the cmake path to take effect.
- Install hppfcl:
  - `mkdir hpp-fcl/build && cd hpp-fcl/build`
  - `git submodule update --init`
  - `cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release -DHPP_FCL_HAS_QHULL=ON ..`
  - `make install` 
- Go back to the root of this repo (`cd ../..` if you are in `hpp-fcl/build`) and install this python library on the conda env: `pip install -e .`

This was succesfully installed and tested on Manjaro `5.15.50` and Ubuntu `20.04`.
The tested compilers were g++ version `9.4.0` and `12.1.0` and clang++ version `13.0.1`
The required version for eigen is `3.4.0`.

# ShapeNet download
Please visit https://shapenet.org.
Download `ShapeNetCore.v2` and place it in `exp/shapenet/data`.

To generate a subset of ShapeNet to run the benchmarks, run `python exp/shapenet/generate_subshapenet.py`

# Quick benchmarks:
To launch a quick benchmark:
- Ellipsoids: `python exp/continuous_ellipsoids/ellipsoids_quick_benchmark.py [--opts]`
- Meshes: `python exp/shapenet/shapenet_quick_benchmark.py [--opts]`.

The param `--opts` can be:
- `--python`: also runs the quick benchmark with the solvers written in Python, off by default
- `--measure_time`: measures execution times, off by default
- `--distance_category`: overlapping, close-proximity, distant
- `--num_pairs`: number of collision pairs
- `--num_poses`: number of relative poses btw each collision pair

To compare the performances between Nesterov-accelerated GJK and vanilla GJK, we measure both the performance on boolean collision detection and distance computation.
- For distance computation, both algorithms run until they have computed the distance which separates the shapes. This is measured by the Frank-Wolfe duality-gap reaching a certain tolerance; please read the paper for more info. 
- For boolean collision check, both algos stop as soon as they find a separating hyperplane between the shapes or when a point inside their intersection has been found.

We thus measure the following metrics for distance computation:
- `dist_to_vanilla`: distance of the solution found by the solver to the solution found by vanilla GJK.
- `numit`: number of iterations to converge.
- `execution_time`
- the suffix `rel` relates to the relative performance to vanilla GJK. Given a solver, a metric and a collision problem, we do `metric of GJK on problem P / metric of solver on problem P`.
We add the suffix `early` to `numit` and `execution_time` to track the performance of the boolean collision check (`early` because boolean collision check is an early stop of distance computation).

# Large benchmarks:
The plots from the paper where obtained from the following benchmarks.
You will need to have `pandas` to save results to `.csv` files and `jupyter` to plot the results: `conda install pandas jupyterlab`

### 1 - Ellipsoids
- Launch the benchmark: `./ellipsoids_benchmark.sh`
- View the results: `jupyter lab` then go to `plot_exp/continuous_ellipsoids/continuous_ellipsoids_plots.ipynb` and run the notebook.

# Citing this repo
To cite Nesterov accelerated GJK and/or the associated benchmarks, please use the following bibtex lines:
```bibtex
@inproceedings{montaut2022GJKNesterov,
  title = {Collision Detection Accelerated: An Optimization Perspective},
  author = {Montaut, Louis and Le Lidec, Quentin and Petrik, Vladimir and Sivic, Josef and Carpentier, Justin},
  booktitle = {Robotics: Science and Systems},
  year = {2022}
}
```

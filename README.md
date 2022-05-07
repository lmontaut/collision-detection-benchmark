# Collision Detection Benchmarks [WIP]
This repo contains the benchmarks for the paper "Collision Detection Accelerated: An Optimization Perspective" published at RSS 2022.
There are two main benchmarks: the **ellipsoid** benchmark (strictly-convex shapes) and the **convex mesh** benchmark (non-strictly convex shapes), which are intended to compare the GJK algorithm and our method: Nesterov accelerated GJK.

These benchmarks call the HPPFCL C++ library in which both GJK and Nesterov-accelerated GJK are implemented.

For prototyping, we have also reimplemented GJK and Nesterov-accelerated GJK in Python.

# Installation
To make the install easy, we recommend using conda to isolate the required packages needed to run the benchmarks from your system.
- Clone this repo: `git clone --recursive https://github.com/lmontaut/collision-detection-benchmark.git && cd collision-detection-benchmark`
- Install conda and a new conda environment: `conda create -n collision_benchmark python=3.8 && conda activate collision_benchmark`
- Install dependencies: `conda install cmake pinocchio pandas tqdm`. For `pinocchio`, add the `conda-forge` channel `conda config --add channels conda-forge`.
- Re-activate the conda env `conda activate collision_benchmark` for the cmake path to take effect.
- Install hppfcl:
  - `mkdir hpp-fcl/build && cd hpp-fcl/build`
  - `git submodule update --init`
  - `cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release -DHPP_FCL_HAS_QHULL=ON ..`
  - `make install` 
- Install this python library on the conda env: `pip install -e .`

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

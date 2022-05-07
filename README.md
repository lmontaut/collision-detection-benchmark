# Collision Detection Benchmarks
This repo contains the benchmarks for the paper "Collision Detection Accelerated: An Optimization Perspective" published at RSS 2022.
There are two main benchmarks: the **ellipsoid** benchmark (strictly-convex shapes) and the **convex mesh** benchmark (non-strictly convex shapes), which are intended to compare the GJK algorithm and our method: Nesterov accelerated GJK.

These benchmarks call the HPPFCL C++ library in which both GJK and Nesterov-accelerated GJK are implemented.

For prototyping, we have also reimplemented GJK and Nesterov-accelerated GJK in Python.

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

import numpy as np
import hppfcl
from hppfcl import GJKVariant, ConvergenceCriterion
from pycolbench.simplexes import newSimplex, newVertex, copySimplex


class CollisionSolverBase(object):
    tolerance: float = 1e-8
    num_call_support: int = 0
    num_call_support_early: int = 0
    support_cumulative_ndotprods: int = 0
    support_cumulative_ndotprods_early: int = 0
    num_call_projection: int = 0
    num_call_projection_early: int = 0
    numit: int = 0
    numit_early: int = 0
    gjk_time: float = 0.
    gjk_time_early: float = 0.
    mean_execution_time: float = 0.
    mean_execution_time_early: float = 0.
    ray: np.ndarray = np.zeros(3)
    name: str = "NO_NAME"
    shape0: hppfcl.ShapeBase
    c0_loc: np.ndarray
    shape1: hppfcl.ShapeBase
    c1_loc: np.ndarray
    mink_diff: hppfcl.MinkowskiDiff
    res: bool

    def __init__(self, max_iterations: int, tolerance: float,
                 shape0: hppfcl.ShapeBase, shape1: hppfcl.ShapeBase,
                 name: str = "NO_NAME"):
        """
        Base collision solver class.
        """
        self.name = name
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.set_shapes(shape0, shape1)

    def set_shapes(self, shape0: hppfcl.ShapeBase, shape1: hppfcl.ShapeBase):
        """
        Set shapes which the solver will use for collision problems.
        """
        if isinstance(shape0, hppfcl.ShapeBase):
            self.shape0 = shape0
            self.shape0.computeLocalAABB()
            self.c0_loc = self.shape0.aabb_center
        else:
            self.shape0 = None
            self.c0_loc = None

        if isinstance(shape1, hppfcl.ShapeBase):
            self.shape1 = shape1
            self.shape1.computeLocalAABB()
            self.c1_loc = self.shape1.aabb_center
        else:
            self.shape1 = None
            self.c1_loc = None

    def reset_metrics(self):
        """
        Reset metrics before solving a problem.
        """
        self.num_call_support = 0
        self.num_call_support_early = 0
        self.support_cumulative_ndotprods = 0
        self.support_cumulative_ndotprods_early = 0
        self.num_call_projection = 0
        self.num_call_projection_early = 0
        self.numit = 0
        self.numit_early = 0
        self.gjk_time = 0.
        self.gjk_time_early = 0.
        self.mean_execution_time = 0.
        self.mean_execution_time_early = 0.
        self.res = False

    def update_metrics(self):
        """
        Updates the metrics (num call support, projection, execution time etc.)
        """
        pass

    def reset(self, T0: hppfcl.Transform3f, T1: hppfcl.Transform3f):
        """
        Resets the solver, given 2 transformations for shape0 and shape1:
            - reset metrics
            - reset Minkowski difference
            - computes initial guess: center bbox shape0 - center bbox shape1
        """
        # Reset metrics
        self.reset_metrics()

        # Init mink diff
        self.mink_diff = hppfcl.MinkowskiDiff()
        self.mink_diff.set(self.shape0, self.shape1, T0, T1)

        # Init ray
        self.c0 = T0.transform(self.c0_loc)
        self.c1 = T1.transform(self.c1_loc)
        self.ray = self.c0 - self.c1 # Initalisation of algorithm

    def evaluate(self, T0: hppfcl.Transform3f, T1: hppfcl.Transform3f):
        pass


class GJKSolverHPPFCL(CollisionSolverBase):
    gjk_solver: hppfcl.GJK
    gjk_variant: GJKVariant = GJKVariant.DefaultGJK
    x0: np.ndarray = np.zeros(3)
    x1: np.ndarray = np.zeros(3)

    def __init__(self, max_iterations: int, tolerance: float,
                 shape0: hppfcl.ShapeBase = None, shape1: hppfcl.ShapeBase = None,
                 gjk_variant: GJKVariant = GJKVariant.DefaultGJK,
                 cv_criterion: ConvergenceCriterion = ConvergenceCriterion.DG,
                 normalize_dir: bool = False, name: str = "NO_NAME",
                 compute_closest_points: bool = False):
        super().__init__(max_iterations, tolerance, shape0, shape1, name)
        self.gjk_variant = gjk_variant
        self.normalize_dir = normalize_dir
        self.cv_criterion = cv_criterion
        self.compute_closest_points = compute_closest_points

    def update_metrics(self):
        self.numit = self.gjk_solver.getIterations()
        self.numit_early = self.gjk_solver.getIterationsEarly()
        self.num_call_support = self.gjk_solver.getNumCallSupport()
        self.num_call_support_early = self.gjk_solver.getNumCallSupportEarly()
        self.num_call_projection = self.gjk_solver.getNumCallProjection()
        self.num_call_projection_early = self.gjk_solver.getNumCallProjectionEarly()
        self.support_cumulative_ndotprods = self.gjk_solver.getCumulativeSupportDotprods()
        self.support_cumulative_ndotprods_early = self.gjk_solver.getCumulativeSupportDotprodsEarly()
        self.gjk_time = self.gjk_solver.getGJKRunTime().user
        self.gjk_time_early = self.gjk_solver.getGJKRunTimeEarly().user

    def evaluate(self, T0: hppfcl.Transform3f, T1: hppfcl.Transform3f):
        self.reset(T0, T1)

        # Reset & set up and run the C++ solver
        self.gjk_solver = hppfcl.GJK(self.max_iterations, self.tolerance)
        self.gjk_solver.setGJKVariant(self.gjk_variant)
        self.gjk_solver.setConvergenceCriterion(self.cv_criterion)
        self.gjk_solver.setNormalizeSupportDirection(self.normalize_dir)
        # Hint = index for first support call for hill climbing (indices of vertex)
        # By default hint = [0, 0] so this line is useless but left for reference
        hint = np.array([0, 0], dtype=np.int32)

        res = self.gjk_solver.evaluate(self.mink_diff, self.ray, hint)

        # Metrics
        self.update_metrics()
        self.ray = self.gjk_solver.ray

        # Witness point when shapes are not in collision
        if self.compute_closest_points:
            self.gjk_solver.computeClosestPoints()
            self.x0 = self.gjk_solver.x0
            self.x1 = self.gjk_solver.x1

        # Boolean collision result
        if res == hppfcl.GJKStatus.Inside:
            self.res = True
        else:
            self.res = False
        return self.res

    def compute_execution_time(self, T0: hppfcl.Transform3f, T1: hppfcl.Transform3f):
        """
        Average computation time over 100 runs on the same problem.
        """
        self.reset(T0, T1)
        # Reset & set up and run the C++ solver
        self.gjk_solver = hppfcl.GJK(self.max_iterations, self.tolerance)
        self.gjk_solver.setGJKVariant(self.gjk_variant)
        self.gjk_solver.setConvergenceCriterion(self.cv_criterion)
        self.gjk_solver.setNormalizeSupportDirection(self.normalize_dir)
        hint = np.array([0, 0], dtype=np.int32)
        self.gjk_solver.measureRunTime()
        self.gjk_solver.computeGJKAverageRunTime(self.mink_diff, self.ray, hint)

        # Update metrics
        self.update_metrics()
        self.ray = self.gjk_solver.ray
        self.mean_execution_time = self.gjk_solver.getAverageGJKRunTime()
        self.mean_execution_time_early = self.gjk_solver.getAverageGJKRunTimeEarly()


class GJKSolver(CollisionSolverBase):
    gjk_solver: hppfcl.GJK
    gjk_variant: GJKVariant
    current_gjk_variant: GJKVariant = GJKVariant.DefaultGJK
    momentum: float = -1.

    x0: np.ndarray = np.zeros(3)
    x1: np.ndarray = np.zeros(3)

    def __init__(self, max_iterations: int, tolerance: float,
                 shape0: hppfcl.ShapeBase = None, shape1: hppfcl.ShapeBase = None,
                 gjk_variant: GJKVariant = GJKVariant.DefaultGJK,
                 cv_criterion: ConvergenceCriterion = ConvergenceCriterion.DG,
                 normalize_dir: bool = False,
                 name: str = "NO_NAME", verbose: bool = False,
                 compute_closest_points: bool = False):
        super().__init__(max_iterations, tolerance, shape0, shape1, name)
        self.gjk_variant = gjk_variant
        self.normalize_dir = normalize_dir
        self.cv_criterion = cv_criterion
        self.verbose = verbose
        self.compute_closest_points = compute_closest_points

    def reset(self, T0: hppfcl.Transform3f, T1: hppfcl.Transform3f):
        """
        Resets the solver to accept a new (T0, T1) tuple.
        Only resets the initial ray if the warm_start flag is False.
        """
        super().reset(T0, T1)

        # Used for projections on simplices
        self.gjk_solver = hppfcl.GJK(self.max_iterations, self.tolerance)

        # hppfcl = hppfcl GJK / duality = GJK with regular duality gap criterion
        self.duality_gap = float('inf')
        self.omega = -1.
        self.alpha = 0.

        # Flags
        self.flag_end = False
        self.flag_inside = False

        # Init ray (point belonging to current simplex, closest point to the origin found so far)
        self.ray_norm = np.linalg.norm(self.ray)
        self.y = self.ray
        self.dir = self.ray

        # 2 simplices
        self.current_s = newSimplex()
        self.next_s = newSimplex()
        self.s0, self.s1, self.s = np.zeros(3), np.zeros(3), self.ray
        self.v = newVertex(self.s0, self.s1, self.s)  # Last added vertex

        # Reset acceleration
        self.current_gjk_variant = self.gjk_variant

        # x = solver solution, x0 & x1 = shapes closest points
        self.x = self.ray
        self.ray_weigths = np.zeros(4)
        self.x0 = self.c0
        self.x1 = self.c1

        # For early stopping
        self.found_positive_lower_bound = False
        self.flag_switch_to_gjk = False

    def cv_check_A(self):
        if self.ray_norm < self.tolerance:
            self.flag_inside = True
            self.flag_end = True

    def compute_dir(self):
        if self.current_gjk_variant == GJKVariant.DefaultGJK:
            self.y = self.ray
            self.dir = self.ray

        # Nesterov acceleration
        if self.current_gjk_variant == GJKVariant.NesterovAcceleration:
            # Normalization heuristic. Detrimental for pairs of strictly convex shapes but good otherwise.
            if self.normalize_dir:
                self.momentum = (self.numit + 2) / (self.numit + 3)
                self.y = self.momentum * self.ray + (1 - self.momentum) * self.v.w
                self.dir = self.momentum * self.dir / np.linalg.norm(self.dir) + (1 - self.momentum) * self.y / np.linalg.norm(self.y)
            else:
                self.momentum = (self.numit + 1) / (self.numit + 3)
                self.y = self.momentum * self.ray + (1 - self.momentum) * self.v.w
                self.dir = self.momentum * self.dir + (1 - self.momentum) * self.y

    def get_support(self, dir):
        """
        Get support in direction -dir, dir needs to be normalized.
        """
        s0 = self.mink_diff.support0(dir, True)
        s1 = self.mink_diff.support1(- dir, True)
        s = s0 - s1
        i0 = self.mink_diff.index_support0 # index for hill-climbing
        i1 = self.mink_diff.index_support1
        return s0, s1, s, i0, i1

    def get_support_and_create_vertex(self):
        s0, s1, s, i0, i1 = self.get_support(-self.dir)
        self.num_call_support += 1

        self.v = newVertex(s0, s1, s, i0, i1)
        self.s0, self.s1, self.s = s0, s1, s

    def check_early_stopping(self):
        """
        Has a separating hyperplane been found ?
        """
        if not self.found_positive_lower_bound:
            self.numit_early = self.numit + 1
            self.num_call_support_early = self.num_call_support
            self.num_call_projection_early = self.num_call_projection

        self.omega = (self.dir @ self.v.w) / np.linalg.norm(self.dir)
        if self.omega > 0:
            self.found_positive_lower_bound = True

    def cv_check_B(self):
        """
        Checks the frank-wolfe duality_gap: 2 * x_k^T(x_k - s_k).
        The default hppfcl cv criterion is also implemented for reference.
        """
        # Duality gap
        if self.cv_criterion == ConvergenceCriterion.DG:
            self.duality_gap = 2 * self.ray @ (self.ray - self.v.w)
            check_passed = (self.duality_gap - self.tolerance) <= 0
        elif self.cv_criterion == ConvergenceCriterion.DG_RELATIVE:
            self.duality_gap = 2 * self.ray @ (self.ray - self.v.w)
            check_passed = ((self.duality_gap / (self.tolerance * self.ray_norm)) - self.tolerance * self.ray_norm) <= 0

        # Default hppfcl criterion
        elif self.cv_criterion == ConvergenceCriterion.DefaultCV:
            self.alpha = max(self.alpha, self.omega)
            self.duality_gap = self.ray_norm - self.alpha
            check_passed = (self.duality_gap - self.tolerance * self.ray_norm) <= 0

        # Duality gap + default hppfcl criterion
        elif self.cv_criterion == ConvergenceCriterion.IDG:
            self.alpha = max(self.alpha, self.omega)
            self.duality_gap = self.ray_norm * self.ray_norm - self.alpha * self.alpha
            check_passed = (self.duality_gap - self.tolerance) <= 0
        elif self.cv_criterion == ConvergenceCriterion.IDG_RELATIVE:
            self.alpha = max(self.alpha, self.omega)
            self.duality_gap = self.ray_norm * self.ray_norm - self.alpha * self.alpha
            check_passed = ((self.duality_gap / (self.tolerance * self.ray_norm)) - self.tolerance * self.ray_norm) <= 0

        else:
            print("INVALID CV CRITERION")
            return False

        if self.verbose:
            print(f"{self.name} - duality gap: {self.duality_gap}")

        if self.numit > 0:
            if check_passed:
                if self.current_gjk_variant != GJKVariant.DefaultGJK:
                    self.flag_switch_to_gjk = True
                else:
                    self.flag_end = True
                    if self.ray_norm < self.tolerance:
                        self.flag_inside = True
                    if self.verbose:
                        print(f"{self.name}:")
                        print(f"\tDuality gap: {self.duality_gap}")
                        print(f"\tCurrent estimate of distance btw shapes: {np.linalg.norm(self.ray)}")
                        print(f"\tCurrent iterate: {self.ray}")
                        print(f"\tSimplex rank: {self.current_s.rank}")

    def update_current_simplex(self):
        """
        Updates current simplex with just found support point in direction -dir.
        """
        self.current_s.setVertex(self.v, self.current_s.rank)
        self.current_s.rank += 1

    def project_and_update_simplex(self):
        """
        Project origin onto simplex --> will be stored in GJK::ray
        Also checks if origin is inside simplex.
        Finally, updates next_s to the new simplex.
        """
        self.gjk_solver.nfree = 4 - self.current_s.rank
        self.gjk_solver.ray = self.ray
        if self.current_s.rank == 1:
            self.ray = self.v.w
            self.next_s.setVertex(self.v, 0)
            self.next_s.rank = 1
        else:
            if self.current_s.rank == 2:
                self.flag_inside = self.gjk_solver.projectLineOrigin(self.current_s, self.next_s)
            elif self.current_s.rank == 3:
                self.flag_inside = self.gjk_solver.projectTriangleOrigin(self.current_s, self.next_s)
            elif self.current_s.rank == 4:
                self.flag_inside = self.gjk_solver.projectTetrahedraOrigin(self.current_s, self.next_s)
            self.ray = self.gjk_solver.ray

        self.num_call_projection += 1

        # Update ray_norm
        self.ray_norm = np.linalg.norm(self.ray)

        # Update simplices
        self.current_s = copySimplex(self.next_s)
        self.next_s = newSimplex()

    def cv_check_C(self):
        """
        Checks the inside flag to see if origin is inside current simplex.
        """
        if self.flag_inside:
            # Shapes are interpenetrating ==> ray = [0, 0, 0]
            self.flag_end = True

    def compute_witness_points(self):
        """
        Updates x, x0 and x1 according to the contents of the current simplex. 
        """
        if self.current_s.rank == 1:
            self.ray_weigths[0] = 1.
        else:
            if self.current_s.rank == 2:
                a = self.current_s.getVertex(0).w
                b = self.current_s.getVertex(1).w
                projection_result = hppfcl.projectLineOrigin(a, b)
            if self.current_s.rank == 3:
                a = self.current_s.getVertex(0).w
                b = self.current_s.getVertex(1).w
                c = self.current_s.getVertex(2).w
                projection_result = hppfcl.projectTriangleOrigin(a, b, c)
            if self.current_s.rank == 4:
                a = self.current_s.getVertex(0).w
                b = self.current_s.getVertex(1).w
                c = self.current_s.getVertex(2).w
                d = self.current_s.getVertex(3).w
                projection_result = hppfcl.projectTetrahedraOrigin(a, b, c, d)
            projection_result.updateParameterization()
            for i in range(self.current_s.rank):
                self.ray_weigths[i] = projection_result.parameterization_eigen[i]

        self.x = np.zeros(3)
        self.x0 = np.zeros(3)
        self.x1 = np.zeros(3)
        for i in range(self.current_s.rank):
            w = self.ray_weigths[i]
            v = self.current_s.getVertex(i).w
            v0 = self.current_s.getVertex(i).w0
            v1 = self.current_s.getVertex(i).w1
            self.x += w * v
            self.x0 += w * v0
            self.x1 += w * v1

    def make_step(self):
        """
        One step of the solver. Run this until numit > max_iterations.
        """
        # Tolerance check
        self.cv_check_A()

        # Get direction for support point
        if not self.flag_end:
            self.compute_dir()

            # Store support point in vertex. Will be added to simplex if check B doesn't pass
            self.get_support_and_create_vertex()

            # Has a separating plane been found?
            self.check_early_stopping()

            # FW gap convergence check
            self.cv_check_B()

        if not self.flag_end:
            if self.flag_switch_to_gjk:
                self.current_gjk_variant = hppfcl.GJKVariant.DefaultGJK
                self.flag_switch_to_gjk = False
            else:
                # Update simplex with previously found vertex
                self.update_current_simplex()

                # Project origin onto current simplex and update current simplex
                self.project_and_update_simplex()

                # Check if origin is inside current simplex
                self.cv_check_C()

                if self.compute_closest_points:
                    self.compute_witness_points()

                # Update number of iteration when iterate has been updated
                self.numit += 1

    def evaluate(self, T0: hppfcl.Transform3f, T1: hppfcl.Transform3f):
        self.reset(T0, T1)

        for _ in range(self.max_iterations):
            self.make_step()
            if self.flag_end:
                if self.verbose:
                    print(f"Id {self.id} - END. Final distance btw shapes: {np.linalg.norm(self.ray)}")
                    print(f"Rank final simplex: {self.current_s.rank}")
                if self.flag_inside:
                    return True
                else:
                    return False
        if self.verbose:
            print("Took too many iterations to converge. Exiting.")
        return False

    def compute_execution_time(self, T0: hppfcl.Transform3f, T1: hppfcl.Transform3f):
        """
        This function does **nothing** more than evaluate.
        To measure execution time of solvers, use the GJKSolverHPPFCL class.
        """
        self.evaluate(T0, T1)
        self.mean_execution_time = 0.
        self.mean_execution_time_early = 0.

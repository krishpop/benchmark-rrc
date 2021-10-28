import cvxpy as cp
import numpy as np
import torch
import multiprocessing
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation
from cvxpylayers.torch import CvxpyLayer
from diffcp import SolverError


class ForceOptProblem:
    def __init__(
        self,
        obj_mu=1.0,
        mass=0.016,
        gravity=-9.81,
        target_n=0.1,
        cone_approx=False,
        object_frame=False,
    ):
        self.obj_mu = obj_mu
        self.mass = mass
        self.gravity = gravity
        self.target_n = target_n  # 1.0
        self.cone_approx = cone_approx
        self.object_frame = object_frame
        self.initialized = False
        self.prob = None

    def setup_cvxpy_layer(self):
        # Try solving optimization problem
        # contact force decision variable
        self.target_n_t = torch.as_tensor(
            np.array([self.target_n, 0, 0] * 3), dtype=torch.float32
        )
        self.target_n_cp = cp.Parameter(
            (9,), name="target_n", value=self.target_n_t.data.numpy()
        )
        self.L = cp.Variable(9, name="l")
        self.W = cp.Parameter((6,), name="w_des")
        self.G = cp.Parameter((6, 9), name="grasp_m")
        cm = np.vstack((np.eye(3), np.zeros((3, 3)))) * self.mass

        inputs = [self.G, self.W, self.target_n_cp]
        outputs = [self.L]
        # self.Cm = cp.Parameter((6, 3), value=cm*self.mass, name='com')

        f_g = np.array([0, 0, self.gravity])
        if self.object_frame:
            self.R_w_2_o = cp.Parameter((6, 6), name="r_w_2_o")
            w_ext = self.W + self.R_w_2_o @ cm @ f_g
            inputs.append(self.R_w_2_o)
        else:
            w_ext = self.W + cm @ f_g

        f = self.G @ self.L - w_ext  # generated contact forces must balance wrench

        # Objective function - minimize force magnitudes
        contact_force = self.L - self.target_n_cp
        cost = cp.sum_squares(contact_force)

        # Friction cone constraints; >= 0
        self.constraints = []
        self.cone_constraints = []
        if self.cone_approx:
            self.cone_constraints += [cp.abs(self.L[1::3]) <= self.obj_mu * self.L[::3]]
            self.cone_constraints += [cp.abs(self.L[2::3]) <= self.obj_mu * self.L[::3]]
        else:
            self.cone_constraints.append(
                cp.SOC(self.obj_mu * self.L[::3], (self.L[2::3] + self.L[1::3])[None])
            )
        self.constraints.append(f == np.zeros(f.shape))

        self.prob = cp.Problem(
            cp.Minimize(cost), self.cone_constraints + self.constraints
        )
        self.policy = CvxpyLayer(self.prob, inputs, outputs)
        self.initialized = True

    def get_grasp_matrix(self, cp_list, obj_pose):
        GT_list = []
        fnum = len(cp_list)
        H = self._get_H_matrix(fnum)
        for cp_of in cp_list:
            if cp_of is not None:
                GT_i = self._get_grasp_matrix_single_cp(cp_of, obj_pose)
                GT_list.append(GT_i)
            else:
                GT_list.append(np.zeros((6, 6)))
        GT_full = np.concatenate(GT_list)
        GT = H @ GT_full
        return GT.T

    def _get_grasp_matrix_single_cp(self, cp_of, obj_pose):
        P = self._get_P_matrix(cp_of[0], obj_pose)
        quat_o_2_w = obj_pose[3:]
        quat_c_2_o = cp_of[1]

        # Orientation of cp frame w.r.t. world frame
        # quat_c_2_w = quat_o_2_w * quat_c_2_o
        # R is rotation matrix from contact frame i to world frame
        if self.object_frame:
            rot = Rotation.from_quat(quat_c_2_o)
        else:
            rot = Rotation.from_quat(quat_o_2_w) * Rotation.from_quat(quat_c_2_o)
        R = rot.as_matrix()
        R_bar = block_diag(R, R)

        G = P @ R_bar
        return G.T

    def _get_P_matrix(self, pos_of, obj_pose):
        quat_o_2_w = obj_pose[3:]
        if self.object_frame:
            r = pos_of
        else:
            r = Rotation.from_quat(quat_o_2_w).as_matrix() @ pos_of
        S = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])

        P = np.eye(6)
        P[3:6, 0:3] = S
        return P

    def _get_H_matrix(self, fnum):
        H_i = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        )
        H = block_diag(H_i, H_i, H_i)
        return H

    def get_w2o_rot(self, obj_pose):
        quat_o_2_w = obj_pose[3:]
        R_w_2_o = Rotation.from_quat(quat_o_2_w).as_matrix().T
        R_w_2_o = block_diag(R_w_2_o, R_w_2_o)
        return R_w_2_o

    def balance_force_test(self, des_wrench, balance_force, cp_list, obj_pose):
        if self.object_frame:
            R_w_2_o = self.get_w2o_rot(obj_pose)
            weight = (
                R_w_2_o
                @ np.vstack([np.eye(3) * self.mass, np.zeros((3, 3))])
                @ np.array([0, 0, self._gravity])
            )
        else:
            weight = np.vstack([np.eye(3), np.zeros((3, 3))]) @ np.array(
                [0, 0, self.gravity * self.mass]
            )
        G = self.get_grasp_matrix(cp_list, obj_pose)
        w_ext = des_wrench + weight
        f = G @ balance_force - w_ext
        return f

    def __call__(self, des_wrench, obj_pose, cp_list):
        if not self.initialized:
            self.setup_cvxpy_layer()
        return self.run_fop(des_wrench, obj_pose, cp_list)

    def run_fop(self, des_wrench, obj_pose, cp_list):
        G = self.get_grasp_matrix(cp_list, obj_pose)
        G_t = torch.from_numpy(G.astype("float32", copy=False))
        des_wrench_t = torch.from_numpy(des_wrench.astype("float32", copy=False))
        inputs = [G_t, des_wrench_t, self.target_n_t]
        if self.object_frame:
            R_w_2_o = self.get_w2o_rot(obj_pose)
            R_w_2_o_t = torch.from_numpy(R_w_2_o.astype("float32", copy=False))
            inputs.append(R_w_2_o_t)
        try:
            (balance_force,) = self.policy(*inputs)
            return balance_force
        except SolverError:
            return torch.zeros(9)


class BatchForceOptProblem:
    def __init__(
        self,
        obj_mu=1.0,
        mass=0.016,
        gravity=-9.81,
        target_n=0.0,
        cone_approx=True,
        device="cuda:0",
    ):
        self.obj_mu = obj_mu
        self.mass = mass
        self.gravity = gravity
        self.target_n = target_n  # 1.0
        self.cone_approx = cone_approx
        self.initialized = False
        self.prob = None
        assert device in ["cuda:0", "cpu"], f'{device} not in ["cuda:0", "cpu"]'
        if device == "cuda:0" and not torch.cuda.is_available():
            print("switching device to CPU because cuda not available")
            device = "cpu"
        self.device = device

    def setup_cvxpy_layer(self):
        # Try solving optimization problem
        # contact force decision variable
        target_n = np.array([self.target_n, 0, 0] * 3).astype("float32", copy=False)
        self.target_n_t = torch.as_tensor(
            target_n, dtype=torch.float32, device=self.device
        )
        self.target_n_cp = cp.Parameter((9,), name="target_n", value=target_n)
        self.L = cp.Variable(9, name="l")
        self.W = cp.Parameter((6,), name="w_des")
        self.G = cp.Parameter((6, 9), name="grasp_m")
        cm = np.vstack((np.eye(3), np.zeros((3, 3)))) * self.mass

        inputs = [self.G, self.W, self.target_n_cp]
        outputs = [self.L]
        # self.Cm = cp.Parameter((6, 3), value=cm*self.mass, name='com')

        f_g = np.array([0, 0, self.gravity])
        w_ext = self.W + cm @ f_g

        f = self.G @ self.L - w_ext  # generated contact forces must balance wrench

        # Objective function - minimize force magnitudes
        contact_force = self.L - self.target_n_cp
        cost = cp.sum_squares(contact_force)

        # Friction cone constraints; >= 0
        self.constraints = []
        self.cone_constraints = []
        if self.cone_approx:
            self.cone_constraints += [cp.abs(self.L[1::3]) <= self.obj_mu * self.L[::3]]
            self.cone_constraints += [cp.abs(self.L[2::3]) <= self.obj_mu * self.L[::3]]
        else:
            self.cone_constraints.append(
                cp.SOC(self.obj_mu * self.L[::3], (self.L[2::3] + self.L[1::3])[None])
            )
        self.constraints.append(f == np.zeros(f.shape))

        self.prob = cp.Problem(
            cp.Minimize(cost), self.cone_constraints + self.constraints
        )
        self.policy = CvxpyLayer(self.prob, inputs, outputs)
        self.initialized = True

    def get_grasp_matrix(self, cp_list):
        GT_list = []
        fnum = len(cp_list)
        H = self._get_H_matrix(fnum)
        if not isinstance(cp_list, list) and len(cp_list.shape) != 2:
            cp_list = cp_list.reshape((3, 7))
        for cp_wf in cp_list:
            if cp_wf is not None and not (cp_wf == 0).all():
                GT_i = self._get_grasp_matrix_single_cp(cp_wf)
                GT_list.append(GT_i)
            else:
                GT_list.append(np.zeros((6, 6)))
        GT_full = np.concatenate(GT_list)
        GT = H @ GT_full
        return GT.T

    def _get_grasp_matrix_single_cp(self, cp_wf):
        P = self._get_P_matrix(cp_wf[:3])
        quat_c_2_w = cp_wf[3:]

        # Orientation of cp frame w.r.t. world frame
        # quat_c_2_w = quat_o_2_w * quat_c_2_o
        # R is rotation matrix from contact frame i to world frame
        R = Rotation.from_quat(quat_c_2_w).as_matrix()
        R_bar = block_diag(R, R)

        G = P @ R_bar
        return G.T

    def _get_P_matrix(self, pos_wf):
        r = pos_wf
        S = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])

        P = np.eye(6)
        P[3:6, 0:3] = S
        return P

    def _get_H_matrix(self, fnum):
        H_i = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        )
        H = block_diag(H_i, H_i, H_i)
        return H

    def balance_force_test(self, des_wrench, balance_force, cp_list):
        weight = np.vstack([np.eye(3), np.zeros((3, 3))]) @ np.array(
            [0, 0, self.gravity * self.mass]
        )
        G = self.get_grasp_matrix(cp_list)
        w_ext = des_wrench + weight
        f = G @ balance_force - w_ext
        return f

    def __call__(self, des_wrench_t, cp_list):
        if not self.initialized:
            self.setup_cvxpy_layer()
        return self.run_fop(des_wrench_t, cp_list)

    def run_fop(self, des_wrench_t, cp_list):
        if len(des_wrench_t.shape) == 2:
            G = np.apply_along_axis(self.get_grasp_matrix, 1, cp_list)
            target_n_t = self.target_n_t.tile((len(cp_list), 1))
        else:
            G = self.get_grasp_matrix(cp_list)
            target_n_t = self.target_n_t

        G_t = torch.from_numpy(G.astype("float32", copy=False)).to(self.device)
        inputs = [G_t, des_wrench_t, target_n_t]
        try:
            (balance_force,) = self.policy(*inputs)
            return balance_force
        except SolverError:
            return torch.zeros((des_wrench_t.shape[0], 9)).to(self.device)

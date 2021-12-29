import cvxpy as cp
import numpy as np
import multiprocessing
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation
from diffcp import SolverError
import functools

try:
    from isaacgym import gymtorch
    from isaacgym import gymapi
    from isaacgym.torch_utils import *

    imported_ig = True
except ImportError:
    print("IsaacGym not installed, FoP torch methods disabled")
    gymtorch = gymapi = None
    imported_ig = False

import torch
from cvxpylayers.torch import CvxpyLayer


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

    def get_grasp_matrix(self, cp_list_of, obj_pose):
        GT_list = []
        fnum = len(cp_list_of)
        H = self._get_H_matrix(fnum)
        for cp_pos_ori in cp_list_of:
            if cp_pos_ori is not None:
                GT_i = self._get_grasp_matrix_single_cp(cp_pos_ori, obj_pose)
                GT_list.append(GT_i)
            else:
                GT_list.append(np.zeros((6, 6)))
        GT_full = np.concatenate(GT_list)
        GT = H @ GT_full
        return GT.T

    def _get_grasp_matrix_single_cp(self, cp_pos_ori, obj_pose):
        P = self._get_P_matrix(cp_pos_ori[:3], obj_pose)
        quat_o_2_w = obj_pose[3:]
        quat_c_2_o = cp_pos_ori[3:]

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

    def _get_P_matrix(self, cp_pos, obj_pose):
        quat_o_2_w = obj_pose[3:]
        if self.object_frame:
            r = cp_pos
        else:
            r = Rotation.from_quat(quat_o_2_w).as_matrix() @ cp_pos
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

    def balance_force_test(self, des_wrench, balance_force, cp_list_of, obj_pose):
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
        G = self.get_grasp_matrix(cp_list_of, obj_pose)
        w_ext = des_wrench + weight
        f = G @ balance_force - w_ext
        return f

    def __call__(self, des_wrench, obj_pose, cp_list):
        if not self.initialized:
            self.setup_cvxpy_layer()
        return self.run_fop(des_wrench, obj_pose, cp_list)

    def run_fop(self, des_wrench, obj_pose, cp_list_of):
        G = self.get_grasp_matrix(cp_list_of, obj_pose)
        G_t = torch.as_tensor(G, dtype=torch.float32)
        des_wrench_t = torch.as_tensor(des_wrench, dtype=torch.float32)
        inputs = [G_t, des_wrench_t, self.target_n_t]
        if self.object_frame:
            R_w_2_o = self.get_w2o_rot(obj_pose)
            R_w_2_o_t = torch.as_tensor(R_w_2_o, dtype=torch.float32)
            inputs.append(R_w_2_o_t)
        try:
            (balance_force,) = self.policy(*inputs)
            return balance_force
        except SolverError:
            return torch.zeros(9, dtype=torch.float32)


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
        self._H = self._get_H_matrix()

    def setup_cvxpy_layer(self):
        # Try solving optimization problem
        # contact force decision variable
        target_n = np.array([self.target_n, 0, 0] * 3, dtype="float32")
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

    def _get_P_matrix(self, pos_wf):
        if torch.is_tensor(pos_wf):
            S = self._skew_mat_t(pos_wf)
            return torch.cat(
                (torch.eye(6, dtype=torch.float, device=pos_wf.device)[:3], S)
            )
        else:
            r = pos_wf
            S = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            P = np.eye(6)
            P[3:6, 0:3] = S
            return P

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

        G_t = torch.as_tensor(G, dtype=torch.float32, device=self.device)
        inputs = [G_t, des_wrench_t, target_n_t]
        try:
            (balance_force,) = self.policy(*inputs)
            return balance_force
        except SolverError:
            return torch.zeros((des_wrench_t.shape[0], 9), device=self.device)

    def get_grasp_matrix(self, cp_list):
        GT_list = []
        H = self._H  # 9 x 18
        if torch.is_tensor(cp_list):
            H = to_torch(H)
        elif all([x is not None for x in cp_list]):
            # make cp_list an np.ndarray
            cp_list = np.asarray(cp_list).reshape(3, 7)
        for cp_wf in cp_list:
            if cp_wf is not None and not (cp_wf == 0).all():
                GT_i = self._get_grasp_matrix_single_cp(cp_wf)
                GT_list.append(GT_i)
            else:
                GT_list.append(np.zeros((6, 6)))
        if torch.is_tensor(cp_list):
            GT_full = torch.cat(GT_list)  # 18 x 6
        else:
            GT_full = np.concatenate(GT_list)
        GT = H @ GT_full  # 9 x 6
        return GT.T  # 6 x 9

    def _get_grasp_matrix_single_cp(self, cp_wf):
        P = self._get_P_matrix(cp_wf[:3])  # 6 x 6
        quat_c_2_w = cp_wf[3:]

        # Orientation of cp frame w.r.t. world frame
        # quat_c_2_w = quat_o_2_w * quat_c_2_o
        # R is rotation matrix from contact frame i to world frame
        if torch.is_tensor(cp_wf):
            quat_c_2_w = quat_c_2_w.unsqueeze(0)
            R = euler_angles_to_matrix(to_torch(list(get_euler_xyz(quat_c_2_w))))
            R_bar = torch.block_diag(R, R)
        else:
            R = Rotation.from_quat(quat_c_2_w).as_matrix()
            R_bar = block_diag(R, R)  # 6 x 6
        G = P @ R_bar  # 6 x 6
        return G.T

    def _skew_mat_t(self, x_vec):
        W_row0 = to_torch([0, 0, 0, 0, 0, 1, 0, -1, 0], device=x_vec.device).view(3, 3)
        W_row1 = to_torch([0, 0, -1, 0, 0, 0, 1, 0, 0], device=x_vec.device).view(3, 3)
        W_row2 = to_torch([0, 1, 0, -1, 0, 0, 0, 0, 0], device=x_vec.device).view(3, 3)
        x_skewmat = torch.stack(
            [
                torch.matmul(x_vec, W_row0.t()),
                torch.matmul(x_vec, W_row1.t()),
                torch.matmul(x_vec, W_row2.t()),
            ],
            dim=-1,
        )
        x_skewmat = torch.cat((x_skewmat, to_torch(torch.eye(3))), dim=1)
        return x_skewmat

    def _get_H_matrix(self):
        H_i = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        )
        H = block_diag(H_i, H_i, H_i)
        return H


# from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L185


def euler_angles_to_matrix(euler_angles):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    convention = "XYZ"
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

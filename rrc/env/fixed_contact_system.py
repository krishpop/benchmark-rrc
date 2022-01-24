import numpy as np
from casadi import vertcat, SX, nlpsol, norm2
from rrc.env import contact_opt_utils as utils


class ContactPoint:
    """
    Class ContactPoint:
    --------------
    A class used to represent a contact point on planar object, in WORLD frame (wf)
    """

    def __init__(self, pos_of, quat_of):
        # In Object frame
        self.pos_of = pos_of
        self.quat_of = quat_of

        self.pos_wf = None
        self.quat_wf = None

    def print_pt_wf(self):
        print("Contact point position WF: ", self.pos_wf, ", Y: ", self.quat_wf)

    def print_pt_of(self):
        print("Contact point position OF: ", self.pos_of, ", Y: ", self.quat_of)


class FixedContactPointOpt:
    def __init__(
        self,
        nGrid=100,
        dt=0.1,
        fnum=3,
        cp_params=None,
        x0=np.array([[0, 0, 0.0325, 0, 0, 0, 1]]),
        x_goal=None,
        obj_shape=None,
        obj_mass=None,
        npz_filepath=None,
    ):

        self.nGrid = nGrid
        self.dt = dt

        # Define system
        self.system = FixedContactPointSystem(
            nGrid=nGrid,
            dt=dt,
            fnum=fnum,
            cp_params=cp_params,
            obj_shape=obj_shape,
            obj_mass=obj_mass,
        )

        # Test various functions
        # x = np.zeros((1,7))
        # x[0,0:3] = obj_pose.position
        # x[0,3] = obj_pose.orientation[3]
        # x[0,4:7] = obj_pose.orientation[0:3]
        # print("x: {}".format(x))
        # self.system.get_grasp_matrix(x)

        # Get decision variables
        self.t, self.s_flat, self.l_flat, self.a = self.system.dec_vars()
        # Pack t,x,u,l into a vector of decision variables
        self.z = self.system.decvar_pack(self.t, self.s_flat, self.l_flat, self.a)

        # print(self.z)
        # print(self.system.s_unpack(self.s_flat))

        # Formulate constraints
        self.g, self.lbg, self.ubg = self.get_constraints(
            self.system, self.t, self.s_flat, self.l_flat, self.a, x_goal
        )

        # Get cost function
        self.cost = self.cost_func(self.t, self.s_flat, self.l_flat, self.a, x_goal)

        # Formulate nlp
        problem = {"x": self.z, "f": self.cost, "g": self.g}
        options = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 10000,
            "ipopt.tol": 1e-4,
            "print_time": 0,
        }
        # options["print_time"] = 0;
        # options = {"iteration_callback": MyCallback('callback',self.z.shape[0],self.g.shape[0],self.system)}
        # options["monitor"] = ["nlp_g"]
        # options = {"monitor":["nlp_f","nlp_g"]}
        self.solver = nlpsol("S", "ipopt", problem, options)

        # TODO: intial guess
        self.z0 = self.system.get_initial_guess(self.z, x0, x_goal)
        # t0, s0, l0 = self.system.decvar_unpack(self.z0)
        # x0, dx0 = self.system.s_unpack(s0)
        # self.get_constraints(self.system,t0,s0,l0)

        # print("\nINITIAL TRAJECTORY")
        # print("time: {}".format(t0))
        # print("x: {}".format(x0))
        # print("dx: {}".format(dx0))
        # print("contact forces: {}".format(l0))

        # TODO: path constraints
        # self.system.get_grasp_matrix(x0)

        self.z_lb, self.z_ub = self.system.path_constraints(
            self.z, x0, x_goal=x_goal, dx0=np.zeros((1, 6)), dx_end=np.zeros((1, 6))
        )

        # Set upper and lower bounds for decision variables
        r = self.solver(
            x0=self.z0, lbg=self.lbg, ubg=self.ubg, lbx=self.z_lb, ubx=self.z_ub
        )
        z_soln = r["x"]

        # Final solution and cost
        self.cost = r["f"]
        self.t_soln, self.s_soln, l_soln_flat, a_soln = self.system.decvar_unpack(
            z_soln
        )
        self.x_soln, self.dx_soln = self.system.s_unpack(self.s_soln)
        self.l_soln = self.system.l_unpack(l_soln_flat)

        # Check that all quaternions are unit quaternions
        # print("Check quaternion magnitudes")
        # for i in range(self.nGrid):
        #  quat = self.x_soln[i, 3:]
        #  print(np.linalg.norm(quat))

        # Transform contact forces from contact point frame to world frame
        self.l_wf_soln = np.zeros(self.l_soln.shape)
        for t_i in range(self.l_soln.shape[0]):
            for f_i in range(self.system.fnum):
                # print(self.l_soln[t_i, :])
                # print("FINGER {}".format(f_i))
                ftip_soln = self.l_soln[
                    t_i, f_i * self.system.l_i : (f_i + 1) * self.system.l_i
                ].T
                l_of = self.system.get_R_cp_2_o(self.system.cp_list[f_i]) @ ftip_soln
                # print(l_of)
                l_wf = self.system.get_R_o_2_w(self.x_soln[t_i, :]) @ l_of
                # print(l_wf)

                for d in range(self.system.l_i):
                    self.l_wf_soln[t_i, f_i * self.system.l_i + d] = (
                        l_wf[:, 0].elements()[d].__float__()
                    )
        # Final distance to goal
        # eef_final = self.system.get_eef_pos_world(self.q_soln)[-1, 0:2]
        # self.final_dist = norm_2(eef_final - eef_goal)

        # Save solver time
        statistics = self.solver.stats()
        self.total_time_sec = statistics["t_wall_total"]

        # Save solution
        if npz_filepath is not None:
            np.savez(
                npz_filepath,
                dt=self.system.dt,
                nGrid=self.system.nGrid,
                x0=x0,
                x_goal=x_goal,
                t=self.t_soln,
                x=self.x_soln,
                dx=self.dx_soln,
                l_of=self.l_soln,
                l_wf=self.l_wf_soln,
                cp_params=cp_params,
            )

    def cost_func(self, t, s_flat, l_flat, a, x_goal):
        """Computes cost"""
        cost = 0
        R = np.eye(self.system.fnum * self.system.l_i) * 1
        Q = np.diag([10, 10, 10, 100, 100, 100, 100])

        lmbda = self.system.l_unpack(l_flat)
        x, dx = self.system.s_unpack(s_flat)

        n = 0.1
        target_normal_forces = np.zeros(lmbda[0, :].shape)
        for f_i in range(self.system.fnum):
            target_normal_forces[0, self.system.qnum * f_i] = n

        # Slack variable penalties
        for i in range(a.shape[0]):
            if i < 3:
                cost += a[i] * 1  # Position of object
            else:
                cost += a[i] * 1  # Orientation of object

        # Contact forces
        for i in range(t.shape[0]):
            cost += (
                0.5
                * (lmbda[i, :] - target_normal_forces)
                @ R
                @ (lmbda[i, :] - target_normal_forces).T
            )

        for i in range(t.shape[0]):
            # Add the current distance to goal
            x_curr = x[i, :]
            delta = x_goal - x_curr
            cost += 0.5 * delta @ Q @ delta.T

        return cost

    def get_constraints(self, system, t, s, lmbda, a, x_goal):
        """Formulates collocation constraints"""
        ds = system.dynamics(s, lmbda)

        x, dx = system.s_unpack(s)
        new_dx, ddx = system.s_unpack(ds)

        # Separate x and new_dx into position and orientation (quaternion) so we can normalize quaterions
        pos = x[:, 0:3]
        new_dpos = new_dx[:, 0:3]
        quat = x[:, 3:]
        new_dquat = new_dx[:, 3:]

        g = []  # Dynamics constraints
        lbg = []  # Dynamics constraints lower bound
        ubg = []  # Dynamics constraints upper bound

        # Loop over entire trajectory
        for i in range(t.shape[0] - 1):
            dt = t[i + 1] - t[i]

            # pose - velocity
            # Handle position and linear velocity constraints first, since they don't need to be normalized
            for j in range(3):
                # dx
                f = (
                    0.5 * dt * (new_dpos[i + 1, j] + new_dpos[i, j])
                    + pos[i, j]
                    - pos[i + 1, j]
                )
                # print("new_dx, x, t{}, dim{}: {}".format(i,j,f))
                g.append(f)
                lbg.append(0)
                ubg.append(0)

            # Handle orientation and angular velocity - normalize first
            quat_i_plus_one = (
                0.5 * dt * (new_dquat[i + 1, :] + new_dquat[i, :]) + quat[i, :]
            )
            quat_i_plus_one_unit = quat_i_plus_one / norm_2(quat_i_plus_one)
            for j in range(4):
                # dx
                f = quat_i_plus_one_unit[0, j] - quat[i + 1, j]
                # print("new_dquat, quat, t{}, dim{}: {}".format(i,j,f))
                g.append(f)
                lbg.append(0)
                ubg.append(0)

            # velocity - acceleration
            # iterate over all dofs
            for j in range(system.dx_dim):
                f = 0.5 * dt * (ddx[i + 1, j] + ddx[i, j]) + dx[i, j] - dx[i + 1, j]
                # print("dx, ddx, t{}, dim{}: {}".format(i,j,f))
                g.append(f)
                lbg.append(0)
                ubg.append(0)

        # Linearized friction cone constraints
        # f_constraints = system.friction_cone_constraints(l)
        # for r in range(f_constraints.shape[0]):
        #  for c in range(f_constraints.shape[1]):
        #    g.append(f_constraints[r,c])
        #    lbg.append(0)
        #    ubg.append(np.inf)

        # tol = 1e-16
        x_goal_constraints = system.x_goal_constraint(s, a, x_goal)
        for r in range(x_goal_constraints.shape[0]):
            for c in range(x_goal_constraints.shape[1]):
                g.append(x_goal_constraints[r, c])
                lbg.append(0)
                ubg.append(np.inf)
        return vertcat(*g), vertcat(*lbg), vertcat(*ubg)


class FixedContactPointSystem:
    def __init__(
        self,
        nGrid=100,
        dt=0.1,
        fnum=3,
        cp_params=None,
        obj_shape=None,
        obj_mass=None,
        log_file=None,
    ):

        # Time parameters
        self.nGrid = nGrid
        self.dt = dt
        self.tf = dt * (nGrid - 1)  # Final time

        self.fnum = fnum
        self.qnum = 3
        self.obj_dof = 6
        self.x_dim = 7  # Dimension of object pose
        self.dx_dim = 6  # Dimension of object twist

        self.p = 100

        self.obj_shape = obj_shape  # (width, length, height), (x, y, z)
        self.obj_mass = obj_mass
        self.obj_mu = 1

        self.gravity = -10

        self.cp_params = cp_params
        self.cp_list = self.get_contact_points_from_cp_params(self.cp_params)

        # Contact model force selection matrix
        l_i = 3
        self.l_i = l_i
        H_i = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                # [0, 0, 0, 1, 0, 0],
            ]
        )
        self.H = np.zeros((l_i * self.fnum, self.obj_dof * self.fnum))
        for i in range(self.fnum):
            self.H[
                i * l_i : i * l_i + l_i,
                i * self.obj_dof : i * self.obj_dof + self.obj_dof,
            ] = H_i

        self.log_file = log_file

    ################################################################################
    # Decision variable management helper functions
    ################################################################################

    def dec_vars(self):
        """
        Define decision variables
            Returns:
                t (int): time
                s_flat: state [x, dx] (flattened vector)
                l_flat: contact forces
                a: Slack variables
        """
        x_dim = self.x_dim
        dx_dim = self.dx_dim
        fnum = self.fnum
        nGrid = self.nGrid

        # time
        t = SX.sym("t", nGrid)

        # object pose at every timestep
        # one row of x is [x, y, z, qw, qx, qy, qz]
        x = SX.sym("x", nGrid, x_dim)

        # object velocity at every timestep
        # one row of dx is [dx, dy, dtheta]
        dx = SX.sym("dx", nGrid, dx_dim)

        # Lamda (applied contact forces) at every timestep
        # one row of l is [normal_force_f1, tangent_force_f1, ..., normal_force_fn, tangent_force_fn]
        lmbd = SX.sym("l", nGrid, fnum * self.l_i)

        # Slack variables for x_goal
        a = SX.sym("a", x_dim)

        # Flatten vectors
        s_flat = self.s_pack(x, dx)
        l_flat = self.l_pack(lmbd)

        return t, s_flat, l_flat, a

    def decvar_pack(self, t, s, l, a):
        """Pack the decision variables into a single horizontal vector"""
        z = vertcat(t, s, l, a)
        return z

    def decvar_unpack(self, z):
        """
        Unpack the decision variable vector z into:
            Returns:
                t: times (nGrid x 1) vector
                s: packed state vector
                u: packed u vector (joint torques)
                l: packed l vector (contact forces)
                a: slack variables
        """
        qnum = self.qnum
        fnum = self.fnum
        nGrid = self.nGrid
        x_dim = self.x_dim
        dx_dim = self.dx_dim

        t = z[:nGrid]

        s_start_ind = nGrid
        s_end_ind = s_start_ind + nGrid * x_dim + nGrid * dx_dim
        s_flat = z[s_start_ind:s_end_ind]

        l_start_ind = s_end_ind
        l_end_ind = l_start_ind + nGrid * fnum * self.l_i
        l_flat = z[l_start_ind:l_end_ind]

        a_start_ind = l_end_ind
        a = z[a_start_ind:]

        return t, s_flat, l_flat, a

    def s_unpack(self, s):
        """
        Unpack the state vector s
            Returns:
                x: (x,y,theta) pose of object (nGrid x dim)
                dx: (dx,dy,dtheta) of object (nGrid x dim)
        """
        nGrid = self.nGrid
        x_dim = self.x_dim
        dx_dim = self.dx_dim

        # Get object pose
        x_flat = s[: nGrid * x_dim]
        x = reshape(x_flat, x_dim, nGrid).T

        # Get object twist
        dx_flat = s[nGrid * x_dim :]
        dx = reshape(dx_flat, dx_dim, nGrid).T

        return x, dx

    def s_pack(self, x, dx):
        """
        Pack the state vector s into a single horizontal vector
        State:
            Returns:
                x: (px, py, pz, qx, qy, qz, qw) pose of object
                dx: d(px, py, pz, qx, qy, qz, qw) velocity of object
        """
        nGrid = self.nGrid
        x_dim = self.x_dim
        dx_dim = self.dx_dim

        x_flat = reshape(x.T, nGrid * x_dim, 1)
        dx_flat = reshape(dx.T, nGrid * dx_dim, 1)

        return vertcat(x_flat, dx_flat)

    def l_pack(self, l):
        """Pack the l vector into single horizontal vector"""
        nGrid = self.nGrid
        fnum = self.fnum
        l_flat = reshape(l.T, nGrid * fnum * self.l_i, 1)
        return l_flat

    def l_unpack(self, l_flat):
        """Unpack flat l fector in a (nGrid x fnum*dim) array"""
        nGrid = self.nGrid
        fnum = self.fnum
        lmbda = reshape(l_flat, self.l_i * fnum, nGrid).T
        return lmbda

    ################################################################################
    # Constraint functions
    ################################################################################

    def dynamics(self, s_flat, l_flat):
        """
        Compute system dynamics (ds/dt):
            Parameters:
                s_flat: state vector
                l_flat: contact forces

            Returns:
                Derivative of state, ds, as a flattened vector with same dimension as s_flat
        """
        # Unpack variables
        x, dx = self.s_unpack(s_flat)
        l = self.l_unpack(l_flat)

        new_dx_list = []
        ddx_list = []
        for t_ind in range(self.nGrid):
            x_i = x[t_ind, :]
            dx_i = dx[t_ind, :]

            # Compute dx at each collocation point
            # dx is a (7x1) vector
            new_dx_i = SX.zeros((7, 1))
            # First 3 elements are position time-derivatives
            new_dx_i[0:3, :] = dx_i[0, 0:3]
            # Last 4 elements are quaternion time-derivatives
            ## Transform angular velocities dx into quaternion time-derivatives
            quat_i = x_i[0, 3:]
            dquat_i = 0.5 * self.get_dx_to_dquat_matrix(quat_i) @ dx_i[0, 3:].T
            new_dx_i[3:, :] = dquat_i
            new_dx_list.append(new_dx_i)

            # Compute ddx at each collocation point
            Mo = self.get_M_obj()
            G = self.get_grasp_matrix(x_i)
            gapp = self.get_gapp()
            l_i = l[t_ind, :].T
            # print("t_ind: {}".format(t_ind))
            # print(x_i)
            # print(l_i)
            # print(inv(Mo))
            # print("gapp: {}".format(gapp))
            # print(G.shape)
            # print((gapp + G@l_i).shape)
            ddx_i = inv(Mo) @ (gapp + G @ l_i)
            ddx_list.append(ddx_i)

        new_dx = horzcat(*new_dx_list).T
        ddx = horzcat(*ddx_list).T

        ds = self.s_pack(new_dx, ddx)
        return ds

    def get_dx_to_dquat_matrix(self, quat):
        """
        Get matrix to transform angular velocity to quaternion time derivative
            Parameters:
                quat: [qx, qy, qz, qw]
        """
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]
        qw = quat[3]

        M = np.array(
            [
                [-qx, -qy, -qz],
                [qw, qz, -qy],
                [-qz, qw, qx],
                [qy, -qx, qw],
            ]
        )
        return SX(M)

    def friction_cone_constraints(self, l_flat):
        """
        Linearized friction cone constraint
        Approximate cone as an inner pyramid
        Handles absolute values by considering positive and negative bound as two constraints
            Returns:
                f_constraints: (nGrid*fnum*2*2)x1 vector with friction cone constraints
                    where nGrid*fnum element corresponds to constraints of finger fnum at time nGrid
                    Every element in f_constraints must be >= 0 (lower bound 0, upper bound np.inf)
        """
        lmbda = self.l_unpack(l_flat)

        # Positive bound
        f1_constraints = SX.zeros((self.nGrid, self.fnum * 2))
        # Negative bound
        f2_constraints = SX.zeros((self.nGrid, self.fnum * 2))

        mu = np.sqrt(2) * self.obj_mu  # Inner approximation of cone

        for col in range(self.fnum):
            # abs(fy) <= mu * fx
            f1_constraints[:, 2 * col] = (
                mu * lmbda[:, col * self.l_i] + lmbda[:, col * self.l_i + 1]
            )
            f2_constraints[:, 2 * col] = (
                -1 * lmbda[:, col * self.l_i + 1] + mu * lmbda[:, col * self.l_i]
            )

            # abs(fz) <= mu * fx
            f1_constraints[:, 2 * col + 1] = (
                mu * lmbda[:, col * self.l_i] + lmbda[:, col * self.l_i + 2]
            )
            f2_constraints[:, 2 * col + 1] = (
                -1 * lmbda[:, col * self.l_i + 2] + mu * lmbda[:, col * self.l_i]
            )

        f_constraints = vertcat(f1_constraints, f2_constraints)
        # print(l)
        # print("friction cones: {}".format(f_constraints))
        # quit()
        return f_constraints

    def x_goal_constraint(self, s_flat, a, x_goal):
        """
        Constrain state at end of trajectory to be at x_goal with a slack variable.
        First, just add tolerance
        """
        x, dx = self.s_unpack(s_flat)
        x_end = x[-1, :]

        con_list = []
        for i in range(self.x_dim):
            f = a[i] - (x_goal[0, i] - x_end[0, i]) ** 2
            con_list.append(f)

        return horzcat(*con_list)

    ################################################################################
    # Helper functions
    ################################################################################

    def get_pnorm(self, cp_param):
        """Get pnorm of cp_param tuple"""
        pnorm = 0
        for param in cp_param:
            pnorm += fabs(param) ** self.p
        pnorm = pnorm ** (1 / self.p)

        return pnorm

    def get_grasp_matrix(self, x):
        """
        Get grasp matrix
            Parameters:
                x: object pose [px, py, pz, qw, qx, qy, qz]
        """

        # Transformation matrix from object frame to world frame
        quat_o_2_w = [x[0, 3], x[0, 4], x[0, 5], x[0, 6]]

        G_list = []

        # Calculate G_i (grasp matrix for each finger)
        for c in self.cp_list:
            cp_pos_of = c["position"]  # Position of contact point in object frame
            quat_cp_2_o = c[
                "orientation"
            ]  # Orientation of contact point frame w.r.t. object frame

            S = np.array(
                [
                    [0, -cp_pos_of[2], cp_pos_of[1]],
                    [cp_pos_of[2], 0, -cp_pos_of[0]],
                    [-cp_pos_of[1], cp_pos_of[0], 0],
                ]
            )

            P_i = np.eye(6)
            P_i[3:6, 0:3] = S

            # Orientation of cp frame w.r.t. world frame
            # quat_cp_2_w = quat_o_2_w * quat_cp_2_o
            quat_cp_2_w = utils.multiply_quaternions(quat_o_2_w, quat_cp_2_o)
            # R_i is rotation matrix from contact frame i to world frame
            R_i = utils.get_matrix_from_quaternion(quat_cp_2_w)
            R_i_bar = SX.zeros((6, 6))
            R_i_bar[0:3, 0:3] = R_i
            R_i_bar[3:6, 3:6] = R_i

            G_iT = R_i_bar.T @ P_i.T
            G_list.append(G_iT)

        # GT_full = np.concatenate(G_list)
        GT_full = vertcat(*G_list)
        GT = self.H @ GT_full
        # print(GT.T)
        return GT.T

    def get_M_obj(self):
        """Get 6x6 object inertia matrix"""
        M = np.zeros((6, 6))
        M[0, 0] = M[1, 1] = M[2, 2] = self.obj_mass
        M[3, 3] = self.obj_mass * (self.obj_shape[0] ** 2 + self.obj_shape[2] ** 2) / 12
        M[4, 4] = self.obj_mass * (self.obj_shape[1] ** 2 + self.obj_shape[2] ** 2) / 12
        M[5, 5] = self.obj_mass * (self.obj_shape[0] ** 2 + self.obj_shape[1] ** 2) / 12
        return M

    def get_gapp(self):
        """Compute external gravity force on object, in -z direction"""

        gapp = np.array([[0], [0], [self.gravity * self.obj_mass], [0], [0], [0]])
        return gapp

    def get_R_cp_2_o(self, cp):
        """
        Get 4x4 tranformation matrix from contact point frame to object frame
            Parameters:
                cp: dict with "position" and "orientation" fields in object frame
        """
        # H = SX.zeros((4,4))
        quat = cp["orientation"]
        p = cp["position"]
        R = utils.get_matrix_from_quaternion(quat)
        return R

    def get_R_o_2_w(self, x):
        quat = [x[0, 3], x[0, 4], x[0, 5], x[0, 6]]
        R = utils.get_matrix_from_quaternion(quat)
        return R

    def get_H_o_2_w(self, x):
        """
        Get 4x4 tranformation matrix from object frame to world frame
            Parameters:
                x: object pose [px, py, pz, qw, qx, qy, qz]
        """
        H = SX.zeros((4, 4))

        quat = [x[0, 3], x[0, 4], x[0, 5], x[0, 6]]
        R = utils.get_matrix_from_quaternion(quat)
        p = np.array([x[0, 0], x[0, 1], x[0, 2]])

        H[3, 3] = 1
        H[0:3, 0:3] = R
        H[0:3, 3] = p[:]
        # Test transformation
        # print("calculated: {}".format(H @ np.array([0,0,0,1])))
        # print("actual: {}".format(p))
        return H

    def get_H_w_2_o(self, x):
        """Get 4x4 transformation matrix from world to object frame"""
        H = np.zeros((4, 4))
        quat = [x[0, 3], x[0, 4], x[0, 5], x[0, 6]]
        p = np.array([x[0, 0], x[0, 1], x[0, 2]])
        p_inv, quat_inv = utils.invert_transform(p, quat)
        R = utils.get_matrix_from_quaternion(quat_inv)
        H[3, 3] = 1
        H[0:3, 0:3] = R
        H[0:3, 3] = p_inv[:]
        # Test transformation
        # print("calculated: {}".format(H @ np.array([0,0,1,1])))
        return H

    def get_contact_points_from_cp_params(self, cp_params):
        """
        Get list of contact point dicts given cp_params list
        Each contact point is: {"position_of", "orientation_of"}
        """
        cp_list = []
        for param in cp_params:
            pos_of, quat_of = self.cp_param_to_cp_of(param)
            cp = {"position": pos_of, "orientation": quat_of}
            cp_list.append(cp)
        return cp_list

    def cp_param_to_cp_of(self, cp_param):
        """
        Get contact point position and orientation in object frame (OF)
            Parameters:
                (x_param, y_param, z_param) tuple
        """
        pnorm = self.get_pnorm(cp_param)

        # print("cp_param: {}".format(cp_param))
        # print("pnorm: {}".format(pnorm))

        cp_of = []
        # Get cp position in OF
        for i in range(3):
            cp_of.append(
                -self.obj_shape[i] / 2 + (cp_param[i] + 1) * self.obj_shape[i] / 2
            )
        cp_of = np.asarray(cp_of)

        # TODO: Find analytical way of computing theta
        # Compute derivatives dx, dy, dz of pnorm
        # d_pnorm_list = []
        # for param in cp_param:
        #  d = (param * (fabs(param) ** (self.p - 2))) / (pnorm**(self.p-1))
        #  d_pnorm_list.append(d)

        # print("d_pnorm: {}".format(d_pnorm_list))

        # dx = d_pnorm_list[0]
        # dy = d_pnorm_list[1]
        # dz = d_pnorm_list[2]

        # w = np.sin(np.arctan2(dz*dz+dy*dy, dx)/2)
        # x = 0
        ## This if case is to deal with the -0.0 behavior in arctan2
        # if dx == 0: # TODO: this is going to be an error for through contact opt, when dx is an SX var
        #  y = np.sin(np.arctan2(dz, dx)/2)
        # else:
        #  y = np.sin(np.arctan2(dz, -dx)/2)

        # if dx == 0: # TODO: this is going to be an error for through contact opt, when dx is an SX var
        #  z = np.sin(np.arctan2(-dy, dx)/2)
        # else:
        #  z = np.sin(np.arctan2(-dy, dx)/2)
        # quat = (w,x,y,z)

        x_param = cp_param[0]
        y_param = cp_param[1]
        z_param = cp_param[2]
        # For now, just hard code quat
        if y_param == -1:
            quat = (0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2)
        elif y_param == 1:
            quat = (0, 0, -np.sqrt(2) / 2, np.sqrt(2) / 2)
        elif x_param == 1:
            quat = (0, 1, 0, 0)
        elif z_param == 1:
            quat = (0, np.sqrt(2) / 2, 0, np.sqrt(2) / 2)
        elif x_param == -1:
            quat = (0, 0, 0, 1)
        elif z_param == -1:
            quat = (0, -np.sqrt(2) / 2, 0, np.sqrt(2) / 2)

        return cp_of, quat

    def test_cp_param_to_cp_of(self):
        print("\nP1")
        p1 = (0, -1, 0)
        q = self.cp_param_to_cp_of(p1)
        print("quat: {}".format(q))

        print("\nP2")
        p2 = (0, 1, 0)
        q = self.cp_param_to_cp_of(p2)
        print("quat: {}".format(q))

        print("\nP3")
        p3 = (1, 0, 0)
        q = self.cp_param_to_cp_of(p3)
        print("quat: {}".format(q))

        print("\nP4")
        p4 = (0, 0, 1)
        q = self.cp_param_to_cp_of(p4)
        print("quat: {}".format(q))

        print("\nP5")
        p5 = (-1, 0, 0)
        q = self.cp_param_to_cp_of(p5)
        print("quat: {}".format(q))

        print("\nP6")
        p6 = (0, 0, -1)
        q = self.cp_param_to_cp_of(p6)
        print("quat: {}".format(q))

    ################################################################################
    # Path constraints
    ################################################################################

    def path_constraints(
        self,
        z,
        x0,
        x_goal=None,
        l0=None,
        dx0=None,
        dx_end=None,
    ):
        """
        Define upper and lower bounds for decision variables
        Constrain initial x, q
        Constrain l if specified
        Constrain initial and final object velocity, if specified
        """
        if self.log_file is not None:
            with open(self.log_file, "a+") as f:
                f.write("\nPath constraints: {}\n")

        t, s_flat, l_flat, a = self.decvar_unpack(z)

        nGrid = self.nGrid

        # Time bounds
        t_range = [0, self.tf]  # initial and final time
        t_lb = np.linspace(t_range[0], t_range[1], nGrid)  # lower bound
        t_ub = t_lb  # upper bound
        # print("Timestamps: {}".format(t_lb))

        # State path constraints
        # Unpack state vector
        x, dx = self.s_unpack(s_flat)  # Object pose constraints
        x_range = np.array(
            [
                [-0.15, 0.15],  # x coord range
                [-0.15, 0.15],  # y coord range
                [0.01, 0.15],  # z coord range TODO Hardcoded
                [-np.inf, np.inf],  # qx range
                [-np.inf, np.inf],  # qy range
                [-np.inf, np.inf],  # qz range
                [-np.inf, np.inf],  # qw range
            ]
        )
        x_lb = np.ones(x.shape) * x_range[:, 0]
        x_ub = np.ones(x.shape) * x_range[:, 1]

        # Object pose boundary contraint (starting position of object)
        if self.log_file is not None:
            with open(self.log_file, "a+") as f:
                f.write("Constrain x0 to {}\n".format(x0))
        x_lb[0] = x0
        x_ub[0] = x0
        # if x_goal is not None:
        #  x_lb[-1] = x_goal
        #  x_ub[-1] = x_goal
        #  # Just z goal
        #  #x_lb[-1,1] = x_goal[0,1]
        #  #x_ub[-1,1] = x_goal[0,1]

        # Object velocity constraints
        dx_range = np.array(
            [
                [-0.05, 0.05],  # x vel range
                [-0.05, 0.05],  # y vel range
                [-0.05, 0.05],  # z vel range
                [-np.pi / 2, np.pi / 2],  # angular velocity range
                [-np.pi / 2, np.pi / 2],  # angular velocity range
                [-np.pi / 2, np.pi / 2],  # angular velocity range
            ]
        )
        dx_lb = np.ones(dx.shape) * dx_range[:, 0]
        dx_ub = np.ones(dx.shape) * dx_range[:, 1]
        if dx0 is not None:
            if self.log_file is not None:
                with open(self.log_file, "a+") as f:
                    f.write("Constrain dx0 to {}\n".format(dx0))
            dx_lb[0] = dx0
            dx_ub[0] = dx0
        if dx_end is not None:
            if self.log_file is not None:
                with open(self.log_file, "a+") as f:
                    f.write("Constrain dx_end to {}\n".format(dx_end))
            dx_lb[-1] = dx_end
            dx_ub[-1] = dx_end

        # Contact force contraints
        # For now, just define min and max forces
        l = self.l_unpack(l_flat)
        l_epsilon = 0
        # Limits for one finger
        f1_l_range = np.array(
            [
                [0, np.inf],  # c1 fn force range
                [-np.inf, np.inf],  # c1 ft force range
                [-np.inf, np.inf],  # c1 ft force range
                # [-np.inf, np.inf], # c1 ft force range
            ]
        )
        l_range = np.tile(f1_l_range, (self.fnum, 1))
        l_lb = np.ones(l.shape) * l_range[:, 0]
        l_ub = np.ones(l.shape) * l_range[:, 1]
        # Initial contact force constraints
        if l0 is not None:
            if self.log_file is not None:
                with open(self.log_file, "a+") as f:
                    f.write("Constrain l0 to {}\n".format(l0))
            l_lb[0] = l0
            l_ub[0] = l0

        # Pack state contraints
        s_lb = self.s_pack(x_lb, dx_lb)
        s_ub = self.s_pack(x_ub, dx_ub)

        a_lb = np.zeros(a.shape)
        a_ub = np.ones(a.shape) * np.inf

        # Pack the constraints for all dec vars
        z_lb = self.decvar_pack(t_lb, s_lb, self.l_pack(l_lb), a_lb)
        z_ub = self.decvar_pack(t_ub, s_ub, self.l_pack(l_ub), a_ub)

        return z_lb, z_ub

    def get_initial_guess(self, z_var, x0, x_goal):
        """Set initial trajectory guess. For now, just define everything to be 0"""

        t_var, s_var, l_var, a_var = self.decvar_unpack(z_var)

        # Define time points to be equally spaced
        t_traj = np.linspace(0, self.tf, self.nGrid)

        x_var, dx_var = self.s_unpack(s_var)
        dx_traj = np.zeros(dx_var.shape)
        x_traj = np.squeeze(np.linspace(x0, x_goal, self.nGrid))
        s_traj = self.s_pack(x_traj, dx_traj)

        l_traj = np.ones(self.l_unpack(l_var).shape)
        # l_traj[:,2] = 0.1
        # l_traj[:,8] = 0.1
        # l_traj[:,0] = 1
        # l_traj[:,6] = 1

        a_traj = np.zeros(a_var.shape)

        z_traj = self.decvar_pack(t_traj, s_traj, self.l_pack(l_traj), a_traj)

        return z_traj

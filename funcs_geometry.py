import numpy as np

from numpy.typing import NDArray


def forward_points(L: float, theta_vec: NDArray[np.float_]) -> [NDArray[np.float_], NDArray[np.float_]]:
    """
    translate angles to x,y coords of tip

    inputs:
    L      - float, length of truss
    theta1 - (iterations, ) shaped array, of angles of 1st hinge in time during measurement
    theta2 - (iterations, ) shaped array, of angles of 2nd hinge in time during measurement

    outputs:
    pos_vec - (2, 2, iterations) shaped array of x y coords of truss tip s
    """
    theta_rads = np.deg2rad(theta_vec)
    if np.shape(np.shape(theta_rads)) == (1,):  # only a single theta
        p0: NDArray([np.float_]) = np.zeros((np.shape(theta_rads)[0], 2))  # origin
        pos_vec: NDArray([np.float_]) = np.zeros((np.shape(theta_rads)[0], 2))
        for i, theta in enumerate(theta_rads):
            if i == 0:
                pos_prev: NDArray([np.float_]) = p0[0, :]
            else:
                pos_prev: NDArray([np.float_]) = pos_vec[i-1, :]
            thetas_sum: float = np.sum(theta_rads[:i+1])
            pos_vec[i, :] = pos_prev + L*np.array([-np.sin(thetas_sum), np.cos(thetas_sum)])
    else:  # all thetas in time
        p0 = np.zeros((np.shape(theta_rads)[0], np.shape(theta_rads)[1], 2))  # origin
        pos_vec = np.zeros((np.shape(theta_rads)[0], np.shape(theta_rads)[1], 2))
        for i, thetas in enumerate(theta_rads):
            # print('thetas ', thetas)
            for j, theta in enumerate(thetas):
                # print('theta ', theta)
                if j == 0:
                    pos_prev: NDArray([np.float_]) = p0[0, 0, :]
                else:
                    pos_prev: NDArray([np.float_]) = pos_vec[i, j-1, :]
                thetas_sum: float = np.sum(thetas[:j+1])
                # print('sum ', thetas_sum)
                pos_vec[i, j, :] = pos_prev + L*np.array([-np.sin(thetas_sum), np.cos(thetas_sum)])
    # p0 = np.array([0.0, 0.0])
    # (2, hinges, iterations)

    # p1 = p0[0, :, :] + L*np.array([-np.sin(theta_rads[0, :]), np.cos(theta_rads[1, :])])
    # p2 = p1 + L*np.array([-np.sin(theta_rads[0, :]+theta_rads[1, :]), np.cos(theta_rads[0, :]+theta_rads[1, :])])
    # return p1, p2, pos_vec
    return pos_vec


def theta_from_xy(x, y, buckle, L=1.0):
    """
    Return (theta1, theta2) where:
      - theta2 is the relative elbow angle (same as before).
      - theta1 is the base angle measured from the VERTICAL (+y axis).
        If ccw_from_vertical=False, it will be clockwise-from-vertical.

    buckle = +1  -> elbow-up branch
    buckle = -1  -> elbow-down branch
    """
    # IK for 2R with L1=L2=L
    c2 = (x**2 + y**2 - 2*L**2) / (2*L**2)
    c2 = np.clip(c2, -1.0, 1.0)

    s2 = np.sqrt(max(0.0, 1.0 - c2**2))
    if buckle == 1:
        theta2 = np.arctan2(+s2, c2)
    else:
        theta2 = np.arctan2(-s2, c2)

    # q1 measured from +x (standard)
    q1_from_x = np.arctan2(y, x) - theta2/2
    theta1 = q1_from_x - np.pi/2

    return theta1, theta2

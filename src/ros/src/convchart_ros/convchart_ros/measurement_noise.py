"""Measurement-noise covariance R for the Kalman filter: the 882k release model's measured pose error as a
function of range. Source: Conv-ChArT paper/results_rev6/26_pose_error_variance/kalman_R_REL882.json
(1000-frame B1 benchmark, 893 accepted solves; robust sigma per apparent-scale octave fitted as a power law
in the range z = tvec_z / square_length, in board squares). The pipeline's per-frame analytic pose_cov is NOT
a filter input (about 6x over-confident, NEES 11.2 against 6); this replaces it on the wire."""
import numpy as np

_A = np.array([1.1196e-04, 1.5808e-04, 3.8022e-05, 5.2027e-05, 4.6866e-05, 1.0236e-04])
_K = np.array([0.9674, 0.8074, 1.0171, 1.5016, 1.4944, 1.7337])
Z_FITTED = (3.88, 70.67)


def measurement_covariance(tvec_z: float, square_length_m: float) -> tuple[np.ndarray, bool]:
    """(6x6 diagonal covariance in PoseWithCovariance order (x, y, z, rot_x, rot_y, rot_z),
    in_fitted_range).
    """
    z = float(tvec_z) / float(square_length_m)
    zc = min(max(z, Z_FITTED[0]), Z_FITTED[1])
    sigma = _A * zc ** _K
    sigma[3:] *= square_length_m
    return np.diag(np.r_[sigma[3:], sigma[:3]] ** 2), zc == z

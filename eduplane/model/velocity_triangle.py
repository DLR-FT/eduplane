# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0

# Velocity triangle (i.e. non-accelerated kinematic model) of the Extended Dubins Airplane

from eduplane.utils.smath import (
    Piecewise, acos, asin, atan, atan2, cos, sin, smath_func, sqrt, tan)

symbols = {
    # angles
    "gamma": {
        "name": r"\gamma",
        "dimension": "dimensionless",
        "unit": None,
        "definition": "asin(-w_K_g/V_K_xy)",
    },
    "gamma_A": {
        "name": r"\gamma_{A}",
        "dimension": "dimensionless",
        "unit": None,
        "definition": "asin(-w_A_g/V_A_xy)",
    },
    "gamma_W": {
        "name": r"\gamma_{W}",
        "dimension": "dimensionless",
        "unit": None,
        "definition": "asin(-w_W_g/V_W_xy)",
        "assumptions": {
            "constant": True
        },
    },
    "chi": {
        "name": r"\chi",
        "dimension": "dimensionless",
        "unit": None,
        "definition": "atan2(v_K_g, u_K_g)",
    },
    "chi_A": {
        "name": r"\chi_{A}",
        "dimension": "dimensionless",
        "unit": None,
        "definition": "atan2(v_A_g, u_A_g)",
    },
    "chi_W": {
        "name": r"\chi_{W}",
        "dimension": "dimensionless",
        "unit": None,
        "definition": "atan2(v_W_g, u_W_g)",
        "assumptions": {
            "constant": True
        },
    },
    "chi_D": {
        "name": r"\chi_{D}",
        "dimension": "dimensionless",
        "unit": None,
        "definition": "chi_A - chi",
    },
    # speeds
    "V_A": {
        "name": r"V_{A}",
        "dimension": "velocity",
        "unit": "m/s",
        "assumptions": {
            "positive": True
        },
        "definition": "sqrt(u_A_g**2 + v_A_g**2 + w_A_g**2)",
    },
    "V_K": {
        "name": r"V_{K}",
        "dimension": "velocity",
        "unit": "m/s",
        "definition": "sqrt(u_K_g**2 + v_K_g**2 + w_K_g**2)",
        "assumptions": {
            "nonnegative": True
        },
    },
    "V_W": {
        "name": r"V_{W}",
        "dimension": "velocity",
        "unit": "m/s",
        "assumptions": {
            "nonnegative": True,
            "constant": True
        },
        "definition": "sqrt(u_W_g**2 + v_W_g**2 + w_W_g**2)",
    },
    # horizontal speed components
    "V_A_xy": {
        "name": r"V_{A,xy}",
        "dimension": "velocity",
        "unit": "m/s",
        "assumptions": {
            "positive": True
        },
        "definition": "V_A*cos(gamma_A)",
    },
    "V_K_xy": {
        "name": r"V_{K,xy}",
        "dimension": "velocity",
        "unit": "m/s",
        "assumptions": {
            "nonnegative": True
        },
        "definition": "V_K*cos(gamma)",
    },
    "V_W_xy": {
        "name": r"V_{W,xy}",
        "dimension": "velocity",
        "unit": "m/s",
        "assumptions": {
            "nonnegative": True,
            "constant": True
        },
        "definition": "V_W*cos(gamma_W)",
    },
    # velocity, airspeed and wind speed components (geodetic frame)
    "u_K_g": {
        "name": r"u_{Kg}",
        "dimension": "velocity",
        "unit": "m/s"
    },
    "v_K_g": {
        "name": r"v_{Kg}",
        "dimension": "velocity",
        "unit": "m/s"
    },
    "w_K_g": {
        "name": r"w_{Kg}",
        "dimension": "velocity",
        "unit": "m/s"
    },
    "u_A_g": {
        "name": r"u_{Ag}",
        "dimension": "velocity",
        "unit": "m/s"
    },
    "v_A_g": {
        "name": r"v_{Ag}",
        "dimension": "velocity",
        "unit": "m/s"
    },
    "w_A_g": {
        "name": r"w_{Ag}",
        "dimension": "velocity",
        "unit": "m/s"
    },
    "u_W_g": {
        "name": r"u_{Ag}",
        "dimension": "velocity",
        "unit": "m/s",
        "assumptions": {
            "constant": True
        },
    },
    "v_W_g": {
        "name": r"v_{Ag}",
        "dimension": "velocity",
        "unit": "m/s",
        "assumptions": {
            "constant": True
        },
    },
    "w_W_g": {
        "name": r"w_{Ag}",
        "dimension": "velocity",
        "unit": "m/s",
        "assumptions": {
            "constant": True
        },
    },
}

# Horizontal velocity / speed magnitudes


@smath_func(symbols, "V_A_xy")
def V_A_xy_from_V_K_xy_chi(V_K_xy, chi, V_W_xy, chi_W):
    return sqrt(V_K_xy**2 + V_W_xy**2 - 2*V_K_xy*V_W_xy*cos(chi_W - chi))


@smath_func(symbols, "V_K_xy")
def V_K_xy_from_V_A_xy_chi_A(V_A_xy, chi_A, V_W_xy, chi_W):
    return sqrt(V_A_xy**2 + V_W_xy**2 + 2*V_A_xy*V_W_xy*cos(chi_A - chi_W))


@smath_func(symbols, "V_W_xy")
def V_W_xy_from_V_A_xy_V_K_xy(V_A_xy, chi_A, V_K_xy, chi):
    return sqrt(V_A_xy**2 + V_K_xy**2 - 2*V_A_xy*V_K_xy*cos(chi_A - chi))


@smath_func(symbols, "V_A_xy")
def V_A_xy_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W):
    V_K_xy_parallel = sqrt(V_K_xy**2 - V_W_xy**2*sin(chi_W - chi_A)**2)
    V_W_xy_parallel = V_W_xy*cos(chi_W - chi_A)
    return V_K_xy_parallel - V_W_xy_parallel


@smath_func(symbols, "V_K_xy")
def V_K_xy_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W):
    V_A_xy_parallel = sqrt(V_A_xy**2 - V_W_xy**2*sin(chi_W - chi)**2)
    V_W_xy_parallel = V_W_xy*cos(chi_W - chi)
    return V_A_xy_parallel + V_W_xy_parallel


# Flight path azimuth (i.e. ground track angle for coordinated flight)


@smath_func(symbols, "chi")
def chi(u_K_g, v_K_g):
    return atan2(v_K_g, u_K_g)


@smath_func(symbols, "chi")
def chi_from_V_A_xy_chi_A(V_A_xy, chi_A, V_W_xy, chi_W):
    return atan2(
        V_A_xy*sin(chi_A) + V_W_xy*sin(chi_W),
        V_A_xy*cos(chi_A) + V_W_xy*cos(chi_W),
    )


@smath_func(symbols, "chi")
def chi_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W):
    return chi_A - asin(V_W_xy/V_K_xy*sin(chi_A - chi_W))


# Air-path azimuth (i.e. heading angle for coordinated flight)


@smath_func(symbols, "chi_A")
def chi_A(u_A_g, v_A_g):
    return atan2(v_A_g, u_A_g)


@smath_func(symbols, "chi_A")
def chi_A_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W):
    return chi + asin(V_W_xy/V_A_xy*sin(chi - chi_W))


@smath_func(symbols, "chi_A")
def chi_A_from_V_K_xy_chi(V_K_xy, chi, V_W_xy, chi_W):
    return atan2(
        V_K_xy*sin(chi) - V_W_xy*sin(chi_W),
        V_K_xy*cos(chi) - V_W_xy*cos(chi_W))


# Wind vector azimuth (aka direction of wind flow)


@smath_func(symbols, "chi_W")
def chi_W(u_W_g, v_W_g):
    return atan2(v_W_g, u_W_g)


# Drift angle (flight path azimuth deviation aka wind correction angle)


@smath_func(symbols, "chi_D")
def chi_D_from_V_A_xy_chi(V_A_xy, chi, V_W_xy, chi_W):
    return asin(V_W_xy/V_A_xy*sin(chi - chi_W))


@smath_func(symbols, "chi_D")
def chi_D_from_V_K_xy_chi_A(V_K_xy, chi_A, V_W_xy, chi_W):
    return asin(V_W_xy/V_K_xy*sin(chi_A - chi_W))


@smath_func(symbols, "chi_D")
def chi_D_from_V_A_xy_V_K_xy(V_A_xy, V_K_xy, V_W_xy, chi_A, chi_W):
    return Piecewise(
        (
            -acos((V_A_xy**2 + V_K_xy**2 - V_W_xy**2)/(2*V_A_xy*V_K_xy)),
            chi_A < chi_W,
        ),
        (acos((V_A_xy**2 + V_K_xy**2 - V_W_xy**2)/(2*V_A_xy*V_K_xy)), True),
    )


# North / east speed components and azimuth angles


@smath_func(symbols, "u_K_g")
def u_K_g_from_V_K_xy(V_K_xy, chi):
    return V_K_xy*cos(chi)


@smath_func(symbols, "v_K_g")
def v_K_g_from_V_K_xy(V_K_xy, chi):
    return V_K_xy*sin(chi)


@smath_func(symbols, "u_A_g")
def u_A_g_from_V_A_xy(V_A_xy, chi_A):
    return V_A_xy*cos(chi_A)


@smath_func(symbols, "v_A_g")
def v_A_g_from_V_A_xy(V_A_xy, chi_A):
    return V_A_xy*sin(chi_A)


@smath_func(symbols, "u_W_g")
def u_W_g_from_V_W_xy(V_W_xy, chi_W):
    return V_W_xy*cos(chi_W)


@smath_func(symbols, "v_W_g")
def v_W_g_from_V_W_xy(V_W_xy, chi_W):
    return V_W_xy*sin(chi_W)


# Vertical speeds


@smath_func(symbols, "w_K_g")
def w_K_g_from_w_A_g_w_W_g(w_A_g, w_W_g):
    return w_A_g + w_W_g


@smath_func(symbols, "w_K_g")
def w_A_g_from_w_K_g_w_W_g(w_K_g, w_W_g):
    return w_K_g - w_W_g


@smath_func(symbols, "w_W_g")
def w_W_g_from_w_A_g_w_K_g(w_A_g, w_K_g):
    return w_K_g - w_A_g


# Vertical and horizontal airspeed components and air-path inclination angle


@smath_func(symbols, "gamma_A")
def gamma_A_from_V_A_xy_w_A_g(V_A_xy, w_A_g):
    return atan(-w_A_g/V_A_xy)


@smath_func(symbols, "gamma_A")
def gamma_A_from_V_A_w_A_g(V_A, w_A_g):
    return asin(-w_A_g/V_A)


@smath_func(symbols, "gamma_A")
def gamma_A_from_V_A_V_K_gamma(V_A, V_K, gamma, w_W_g):
    return asin((V_K*sin(gamma) + w_W_g)/V_A)


@smath_func(symbols, "w_A_g")
def w_A_g_from_V_A_gamma_A(V_A, gamma_A):
    return -sin(gamma_A)*V_A


@smath_func(symbols, "w_A_g")
def w_A_g_from_V_A_xy_gamma_A(V_A_xy, gamma_A):
    return -tan(gamma_A)*V_A_xy


@smath_func(symbols, "w_A_g")
def w_A_g_abs_from_V_A_V_A_xy(V_A, V_A_xy):
    return sqrt(V_A**2 - V_A_xy**2)


@smath_func(symbols, "V_A_xy")
def V_A_xy_from_V_A_gamma_A(V_A, gamma_A):
    return cos(gamma_A)*V_A


@smath_func(symbols, "V_A_xy")
def V_A_xy_from_V_A_w_A_g(V_A, w_A_g):
    return sqrt(V_A**2 - w_A_g**2)


@smath_func(symbols, "V_A_xy")
def V_A_xy_from_w_A_g_gamma_A(w_A_g, gamma_A):
    return -w_A_g/tan(gamma_A)


@smath_func(symbols, "V_A")
def V_A_from_V_A_xy_gamma_A(V_A_xy, gamma_A):
    return V_A_xy/cos(gamma_A)


# Vertical and horizontal velocity components and flight path inclination angle


@smath_func(symbols, "gamma")
def gamma_from_V_K_xy_w_K_g(V_K_xy, w_K_g):
    return atan(-w_K_g/V_K_xy)


@smath_func(symbols, "gamma")
def gamma_from_V_K_w_K_g(V_K, w_K_g):
    return asin(-w_K_g/V_K)


@smath_func(symbols, "gamma")
def gamma_from_V_K_V_A_gamma_A(V_K, V_A, gamma_A, w_W_g):
    return asin((V_A*sin(gamma_A) - w_W_g)/V_K)


@smath_func(symbols, "w_K_g")
def w_K_g_from_V_K_gamma(V_K, gamma):
    return -sin(gamma)*V_K


@smath_func(symbols, "w_K_g")
def w_K_g_from_V_K_xy_gamma(V_K_xy, gamma):
    return -tan(gamma)*V_K_xy


@smath_func(symbols, "V_K_xy")
def V_K_xy_from_V_K_gamma(V_K, gamma):
    return cos(gamma)*V_K


@smath_func(symbols, "V_K_xy")
def V_K_xy_from_w_K_g_gamma(w_K_g, gamma):
    return -w_K_g/tan(gamma)


@smath_func(symbols, "V_A")
def V_K_from_V_K_xy_gamma(V_K_xy, gamma):
    return V_K_xy/cos(gamma)


# Vertical and horizontal wind speed components and wind speed vector
# inclination angle


# TODO: It would be sensible and certainly more robust, to define this function
#       piecewise for V_W_xy > 0 and V_W_xy = 0, like this:
#         Piecewise((atan(-w_W_g/V_W_xy), V_W_xy > 0, (0.0, True))
#       However, sympy.piecewise is useless with the numpy backend, as it
#       doesn't avoid any ZeroDivisionErrors.
#       Once the following issues are solved, we could add the piecewise
#       implementation:
#         - https://github.com/sympy/sympy/issues/15014
#         - https://github.com/sympy/sympy/issues/14789
@smath_func(symbols, "gamma_W")
def gamma_W_from_V_W_xy_w_W_g(V_W_xy, w_W_g):
    return atan(-w_W_g/V_W_xy)


# TODO: It would be sensible and certainly more robust, to define this function
#       piecewise for V_W_xy > 0 and V_W_xy = 0, like this:
#         Piecewise((asin(-w_W_g/V_W_xy), V_W_xy > 0, (0.0, True))
#       However, sympy.piecewise is useless with the numpy backend, as it
#       doesn't avoid any ZeroDivisionErrors.
#       Once the following issues are solved, we could add the piecewise
#       implementation:
#         - https://github.com/sympy/sympy/issues/15014
#         - https://github.com/sympy/sympy/issues/14789
@smath_func(symbols, "gamma_W")
def gamma_W_from_V_W_w_W_g(V_W, w_W_g):
    return asin(-w_W_g/V_W)


@smath_func(symbols, "w_W_g")
def w_W_g_from_V_W_gamma_W(V_W, gamma):
    return -sin(gamma)*V_W


@smath_func(symbols, "w_W_g")
def w_W_g_from_V_W_xy_gamma(V_W_xy, gamma_W):
    return -tan(gamma_W)*V_W_xy


@smath_func(symbols, "V_W_xy")
def V_W_xy_from_V_W_gamma_W(V_W, gamma_W):
    return cos(gamma_W)*V_W


@smath_func(symbols, "V_W_xy")
def V_W_xy_from_w_W_g_gamma(w_W_g, gamma):
    return -w_W_g/tan(gamma)


@smath_func(symbols, "V_A")
def V_W_from_V_W_xy_gamma(V_W_xy, gamma):
    return V_W_xy/cos(gamma)


# Speed magnitudes from horizontal and vertical components


@smath_func(symbols, "V_A")
def V_A_from_V_A_xy_w_A_g(V_A_xy, w_A_g):
    return sqrt(V_A_xy**2 + w_A_g**2)


@smath_func(symbols, "V_A")
def V_K_from_V_K_xy_w_K_g(V_K_xy, w_K_g):
    return sqrt(V_K_xy**2 + w_K_g**2)


@smath_func(symbols, "V_W")
def V_W_from_V_W_xy_w_W_g(V_W_xy, w_W_g):
    return sqrt(V_W_xy**2 + w_W_g**2)


# Speed magnitude from airspeed and path angles
# (3D velocity triangle solution)
# Returns positive and maximum solutions (i.e. if two valid positive solutions
# exist, the greater speed magnitude is returned).
@smath_func(symbols, "V_K")
def V_K_from_V_A_chi_gamma(V_A, chi, gamma, V_W, chi_W, gamma_W):
    cos_gamma = cos(gamma)
    cos_gamma_W = cos(gamma_W)
    n_K_V_W = (
        cos(chi)*cos_gamma*V_W*cos(chi_W)*cos_gamma_W +
        sin(chi)*cos_gamma*V_W*sin(chi_W)*cos_gamma_W +
        sin(gamma)*V_W*sin(gamma_W))
    radicand = n_K_V_W**2 + V_A**2 - V_W**2
    return n_K_V_W + sqrt(radicand)


@smath_func(symbols, "V_A")
def V_A_from_V_K_chi_A_gamma_A(V_K, chi_A, gamma_A, V_W, chi_W, gamma_W):
    cos_gamma_A = cos(gamma_A)
    cos_gamma_W = cos(gamma_W)
    n_A_V_W = (
        cos(chi_A)*cos_gamma_A*V_W*cos(chi_W)*cos_gamma_W +
        sin(chi_A)*cos_gamma_A*V_W*sin(chi_W)*cos_gamma_W +
        sin(gamma_A)*V_W*sin(gamma_W))
    radicand = n_A_V_W**2 + V_K**2 - V_W**2
    return -n_A_V_W + sqrt(radicand)

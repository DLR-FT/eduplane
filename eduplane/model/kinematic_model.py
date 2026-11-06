# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0

# Kinematics of the Extended Dubins Airplane

from eduplane.utils.smath import atan, cos, sin, smath_func, sqrt, tan

from . import velocity_triangle

symbols = {
    # NED position
    "x_g": {
        "name": r"x_{g}",
        "dimension": "length",
        "unit": "m"
    },
    "y_g": {
        "name": r"y_{g}",
        "dimension": "length",
        "unit": "m"
    },
    "z_g": {
        "name": r"z_{g}",
        "dimension": "length",
        "unit": "m"
    },
    # Height above origin
    "h_g": {
        "name": r"h_{g}",
        "dimension": "length",
        "unit": "m",
        "definition": "-z_g"
    },
    # angles, angular rates
    "gamma_dot": {
        "name": r"\dot{\gamma}",
        "dimension": "1/time",
        "unit": "1/s",
        "definition": "diff(gamma)"
    },
    "chi_dot": {
        "name": r"\dot{\chi}",
        "dimension": "1/time",
        "unit": "1/s",
        "definition": "diff(chi)"
    },
    "gamma_A_dot": {
        "name": r"\dot{\gamma}_{A}",
        "dimension": "1/time",
        "unit": "1/s",
        "definition": "diff(gamma_A)"
    },
    "chi_A_dot": {
        "name": r"\dot{\chi}_{A}",
        "dimension": "1/time",
        "unit": "1/s",
        "definition": "diff(chi_A)"
    },
    "chi_A_dot_abs": {
        "name": r"|\dot{\chi}_{A}|",
        "dimension": "1/time",
        "unit": "1/s",
        "definition": "abs(chi_A_dot)",
        "assumptions": {
            "positive": True
        },
    },
    "chi_D_dot": {
        "name": r"\dot{\chi}_{D}",
        "dimension": "1/time",
        "unit": "1/s",
        "definition": "diff(chi_D)"
    },
    "mu_A": {
        "name": r"\mu_{A}",
        "dimension": "dimensionless",
        "unit": None
    },
    # air-relative accelerations and components
    "V_A_dot": {
        "name": r"\dot{V}_{A}",
        "dimension": "acceleration",
        "unit": "m/s**2",
        "definition": "diff(V_A)"
    },
    "V_A_xy_dot": {
        "name": r"\dot{V}_{A,xy}",
        "dimension": "acceleration",
        "unit": "m/s**2",
        "definition": "cos(gamma_A)*V_A_dot"
    },
    "u_A_g_dot": {
        "name": r"\dot{u}_{Ag}",
        "dimension": "acceleration",
        "unit": "m/s**2",
        "definition": "diff(u_A_g)"
    },
    "v_A_g_dot": {
        "name": r"\dot{v}_{Ag}",
        "dimension": "acceleration",
        "unit": "m/s**2",
        "definition": "diff(v_A_g)"
    },
    "w_A_g_dot": {
        "name": r"\dot{w}_{Ag}",
        "dimension": "acceleration",
        "unit": "m/s**2",
        "definition": "diff(w_A_g)"
    },
    # accelerations and components
    "V_K_dot": {
        "name": r"\dot{V}_{K}",
        "dimension": "acceleration",
        "unit": "m/s**2",
        "definition": "diff(V_K)"
    },
    "V_K_xy_dot": {
        "name": r"\dot{V}_{K,xy}",
        "dimension": "acceleration",
        "unit": "m/s**2",
        "definition": "cos(gamma)*V_K_dot"
    },
    "u_K_g_dot": {
        "name": r"\dot{u}_{Kg}",
        "dimension": "acceleration",
        "unit": "m/s**2",
        "definition": "diff(u_K_g)"
    },
    "v_K_g_dot": {
        "name": r"\dot{v}_{Kg}",
        "dimension": "acceleration",
        "unit": "m/s**2",
        "definition": "diff(v_K_g)"
    },
    "w_K_g_dot": {
        "name": r"\dot{w}_{Kg}",
        "dimension": "acceleration",
        "unit": "m/s**2",
        "definition": "diff(w_K_g)"
    },
    # constants / others
    "m": {
        "name": r"m",
        "dimension": "mass",
        "unit": "kg",
        "assumptions": {
            "positive": True,
            "constant": True
        }
    },
    "g": {
        "name": r"g",
        "dimension": "acceleration",
        "unit": "m/s**2",
        "assumptions": {
            "positive": True,
            "constant": True
        }
    },
    **velocity_triangle.symbols
}

# Equations of motion


@smath_func(symbols, "u_A_g")
def u_A_g_from_V_A(V_A, chi_A, gamma_A):
    return V_A*cos(chi_A)*cos(gamma_A)


@smath_func(symbols, "v_A_g")
def v_A_g_from_V_A(V_A, chi_A, gamma_A):
    return V_A*sin(chi_A)*cos(gamma_A)


@smath_func(symbols, "w_A_g")
def w_A_g_from_V_A(V_A, gamma_A):
    return V_A*sin(gamma_A)


@smath_func(symbols, "u_A_g_dot")
def u_A_g_dot_from_V_A_dot(
        V_A, V_A_dot, chi_A, chi_A_dot, gamma_A, gamma_A_dot):
    return (
        V_A_dot*cos(chi_A)*cos(gamma_A) + V_A*(
            -chi_A_dot*sin(chi_A)*cos(gamma_A) -
            gamma_A_dot*cos(chi_A)*sin(gamma_A)))


@smath_func(symbols, "v_A_g_dot")
def v_A_g_dot_from_V_A_dot(
        V_A, V_A_dot, chi_A, chi_A_dot, gamma_A, gamma_A_dot):
    return (
        V_A_dot*sin(chi_A)*cos(gamma_A) + V_A*(
            chi_A_dot*cos(chi_A)*cos(gamma_A) -
            gamma_A_dot*sin(chi_A)*sin(gamma_A)))


@smath_func(symbols, "w_A_g_dot")
def w_A_g_dot_from_gamma_A_dot(V_A, V_A_dot, gamma_A, gamma_A_dot):
    return -V_A_dot*sin(gamma_A) - V_A*(gamma_A_dot*cos(gamma_A))


# Turn rate, air-path inclination rate and roll angle (derived from load factor equations)


@smath_func(symbols, "chi_A_dot")
def chi_A_dot_from_mu_A(V_A, gamma_A, gamma_A_dot, mu_A, g):
    return tan(mu_A)*(g/V_A + gamma_A_dot/cos(gamma_A))


@smath_func(symbols, "gamma_A_dot")
def gamma_A_dot_from_mu_A(V_A, chi_A_dot, gamma_A, mu_A, g):
    return cos(gamma_A)*(chi_A_dot/tan(mu_A) - g/V_A)


@smath_func(symbols, "mu_A")
def mu_A_from_chi_A_dot_gamma_A_dot(V_A, chi_A_dot, gamma_A, gamma_A_dot, g):
    return atan(chi_A_dot*V_A*cos(gamma_A)/(g*cos(gamma_A) + gamma_A_dot*V_A))


# Azimuth rates


# Air-path azimuth angle rate from flight path azimuth angle rate
# and horizontal airspeed rate (derived from drift angle)
# TODO: The function should be defined piecewise for V_W_xy > 0 and
#       V_W_xy = 0, like this:
#         Piecewise((chi_dot - num/denom, V_W_xy > 0), (chi_dot, True))
#       However, sympy.piecewise is useless with the numpy backend, as it
#       doesn't avoid any ZeroDivisionErrors.
#       Once the following issues are solved, we could add the piecewise
#       implementation:
#         - https://github.com/sympy/sympy/issues/15014
#         - https://github.com/sympy/sympy/issues/14789
@smath_func(symbols, "chi_A_dot")
def chi_A_dot_from_chi_dot_V_A_xy_dot(
        V_A_xy, V_A_xy_dot, chi, chi_dot, V_W_xy, chi_W):
    num = chi_dot*cos(chi - chi_W) - V_A_xy_dot/V_A_xy*sin(chi - chi_W)
    denom = sqrt(V_A_xy**2/V_W_xy**2 - sin(chi - chi_W)**2)
    return num/denom + chi_dot


# Air-path azimuth angle rate from flight path azimuth angle rate
# and horizontal acceleration (derived from drift angle)
# TODO: The function should be defined piecewise for V_W_xy > 0 and
#       V_W_xy = 0, like this:
#         Piecewise((chi_dot - num/denom, V_W_xy > 0), (chi_dot, True))
#       However, sympy.piecewise is useless with the numpy backend, as it
#       doesn't avoid any ZeroDivisionErrors.
#       Once the following issues are solved, we could add the piecewise
#       implementation:
#         - https://github.com/sympy/sympy/issues/15014
#         - https://github.com/sympy/sympy/issues/14789
@smath_func(symbols, "chi_A_dot")
def chi_A_dot_from_chi_dot_V_K_xy_dot(
        V_K_xy, V_K_xy_dot, chi_A, chi_dot, V_W_xy, chi_W):
    sqrt_term = sqrt(V_K_xy**2/V_W_xy**2 - sin(chi_A - chi_W)**2)
    num = chi_dot*sqrt_term - V_K_xy_dot/V_K_xy*sin(chi_A - chi_W)
    denom = sqrt_term - cos(chi_A - chi_W)
    return num/denom


# Flight path azimuth angle rate from air-path azimuth angle rate
# and horizontal airspeed rate (derived from drift angle)
# TODO: The function should be defined piecewise for V_W_xy > 0 and
#       V_W_xy = 0, like this:
#         Piecewise((chi_dot - num/denom, V_W_xy > 0), (chi_dot, True))
#       However, sympy.piecewise is useless with the numpy backend, as it
#       doesn't avoid any ZeroDivisionErrors.
#       Once the following issues are solved, we could add the piecewise
#       implementation:
#         - https://github.com/sympy/sympy/issues/15014
#         - https://github.com/sympy/sympy/issues/14789
@smath_func(symbols, "chi_dot")
def chi_dot_from_chi_A_dot_V_A_xy_dot(
        V_A_xy, V_A_xy_dot, chi, chi_A_dot, V_W_xy, chi_W):
    sqrt_term = sqrt(V_A_xy**2/V_W_xy**2 - sin(chi - chi_W)**2)
    num = chi_A_dot*sqrt_term + V_A_xy_dot/V_A_xy*sin(chi - chi_W)
    denom = sqrt_term + cos(chi - chi_W)
    return num/denom


# Flight path azimuth angle rate from air-path azimuth angle rate
# and horizontal acceleration (derived from drift angle)
# TODO: The function should be defined piecewise for V_W_xy > 0 and
#       V_W_xy = 0, like this:
#         Piecewise((chi_dot - num/denom, V_W_xy > 0), (chi_dot, True))
#       However, sympy.piecewise is useless with the numpy backend, as it
#       doesn't avoid any ZeroDivisionErrors.
#       Once the following issues are solved, we could add the piecewise
#       implementation:
#         - https://github.com/sympy/sympy/issues/15014
#         - https://github.com/sympy/sympy/issues/14789
@smath_func(symbols, "chi_dot")
def chi_dot_from_chi_A_dot_V_K_xy_dot(
        V_K_xy, V_K_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W):
    num = chi_A_dot*cos(chi_A - chi_W) - V_K_xy_dot/V_K_xy*sin(chi_A - chi_W)
    denom = sqrt(V_K_xy**2/V_W_xy**2 - sin(chi_A - chi_W)**2)
    return chi_A_dot - num/denom


@smath_func(symbols, "chi_D_dot")
def chi_D_dot_from_chi_A_dot_V_K_xy_dot(
        V_K_xy, V_K_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W):
    num = chi_A_dot*cos(chi_A - chi_W) - V_K_xy_dot/V_K_xy*sin(chi_A - chi_W)
    denom = sqrt(V_K_xy**2/V_W_xy**2 - sin(chi_A - chi_W)**2)
    return num/denom


@smath_func(symbols, "chi_D_dot")
def chi_D_dot_from_chi_dot_V_A_xy_dot(
        V_A_xy, V_A_xy_dot, chi, chi_dot, V_W_xy, chi_W):
    num = chi_dot*cos(chi - chi_W) - V_A_xy_dot/V_A_xy*sin(chi - chi_W)
    denom = sqrt(V_A_xy**2/V_W_xy**2 - sin(chi - chi_W)**2)
    return num/denom


# Airspeed rate and horizontal/vertical acceleration components


@smath_func(symbols, "V_A_dot")
def V_A_dot_from_V_K_dot(
        V_K, V_K_dot, chi_A, chi_A_dot, gamma_A, gamma_A_dot, V_W, chi_W,
        gamma_W):
    sin_chi_A = sin(chi_A)
    cos_chi_A = cos(chi_A)
    sin_gamma_A = sin(gamma_A)
    cos_gamma_A = cos(gamma_A)
    sin_chi_W = sin(chi_W)
    cos_chi_W = cos(chi_W)
    sin_gamma_W = sin(gamma_W)
    cos_gamma_W = cos(gamma_W)
    u_W_g = V_W*cos_chi_W*cos_gamma_W
    v_W_g = V_W*sin_chi_W*cos_gamma_W
    w_W_g = -V_W*sin_gamma_W
    n_A_V_W = (
        cos_chi_A*cos_gamma_A*u_W_g + sin_chi_A*cos_gamma_A*v_W_g -
        sin_gamma_A*w_W_g)
    n_A_dot_V_W = (
        u_W_g*
        (-chi_A_dot*sin_chi_A*cos_gamma_A - gamma_A_dot*cos_chi_A*sin_gamma_A) +
        v_W_g*
        (chi_A_dot*cos_chi_A*cos_gamma_A - gamma_A_dot*sin_chi_A*sin_gamma_A) +
        w_W_g*(-gamma_A_dot*cos_gamma_A))
    denom = n_A_dot_V_W*n_A_V_W + V_K_dot*V_K
    radicand = n_A_V_W**2 + V_K**2 - V_W**2
    return -n_A_dot_V_W + denom/sqrt(radicand)


@smath_func(symbols, "V_A_dot")
def V_A_dot_from_V_A_xy_dot_w_A_g_dot(V_A_xy, V_A_xy_dot, w_A_g, w_A_g_dot):
    V_A = sqrt(V_A_xy**2 + w_A_g**2)
    return (V_A_xy*V_A_xy_dot + w_A_g*w_A_g_dot)/V_A


@smath_func(symbols, "V_A_xy_dot")
def V_A_xy_dot_from_V_A_dot_w_A_g_dot(V_A, V_A_dot, w_A_g, w_A_g_dot):
    V_A_xy = sqrt(V_A**2 - w_A_g**2)
    return (V_A_dot*V_A - w_A_g_dot*w_A_g)/V_A_xy


@smath_func(symbols, "w_A_g_dot")
def w_A_g_dot_from_V_A_dot_V_A_xy_dot(V_A, V_A_dot, V_A_xy, V_A_xy_dot):
    w_A_g = sqrt(V_A**2 - V_A_xy**2)
    return (V_A_dot*V_A - V_A_xy_dot*V_A_xy)/w_A_g


# Total/horizontal/vertical acceleration


@smath_func(symbols, "V_K_dot")
def V_K_dot_from_V_A_dot(
        V_A, V_A_dot, chi, chi_dot, gamma, gamma_dot, V_W, chi_W, gamma_W):
    sin_chi = sin(chi)
    cos_chi = cos(chi)
    sin_gamma = sin(gamma)
    cos_gamma = cos(gamma)
    sin_chi_W = sin(chi_W)
    cos_chi_W = cos(chi_W)
    sin_gamma_W = sin(gamma_W)
    cos_gamma_W = cos(gamma_W)
    u_W_g = V_W*cos_chi_W*cos_gamma_W
    v_W_g = V_W*sin_chi_W*cos_gamma_W
    w_W_g = -V_W*sin_gamma_W
    n_k_V_W = (
        cos_chi*cos_gamma*u_W_g + sin_chi*cos_gamma*v_W_g - sin_gamma*w_W_g)
    n_k_dot_V_W = (
        u_W_g*(-chi_dot*sin_chi*cos_gamma - gamma_dot*cos_chi*sin_gamma) +
        v_W_g*(chi_dot*cos_chi*cos_gamma - gamma_dot*sin_chi*sin_gamma) + w_W_g*
        (-gamma_dot*cos_gamma))
    denom = n_k_dot_V_W*n_k_V_W + V_A_dot*V_A
    radicand = n_k_V_W**2 + V_A**2 - V_W**2
    return n_k_dot_V_W + denom/sqrt(radicand)


@smath_func(symbols, "V_K_dot")
def V_K_dot_from_V_K_xy_dot_w_K_g_dot(V_K_xy, V_K_xy_dot, w_K_g, w_K_g_dot):
    V_K = sqrt(V_K_xy**2 + w_K_g**2)
    return (V_K_xy*V_K_xy_dot + w_K_g*w_K_g_dot)/V_K


@smath_func(symbols, "V_K_xy_dot")
def V_K_xy_dot_from_V_K_dot_w_K_g_dot(V_K, V_K_dot, w_K_g, w_K_g_dot):
    V_K_xy = sqrt(V_K**2 - w_K_g**2)
    return (V_K_dot*V_K - w_K_g_dot*w_K_g)/V_K_xy


@smath_func(symbols, "w_K_g_dot")
def w_K_g_dot_from_V_K_dot_V_K_xy_dot(V_K, V_K_dot, V_K_xy, V_K_xy_dot):
    w_K_g = sqrt(V_K**2 - V_K_xy**2)
    return (V_K_dot*V_K - V_K_xy_dot*V_K_xy)/w_K_g


# Air-path inclination rate and airspeed rate


@smath_func(symbols, "gamma_A_dot")
def gamma_A_dot_from_V_A_xy_dot_w_A_g_dot(V_A, V_A_xy_dot, w_A_g_dot, gamma_A):
    V_A_xy = V_A*cos(gamma_A)
    w_A_g = -V_A*sin(gamma_A)
    return (w_A_g*V_A_xy_dot - w_A_g_dot*V_A_xy)/V_A**2


@smath_func(symbols, "gamma_A_dot")
def gamma_A_dot_from_V_A_dot_w_A_g_dot(V_A, V_A_dot, w_A_g_dot, gamma_A):
    return (-w_A_g_dot - V_A_dot*sin(gamma_A))/(V_A*cos(gamma_A))


@smath_func(symbols, "gamma_A_dot")
def gamma_A_dot_from_V_K_dot_V_A_dot_gamma_dot(
        V_K, V_K_dot, gamma, gamma_dot, V_A, V_A_dot, V_W, gamma_W):
    num = (
        V_K_dot*sin(gamma) + V_K*gamma_dot*cos(gamma) - V_A_dot/V_A*
        (V_K*sin(gamma) - V_W*sin(gamma_W)))
    denom = sqrt(V_A**2 - (V_K*sin(gamma) - V_W*sin(gamma_W))**2)
    return num/denom


@smath_func(symbols, "V_A_xy_dot")
def V_A_xy_dot_from_w_A_g_dot_gamma_A_dot(V_A, w_A_g_dot, gamma_A, gamma_A_dot):
    V_A_xy = V_A*cos(gamma_A)
    w_A_g = -V_A*sin(gamma_A)
    return (gamma_A_dot*V_A**2 + w_A_g_dot*V_A_xy)/w_A_g


@smath_func(symbols, "w_A_g_dot")
def w_A_g_dot_from_V_A_xy_dot_gamma_A_dot(
        V_A, V_A_xy_dot, gamma_A, gamma_A_dot):
    V_A_xy = V_A*cos(gamma_A)
    w_A_g = -V_A*sin(gamma_A)
    return (w_A_g*V_A_xy_dot - gamma_A_dot*V_A**2)/V_A_xy


@smath_func(symbols, "V_A_xy_dot")
def V_A_xy_dot_from_V_A_dot_gamma_A_dot(V_A, V_A_dot, gamma_A, gamma_A_dot):
    return V_A_dot*cos(gamma_A) - gamma_A_dot*V_A*sin(gamma_A)


# Flight path inclination angle rate and velocity


@smath_func(symbols, "gamma_dot")
def gamma_dot_from_V_K_xy_dot_w_K_g_dot(V_K, V_K_xy_dot, w_K_g_dot, gamma):
    V_K_xy = V_K*cos(gamma)
    w_K_g = -V_K*sin(gamma)
    return (w_K_g*V_K_xy_dot - w_K_g_dot*V_K_xy)/V_K**2


@smath_func(symbols, "gamma_dot")
def gamma_dot_from_V_K_dot_w_K_g_dot(V_K, V_K_dot, w_K_g_dot, gamma):
    return (-w_K_g_dot - V_K_dot*sin(gamma))/(V_K*cos(gamma))


@smath_func(symbols, "gamma_dot")
def gamma_dot_from_V_K_dot_V_K_xy_dot(V_K, V_K_dot, V_K_xy_dot, gamma):
    return (-V_K_xy_dot + V_K_dot*cos(gamma))/(V_K*sin(gamma))


@smath_func(symbols, "V_K_xy_dot")
def V_K_xy_dot_from_w_K_g_dot_gamma_dot(V_K, w_K_g_dot, gamma, gamma_dot):
    V_K_xy = V_K*cos(gamma)
    w_K_g = -V_K*sin(gamma)
    return (gamma_dot*V_K**2 + w_K_g_dot*V_K_xy)/w_K_g


@smath_func(symbols, "w_K_g_dot")
def w_K_g_dot_from_V_K_xy_dot_gamma_dot(V_K, V_K_xy_dot, gamma, gamma_dot):
    V_K_xy = V_K*cos(gamma)
    w_K_g = -V_K*sin(gamma)
    return (w_K_g*V_K_xy_dot - gamma_dot*V_K**2)/V_K_xy


@smath_func(symbols, "V_K_xy_dot")
def V_K_xy_dot_from_V_K_dot_gamma_dot(V_K, V_K_dot, gamma, gamma_dot):
    return V_K_dot*cos(gamma) - gamma_dot*V_K*sin(gamma)


@smath_func(symbols, "w_K_g_dot")
def w_K_g_dot_from_V_K_dot_gamma_dot(V_K, V_K_dot, gamma, gamma_dot):
    return -V_K_dot*sin(gamma) - gamma_dot*V_K*cos(gamma)


# Horizontal speed components


@smath_func(symbols, "V_K_xy_dot")
def V_K_xy_dot_from_V_A_xy_dot_chi_A_dot(
        V_A_xy, V_A_xy_dot, chi_A, chi_A_dot, V_W_xy, chi_W):
    num = (
        V_A_xy*V_A_xy_dot + V_W_xy*
        (V_A_xy_dot*cos(chi_A - chi_W) - chi_A_dot*V_A_xy*sin(chi_A - chi_W)))
    denom = sqrt(V_A_xy**2 + V_W_xy**2 + 2*V_A_xy*V_W_xy*cos(chi_A - chi_W))
    return num/denom


# TODO: The function should be defined piecewise for
#       (V_W_xy > 0 && abs(chi_W-chi_A) > 0) or
#       (V_W_xy = 0 || abs(chi_W-chi_A) = 0), like this:
#         Piecewise((V_A_xy_dot, V_W_xy = 0 || abs(chi_W-chi_A) = 0),
#                   (*as below*, True))
#       However, sympy.piecewise is useless with the numpy backend, as it
#       doesn't avoid any ZeroDivisionErrors.
#       Once the following issues are solved, we could add the piecewise
#       implementation:
#         - https://github.com/sympy/sympy/issues/15014
#         - https://github.com/sympy/sympy/issues/14789
@smath_func(symbols, "V_K_xy_dot")
def V_K_xy_dot_from_chi_dot_chi_A_dot(
        V_K_xy, chi_dot, chi_A, chi_A_dot, V_W_xy, chi_W):
    sqrt_term = sqrt(V_K_xy**2/V_W_xy**2 - sin(chi_W - chi_A)**2)
    num = (
        chi_A_dot*V_K_xy*cos(chi_A - chi_W) -
        (chi_A_dot-chi_dot)*V_K_xy*sqrt_term)
    denom = sin(chi_A - chi_W)
    return num/denom


@smath_func(symbols, "V_A_xy_dot")
def V_A_xy_dot_from_V_K_xy_dot_chi_dot(
        V_K_xy, V_K_xy_dot, chi, chi_dot, V_W_xy, chi_W):
    num = (
        V_K_xy*V_K_xy_dot - V_W_xy*
        (V_K_xy_dot*cos(chi_W - chi) + chi_dot*V_K_xy*sin(chi_W - chi)))
    denom = sqrt(V_K_xy**2 + V_W_xy**2 - 2*V_K_xy*V_W_xy*cos(chi_W - chi))
    return num/denom


# TODO: The function should be defined piecewise for
#       (V_W_xy > 0 && abs(chi_W-chi_A) > 0) or
#       (V_W_xy = 0 || abs(chi_W-chi_A) = 0), like this:
#         Piecewise((V_A_xy_dot, V_W_xy = 0 || abs(chi_W-chi_A) = 0),
#                   (*as below*, True))
#       However, sympy.piecewise is useless with the numpy backend, as it
#       doesn't avoid any ZeroDivisionErrors.
#       Once the following issues are solved, we could add the piecewise
#       implementation:
#         - https://github.com/sympy/sympy/issues/15014
#         - https://github.com/sympy/sympy/issues/14789
@smath_func(symbols, "V_A_xy_dot")
def V_A_xy_dot_from_chi_A_dot_chi_dot(
        V_A_xy, chi_A_dot, chi, chi_dot, V_W_xy, chi_W):
    sqrt_term = sqrt(V_A_xy**2/V_W_xy**2 - sin(chi - chi_W)**2)
    num = (
        chi_dot*V_A_xy*cos(chi - chi_W) - (chi_A_dot-chi_dot)*V_A_xy*sqrt_term)
    denom = sin(chi - chi_W)
    return num/denom

# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0

from eduplane.utils.smath import (
    Max, Min, Piecewise, acos, asin, cos, sin, smath_func, sqrt)

from . import kinematic_model

symbols = {
    # forces
    "D": {
        "name": r"D",
        "dimension": "force",
        "unit": "N",
        "definition": "C_D*rho/2*V_A**2*S_"
    },
    "D_p": {
        "name": r"D_p",
        "dimension": "force",
        "unit": "N",
        "definition": "C_D0*V_A**2*rho*S_/2"
    },
    "D_i": {
        "name": r"D_i",
        "dimension": "force",
        "unit": "N",
        "definition": "k*(rho*V_A**2*S_*C_L0-2*n*m*g)**2/(2*rho*V_A**2*S_)"
    },
    "L": {
        "name": r"L",
        "dimension": "force",
        "unit": "N",
        "definition": "C_L*rho/2*V_A**2*S_"
    },
    "L_z": {
        "name": r"L_z",
        "dimension": "force",
        "unit": "N",
        "definition": "L*cos(mu_A)"
    },
    "L_xy": {
        "name": r"L_xy",
        "dimension": "force",
        "unit": "N",
        "definition": "L*sin(mu_A)"
    },
    "T": {
        "name": r"T",
        "dimension": "force",
        "unit": "N"
    },
    "T_s": {
        "name": r"T_s",
        "dimension": "force",
        "unit": "N"
    },
    "P": {
        "name": r"P",
        "dimension": "power",
        "unit": "W"
    },
    "W": {
        "name": r"W",
        "dimension": "force",
        "unit": "N",
        "definition": "m*g",
        "assumptions": {
            "positivie": True,
            "constant": True,
        }
    },
    # coefficients
    "C_D": {
        "name": r"C_D",
        "dimension": "dimensionless",
        "unit": None,
        "definition": "C_D0 + k*(C_L - C_L0)**2"
    },
    "C_D0": {
        "name": r"C_{D0}",
        "dimension": "dimensionless",
        "unit": None,
        "assumptions": {
            "constant": True
        },
    },
    "C_L": {
        "name": r"C_L",
        "dimension": "dimensionless",
        "unit": None,
        "definition": "L/(rho/2*V_A**2*S_)"
    },
    "C_L0": {
        "name": r"C_{L0}",
        "dimension": "dimensionless",
        "unit": None,
        "assumptions": {
            "constant": True
        },
    },
    # constants / others
    "n": {
        "name": r"n",
        "dimension": "dimensionless",
        "unit": None,
        "definition": "L/W"
    },
    "n_s": {
        "name": r"n",
        "dimension": "dimensionless",
        "unit": None,
        "definition": "L/W"
    },
    "rho": {
        "name": r"\rho",
        "dimension": "density",
        "unit": "kg/m**3",
        "assumptions": {
            "positive": True,
            "constant": True
        }
    },
    # NOTE: 'S' is sympy internal name
    "S_": {
        "name": r"{S}",
        "dimension": "area",
        "unit": "m**2",
        "assumptions": {
            "positive": True,
            "constant": True
        }
    },
    "k": {
        "name": r"k",
        "dimension": "dimensionless",
        "unit": None,
        "assumptions": {
            "positive": True,
            "constant": True
        }
    },
    **kinematic_model.symbols
}

# Equilibrium of Forces


@smath_func(symbols, "T")
def forces_x_a(D, W, V_A_dot, gamma_A, m):
    return D + W*sin(gamma_A) + V_A_dot*m


@smath_func(symbols, "L_xy")
def forces_y_a1(V_A, gamma_A, chi_A_dot, m):
    return m*chi_A_dot*V_A*cos(gamma_A)


@smath_func(symbols, "L_z")
def forces_z_a2(W, V_A, gamma_A, gamma_A_dot, m):
    return cos(gamma_A)*W + m*gamma_A_dot*V_A


# Load factor from vertical forces


@smath_func(symbols, "n")
def n_from_forces_z_a2(V_A, gamma_A, gamma_A_dot, mu_A, g):
    return cos(gamma_A)/cos(mu_A) + gamma_A_dot*V_A/(g*cos(mu_A))


# Load factor from lateral forces


@smath_func(symbols, "n")
def n_from_forces_y_a1(V_A, chi_A_dot, gamma_A, mu_A, g):
    return chi_A_dot*V_A*cos(gamma_A)/(g*sin(mu_A))


# Load factor from total lift equation
@smath_func(symbols, "n")
def n_from_chi_A_dot_gamma_A_dot(V_A, chi_A_dot, gamma_A, gamma_A_dot, g):
    gamma_A_dot_n_pos = -cos(gamma_A)*g/V_A
    radicand = (g*cos(gamma_A))**2 + 2*g*gamma_A_dot*V_A*cos(gamma_A) + (
        gamma_A_dot*V_A)**2 + (chi_A_dot*V_A*cos(gamma_A))**2
    return Piecewise(
        (sqrt(radicand)/g, gamma_A_dot >= gamma_A_dot_n_pos),
        (-sqrt(radicand)/g, gamma_A_dot < gamma_A_dot_n_pos))


# Kinematics from total lift / load factor equations


@smath_func(symbols, "chi_A_dot_abs")
def chi_A_dot_abs_from_n_gamma_A_dot(n, V_A, gamma_A, gamma_A_dot, g):
    radicand_f1 = (g*n + g*cos(gamma_A) + V_A*gamma_A_dot)
    radicand_f2 = (g*n - g*cos(gamma_A) - V_A*gamma_A_dot)
    return sqrt(radicand_f1*radicand_f2)/(V_A*cos(gamma_A))


@smath_func(symbols, "gamma_A_dot")
def gamma_A_dot_from_n_chi_A_dot(n, V_A, chi_A_dot, gamma_A, g):
    radicand = (g*n)**2 - (chi_A_dot*V_A*cos(gamma_A))**2
    return Piecewise(
        ((sqrt(radicand) - g*cos(gamma_A))/V_A, n >= 0),
        ((-sqrt(radicand) - g*cos(gamma_A))/V_A, n < 0))


@smath_func(symbols, "mu_A")
def mu_A_from_n_chi_A_dot(V_A, gamma_A, chi_A_dot, n, g):
    return asin(chi_A_dot*V_A*cos(gamma_A)/(g*n))


@smath_func(symbols, "mu_A")
def mu_A_from_n_gamma_A_dot(V_A, gamma_A, gamma_A_dot, n, g):
    return acos((cos(gamma_A) + gamma_A_dot*V_A/g)/n)


# Lift coefficient


@smath_func(symbols, "C_L")
def C_L_from_chi_A_dot_gamma_A_dot(
        V_A, chi_A_dot, gamma_A, gamma_A_dot, m, g, S_, rho):
    factor = 2*m/(rho*V_A**2*S_)
    radicand = (g*cos(gamma_A))**2          \
        + (chi_A_dot*V_A*cos(gamma_A))**2   \
        + 2*g*gamma_A_dot*V_A*cos(gamma_A)  \
        + (gamma_A_dot*V_A)**2
    return factor*sqrt(radicand)


@smath_func(symbols, "C_L")
def C_L_from_n(V_A, n, m, g, S_, rho):
    return 2*n*m*g/(rho*V_A**2*S_)


@smath_func(symbols, "n")
def n_from_C_L(V_A, C_L, m, g, S_, rho):
    return C_L*(rho*V_A**2*S_)/(2*m*g)


@smath_func(symbols, "chi_A_dot_abs")
def chi_A_dot_abs_from_C_L_gamma_A_dot(
        C_L, V_A, gamma_A, gamma_A_dot, m, g, S_, rho):
    radicand = (rho*V_A*S_*C_L/(2*m*cos(gamma_A)))**2   \
        - (g/V_A)**2                                    \
        - 2*g*gamma_A_dot/(V_A*cos(gamma_A))            \
        - (gamma_A_dot/cos(gamma_A))**2
    return sqrt(radicand)


@smath_func(symbols, "gamma_A_dot")
def gamma_A_dot_from_C_L_chi_A_dot(C_L, V_A, chi_A_dot, gamma_A, m, g, S_, rho):
    radicand = (rho*V_A*S_*C_L/(2*m))**2 - (chi_A_dot*cos(gamma_A))**2
    return sqrt(radicand) - g*cos(gamma_A)/V_A


@smath_func(symbols, "V_A")
def V_A_from_C_L_n(C_L, n, m, g, S_, rho):
    return sqrt(2*g*m*n/(C_L*rho*S_))


# NOTE: Only valid for quasi-steady flight state
@smath_func(symbols, "V_A")
def V_A_from_C_L_gamma_A(C_L, gamma_A, chi_A_dot, m, g, S_, rho):
    po2_term = (sqrt(2)*m*chi_A_dot/(C_L*rho*S_))**2
    q_term = (2*m*g*cos(gamma_A)/(C_L*rho*S_))**2
    return sqrt(po2_term + sqrt(po2_term**2 + q_term))


# Drag, thrust and load factor


@smath_func(symbols, "D")
def D_from_n(n, V_A, C_D0, C_L0, m, g, k, S_, rho):
    parasite_drag = C_D0*V_A**2*rho*S_/2
    induced_drag = k/(2*V_A**2*rho*S_)*(C_L0*V_A**2*rho*S_ - 2*g*m*n)**2
    return parasite_drag + induced_drag


@smath_func(symbols, "T")
def T_from_n_gamma_A(n, V_A, V_A_dot, gamma_A, C_D0, C_L0, m, g, k, S_, rho):
    inertial_force = V_A_dot*m
    weight_force_comp = m*g*sin(gamma_A)
    parasite_drag = C_D0*V_A**2*rho*S_/2
    induced_drag = k/(2*V_A**2*rho*S_)*(C_L0*V_A**2*rho*S_ - 2*g*m*n)**2
    return inertial_force + weight_force_comp + parasite_drag + induced_drag


# NOTE: Only valid for quasi-steady flight state
@smath_func(symbols, "T_s")
def T_s_from_gamma_A(V_A, gamma_A, chi_A_dot, C_D0, C_L0, m, g, k, S_, rho):
    n = cos(gamma_A)*sqrt(1 + (V_A*chi_A_dot/g)**2)
    weight_force_comp = m*g*sin(gamma_A)
    parasite_drag = C_D0*V_A**2*rho*S_/2
    induced_drag = k/(2*V_A**2*rho*S_)*(C_L0*V_A**2*rho*S_ - 2*g*m*n)**2
    return weight_force_comp + parasite_drag + induced_drag


# NOTE: This will return the pull-up load factor (always positive if C_L0=0)
@smath_func(symbols, "n")
def n_pull_from_T_gamma_A(
        T, V_A, V_A_dot, gamma_A, C_D0, C_L0, m, g, k, S_, rho):
    radicand = 2*T - 2*m*g*sin(gamma_A) - 2*V_A_dot*m - rho*V_A**2*S_*C_D0
    denom = 2*m*g*sqrt(k)
    return V_A/denom*(rho*V_A*S_*sqrt(k)*C_L0 + sqrt(rho*S_)*sqrt(radicand))


# NOTE: This will return the push-down load factor (always negative if C_L0=0)
@smath_func(symbols, "n")
def n_push_from_T_gamma_A(
        T, V_A, V_A_dot, gamma_A, C_D0, C_L0, m, g, k, S_, rho):
    radicand = 2*T - 2*m*g*sin(gamma_A) - 2*V_A_dot*m - rho*V_A**2*S_*C_D0
    denom = 2*m*g*sqrt(k)
    return V_A/denom*(rho*V_A*S_*sqrt(k)*C_L0 - sqrt(rho*S_)*sqrt(radicand))


# Calculate the quasi-steady state load factor (always positive)
# NOTE: Only valid for quasi-steady flight state
@smath_func(symbols, "n_s")
def n_s_from_V_A_gamma_A(V_A, gamma_A, chi_A_dot, g):
    return cos(gamma_A)*sqrt(1 + (V_A*chi_A_dot/g)**2)


# Calculate steady-state air-path inlination angle based on load factor
# NOTE: Only valid for quasi-steady flight state
@smath_func(symbols, "gamma_A")
def gamma_A_from_n_s(n_s, V_A, chi_A_dot, g):
    return acos(n_s/sqrt(1 + (V_A*chi_A_dot/g)**2))


# Calculate steady-state vertical speed based on load factor
# NOTE: Only valid for quasi-steady flight state
@smath_func(symbols, "w_A_g")
def w_A_g_from_n_s(n_s, V_A, chi_A_dot, g):
    gamma_A = acos(n_s/sqrt(1 + (V_A*chi_A_dot/g)**2))
    return -sin(gamma_A)*V_A


@smath_func(symbols, "T")
def T_from_n_w_A_g(n, V_A, V_A_dot, w_A_g, C_D0, C_L0, m, g, k, S_, rho):
    inertial_force = V_A_dot*m
    weight_force_comp = -m*g*w_A_g/V_A
    parasite_drag = C_D0*V_A**2*rho*S_/2
    induced_drag = k/(2*V_A**2*rho*S_)*(C_L0*V_A**2*rho*S_ - 2*g*m*n)**2
    return inertial_force + weight_force_comp + parasite_drag + induced_drag


@smath_func(symbols, "D_i")
def D_i_from_n(n, V_A, C_L0, m, g, k, S_, rho):
    induced_drag = k/(2*V_A**2*rho*S_)*(C_L0*V_A**2*rho*S_ - 2*g*m*n)**2
    return induced_drag


# Calculate quasi-steady-state thrust
# NOTE: Only valid for quasi-steady flight state
@smath_func(symbols, "T_s")
def T_s_from_w_A_g(V_A, w_A_g, chi_A_dot, C_D0, C_L0, m, g, k, S_, rho):
    gamma_A = asin(-w_A_g/V_A)
    n = cos(gamma_A)*sqrt(1 + (V_A*chi_A_dot/g)**2)
    weight_force_comp = -m*g*w_A_g/V_A
    parasite_drag = C_D0*V_A**2*rho*S_/2
    induced_drag = k/(2*V_A**2*rho*S_)*(C_L0*V_A**2*rho*S_ - 2*g*m*n)**2
    return weight_force_comp + parasite_drag + induced_drag


# NOTE: This will return the pull-up load factor (always positive if C_L0=0)
@smath_func(symbols, "n")
def n_pull_from_T_w_A_g(T, V_A, V_A_dot, w_A_g, C_D0, C_L0, m, g, k, S_, rho):
    radicand = 2*T + 2*m*g*w_A_g/V_A - 2*V_A_dot*m - rho*V_A**2*S_*C_D0
    denom = 2*m*g*sqrt(k)
    return V_A/denom*(rho*V_A*S_*sqrt(k)*C_L0 + sqrt(rho*S_)*sqrt(radicand))


# NOTE: This will return the push-down load factor (always negative if C_L0=0)
@smath_func(symbols, "n")
def n_push_from_T_w_A_g(T, V_A, V_A_dot, w_A_g, C_D0, C_L0, m, g, k, S_, rho):
    radicand = 2*T + 2*m*g*w_A_g/V_A - 2*V_A_dot*m - rho*V_A**2*S_*C_D0
    denom = 2*m*g*sqrt(k)
    return V_A/denom*(rho*V_A*S_*sqrt(k)*C_L0 - sqrt(rho*S_)*sqrt(radicand))


# Calculate the quasi-steady state load factor (always positive)
# NOTE: Only valid for quasi-steady flight state
@smath_func(symbols, "n_s")
def n_s_from_V_A_w_A_g(V_A, w_A_g, chi_A_dot, g):
    gamma_A = asin(-w_A_g/V_A)
    return cos(gamma_A)*sqrt(1 + (V_A*chi_A_dot/g)**2)


# NOTE: For future reference, here is an implementation using sympy.Piecewise
# which, unfortunately, evaluates all branches before selecting the solution,
# leading to domain errors.
# return Piecewise(
#     (0.5*pi, (T/(g*m) - counter_forces_term >= 1)), (
#         asin(T/(g*m) - counter_forces_term),
#         And(
#             T/(g*m) - counter_forces_term > -1, T/
#             (g*m) - counter_forces_term < 1)),
#     (-0.5*pi, (T/(g*m) - counter_forces_term <= -1)))
@smath_func(symbols, "gamma_A")
def gamma_A_from_T(T, n, V_A, V_A_dot, C_D0, C_L0, m, g, k, S_, rho):
    parasite_drag_term = C_D0*V_A**2*rho*S_/(2*g*m)
    induced_drag_term = k*(C_L0*V_A**2*rho*S_ -
                           2*g*m*n)**2/(2*V_A**2*rho*g*m*S_)
    counter_forces_term = V_A_dot/g + parasite_drag_term + induced_drag_term
    return asin(Min(Max(T/(g*m) - counter_forces_term, -1), 1))


# NOTE: The equation w_A_g=f(T) does not take into account kinematics, i.e.
# it will return vertical speeds exceeding the given airspeed.
@smath_func(symbols, "w_A_g")
def w_A_g_from_T(T, n, V_A, V_A_dot, C_D0, C_L0, m, g, k, S_, rho):
    parasite_drag_term = C_D0*V_A**3*rho*S_/(2*g*m)
    induced_drag_term = k*(C_L0*V_A**2*rho*S_ - 2*g*m*n)**2/(2*V_A*rho*g*m*S_)
    return V_A_dot*V_A/g + parasite_drag_term + induced_drag_term - T*V_A/(g*m)
    # return Min(V_A, V_A_dot*V_A/g + parasite_drag_term + induced_drag_term - T*V_A/(g*m))


@smath_func(symbols, "V_A")
def V_A_from_T_gamma_A(T, n, gamma_A, V_A_dot, C_D0, C_L0, m, g, k, S_, rho):
    neg_po2 = (T - V_A_dot*m + g*m*
               (2*k*n*C_L0 - sin(gamma_A)))/(rho*S_*(C_D0 + k*C_L0**2))
    q = 4*k*(m*g*n)**2/((rho*S_)**2*(C_D0 + k*C_L0**2))
    return sqrt(neg_po2 + sqrt(neg_po2**2 - q))


@smath_func(symbols, "V_A")
def V_A_min_from_T_gamma_A(
        T, n, gamma_A, V_A_dot, C_D0, C_L0, m, g, k, S_, rho):
    # po2 = (m*g*sin(gamma_A) - T -
    #        2*k*g*m*C_L0*n)/(rho*S_*C_D0 + k*rho*S_*C_L0**2)
    neg_po2 = (T - V_A_dot*m + g*m*
               (2*k*n*C_L0 - sin(gamma_A)))/(rho*S_*(C_D0 + k*C_L0**2))
    q = 4*k*(m*g*n)**2/((rho*S_)**2*(C_D0 + k*C_L0**2))
    return sqrt(neg_po2 - sqrt(neg_po2**2 - q))


@smath_func(symbols, "V_A_dot")
def V_A_dot_from_T(T, n, V_A, gamma_A, C_D0, C_L0, m, g, k, S_, rho):
    parasite_drag_term = C_D0*V_A**2*rho*S_/(2*m)
    induced_drag_term = k*(C_L0*V_A**2*rho*S_ - 2*g*m*n)**2/(2*V_A**2*rho*m*S_)
    return T/m - g*sin(gamma_A) - parasite_drag_term - induced_drag_term


@smath_func(symbols, "P")
def P_from_T_V_A(T, V_A):
    return T*V_A


@smath_func(symbols, "T")
def T_from_P_V_A(P, V_A):
    return P/V_A

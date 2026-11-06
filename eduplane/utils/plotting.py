# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0

import math
import textwrap
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy
import pandas
from matplotlib import gridspec, ticker
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from eduplane.utils import utils

# import matplotlib
# matplotlib.use("TkAgg")

# Maximum figure sizes (text width and height) for different page formats
figure_sizes_print = {
    "A4_full": [5.3929055118, 8.1838425197],
    "A5_full": [3.6845, 5.7865551181],
    "A4_large": [5.3929055118, 8.1838425197*0.618],
    "A5_large": [3.6845, 5.7865551181*0.618],
    "A4_small": [5.3929055118, 8.1838425197*0.382],
    "A5_small": [3.6845, 5.7865551181*0.382],
    "IEEE_single_column": [3.5, 3.5]
}

figure_sizes_screen = {
    "HD": [1920, 1080],
    "SD": [720, 576],
}

# English (mostly) text-based labels
labels_en_text = {
    "3D view": r"3D View",
    "top view": r"Top View",
    "vert. profile": r"Vertical Profile",
    "3d-dubins-path": r"3D Dubins Path",
    "timeline": r"Timeline",
    "timelines": r"Timelines",
    "acceleration": r"acceleration (m/s²)",
    "speed": r"speed (m/s)",
    "airspeed": r"airspeed (m/s)",
    "time": r"time (s)",
    "x": r'$x$ (m)',
    "y": r'$y$ (m)',
    "z": r'$z$ (m)',
    "h": r'height (m)',
    "x_g": r'$x_g$ (m)',
    "y_g": r'$y_g$ (m)',
    "z_g": r'$z_g$ (m)',
    "z_g_tf": r'final alt. (m)',
    "h_g": r'height (m)',
    "l_xy": r'hor. dist. (m)',
    "l_xy": r'hor. dist. (m)',
    "psi_dot": r"turnrate (°/s)",
    "gamma": r"flight path angle (-)",
    "gamma_deg": r"flight path angle (°)",
    "gamma_dot": r"flight path angle rate (1/s)",
    "gamma_dot_deg": r"flight path angle rate (°/s)",
    "gamma_A": r"air-path inclination angle (-)",
    "gamma_A_deg": r"air-path inclination angle (°)",
    "gamma_A_dot": r"air-path inclination angle rate (1/s)",
    "gamma_A_dot_deg": r"air-path inclination angle rate (°/s)",
    "V_A": r"airspeed (m/s)",
    "V_A_dot": r"airspeed rate (m/s²)",
    "V_A_xy": r"hor. airspeed (m/s)",
    "V_A_xy_dot": r"hor. airspeed rate (m/s²)",
    "V_K": r"velocity (m/s)",
    "V_K_dot": r"acceleration (m/s²)",
    "V_K_xy": r"hor. velocity (m/s)",
    "V_K_xy_dot": r"hor. acceleration (m/s²)",
    "t": r"time (s)",
    "w": r"vert. speed (m/s)",
    "w_dot": r"vert. acc. (m/s²)",
    "w_A_g": r"vert. airspeed (m/s)",
    "w_A_g_dot": r"vert. air-rel. acc. (m/s²)",
    "d_E": r"path deviation (m)",
    "d_E_xy": r"hor. path deviation (m)",
    "d_E_z": r"vert. path deviation (m)",
    "T": r"thrust (N)",
    "T_bracket": r"thrust bracket width (N)",
    "T_current": r"thrust (N)",
    "P": r"power (W)",
    "E": r"enery (J)",
    "n": r"load factor",
    "C_L": r"lift coeff.",
    "residual": r"residual {}",
}

# German (mostly) text-based labels
labels_de_text = {
    "3D view": r"3D Ansicht",
    "top view": r"Draufsicht",
    "vert. profile": r"Vertikalprofil",
    "3d-dubins-path": r"3D Dubins Pfad",
    "timeline": r"Zeitlicher Verlauf",
    "timelines": r"Zeitliche Verläufe",
    "acceleration": r"Beschl. (m/s²)",
    "speed": r"Geschw. (m/s)",
    "airspeed": r"Fluggeschw. (m/s)",
    "time": r"Zeit (s)",
    "x": r'$x$ (m)',
    "y": r'$y$ (m)',
    "z": r'$z$ (m)',
    "h": r'Höhe (m)',
    "x_g": r'$x_g$ (m)',
    "y_g": r'$y_g$ (m)',
    "z_g": r'$z_g$ (m)',
    "z_g_tf": r'Endh. (m)',
    "l_xy": r'horiz. Dist. (m)',
    "psi_dot": r"Wenderate (°/s)",
    "gamma": r"Bahnwinkel (-)",
    "gamma_deg": r"Bahnwinkel (°)",
    "gamma_dot": r"Bahnwinkelrate (1/s)",
    "gamma_deg_dot": r"Bahnwinkelrate (°/s)",
    "gamma_A": r"Flugwindneigungswinkel (-)",
    "gamma_A_deg": r"Flugwindneigungswinkel (°)",
    "gamma_A_dot": r"Flugwindneigungsrate (-)",
    "gamma_A_dot_deg": r"Flugwindneigungsrate (°)",
    "V_A": r"Fluggeschw. (m/s)",
    "V_A_dot": r"Fluggeschw.-rate (m/s²)",
    "V_A_xy": r"horiz. Fluggeschw. (m/s)",
    "V_A_xy_dot": r"horiz. Fluggeschw.-rate (m/s²)",
    "V_K": r"Geschw. (m/s)",
    "V_K_dot": r"Geschw. (m/s²)",
    "V_K_xy": r"horiz. Geschw (m/s)",
    "t": r"Zeit (s)",
    "w": r"vert. Geschw. (m/s)",
    "w_dot": r"vert. Beschl. (m/s²)",
    "w_A_g": r"vert. Fluggeschw. (m/s)",
    "w_A_g_dot": r"vert. Fluggeschw.-rate (m/s²)",
    "d_E": r"Bahnablage (m)",
    "d_E_xy": r"horiz. Bahnablage (m)",
    "d_E_z": r"vert. Bahnablage (m)",
    "T": r"Schub (N)",
    "T_bracket": r"Schub-Intervallbreite (N)",
    "T_current": r"Schub (N)",
    "P": r"Leistung (W)",
    "E": r"Energie (J)",
    "n": r"Lastvielfaches",
    "C_L": r"Auftriebskoeff.",
    "residual": r"Residuum - {}",
}

# Labels using Latex
labels_en_mathtext = {
    "3D view": r"3D View",
    "top view": r"Top View",
    "vert. profile": r"Vertical Profile",
    "3d-dubins-path": r"3D Dubins Path",
    "timeline": r"Timeline",
    "timelines": r"Timelines",
    "acceleration": r"$\dot{V}\;(\mathrm{m/s^2})$",
    "speed": r"$V\;(\mathrm{m/s})$",
    "airspeed": r"$V_{A}\;(\mathrm{m/s})$",
    "time": r"$t\;(\mathrm{s}$",
    "x": r'$x\;(\mathrm{m})$',
    "y": r'$y\;(\mathrm{m})$',
    "z": r'$z\;(\mathrm{m})$',
    "h": r'$h\;(\mathrm{m})$',
    "x_g": r'$x_g\;(\mathrm{m})$',
    "y_g": r'$y_g\;(\mathrm{m})$',
    "z_g": r'$z_g\;(\mathrm{m})$',
    "z_g_tf": r'$z_{g,tf}\;(\mathrm{m})$',
    "h_g": r'$h_g\;(\mathrm{m})$',
    "l_xy": r'$l_{xy}\;(\mathrm{m})$',
    "psi_dot": r"$\dot{\varPsi}\;(\mathrm{^\circ/s})$",
    "gamma": r"$\gamma\;(-)$",
    "gamma_dot": r"$\dot{\gamma}\;(\mathrm{1/s})$",
    "gamma_deg": r"$\gamma\;(\mathrm{^\circ})$",
    "gamma_deg_dot": r"$\dot{\gamma}\;(\mathrm{^\circ/s})$",
    "gamma_A": r"$\gamma_A\;(-)$",
    "gamma_A_deg": r"$\gamma_A\;(\mathrm{^\circ})$",
    "gamma_A_dot": r"$\dot{\gamma}_A\;(\mathrm{1/s})$",
    "gamma_A_dot_deg": r"$\dot{\gamma}_A\;(\mathrm{^\circ/s})$",
    "chi": r"$\chi\;(-)$",
    "chi_deg": r"$\chi\;(\mathrm{^\circ})$",
    "chi_dot": r"$\dot{\chi}\;(\mathrm{1/s})$",
    "chi_dot_deg": r"$\dot{\chi}\;(\mathrm{^\circ/s})$",
    "chi_A": r"$\chi_A\;(-)$",
    "chi_A_deg": r"$\chi_A\;(\mathrm{^\circ})$",
    "chi_A_dot": r"$\dot{\chi}_A\;(\mathrm{1/s})$",
    "chi_A_dot_deg": r"$\dot{\chi}_A\;(\mathrm{^\circ/s})$",
    "V_A": r"$V_{A}\;(\mathrm{m/s})$",
    "V_A_dot": r"$\dot{V}_{A}\;(\mathrm{m/s^2})$",
    "V_A_xy": r"$V_{A,xy}\;(\mathrm{m/s})$",
    "V_A_xy_dot": r"$\dot{V}_{A,xy}\;(\mathrm{m/s^2})$",
    "V_K": r"$V_{K}\;(\mathrm{m/s})$",
    "V_K_dot": r"$\dot{V}_{K}\;(\mathrm{m/s^2})$",
    "V_K_xy": r"$V_{K,xy}\;(\mathrm{m/s})$",
    "V_K_xy_dot": r"$\dot{V}_{K,xy}\;(\mathrm{m/s})$",
    "t": r"$t\;(s)$",
    "w": r"$w\;(\mathrm{m/s})$",
    "w_dot": r"$\dot{w}\;(\mathrm{m/s^2})$",
    "w_A_g": r"$w_{Ag}\;(\mathrm{m/s})$",
    "w_A_g_dot": r"$\dot{w}_{Ag}\;(\mathrm{m/s^2})$",
    "d_E": r"$d_E\;(\mathrm{m})$",
    "d_E_xy": r"$d_{E,xy}\;(\mathrm{m})$",
    "d_E_z": r"$d_{E,z}\;(\mathrm{m})$",
    "T": r"$T\;(\mathrm{N})$",
    "T_bracket": r"$\Delta T\;(\mathrm{N})$",
    "T_current": r"$T\;(\mathrm{N})$",
    "P": r"$P\;(\mathrm{N})$",
    "E": r"$E\;(\mathrm{J})$",
    "n": r"$n$",
    "C_L": r"$C_L$",
    "residual": r"$\Delta\,{}$",
}

labels_de_mathtext = labels_en_mathtext.copy()
labels_de_mathtext["3D view"] = labels_de_text["3D view"]
labels_de_mathtext["top view"] = labels_de_text["top view"]
labels_de_mathtext["vert. profile"] = labels_de_text["vert. profile"]
labels_de_mathtext["3d-dubins-path"] = labels_de_text["3d-dubins-path"]
labels_de_mathtext["timeline"] = labels_de_text["timeline"]
labels_de_mathtext["timelines"] = labels_de_text["timelines"]

# Labels using Latex
labels_en_latex = {
    "3D view": r"3D View",
    "top view": r"Top View",
    "vert. profile": r"Vertical Profile",
    "3d-dubins-path": r"3D Dubins Path",
    "timeline": r"Timeline",
    "timelines": r"Timelines",
    "acceleration": r"$\dot{V}\;(\unit{\metre\per\second\squared})$",
    "speed": r"$V\;(\unit{\metre\per\second})$",
    "airspeed": r"$V_{A}\;(\unit{\metre\per\second})$",
    "time": r"$t\;(\unit{\second})$",
    "x": r'$x\;(\unit{\metre})$',
    "y": r'$y\;(\unit{\metre})$',
    "z": r'$z\;(\unit{\metre})$',
    "h": r'$h\;(\unit{\metre})$',
    "x_g": r'$x_g\;(\unit{\metre})$',
    "y_g": r'$y_g\;(\unit{\metre})$',
    "z_g": r'$z_g\;(\unit{\metre})$',
    "z_g_tf": r'$z_{g,tf}\;(\unit{\metre})$',
    "h_g": r'$h_g\;(\unit{\metre})$',
    "l_xy": r'$l_{xy}\;(\unit{\metre})$',
    "psi_dot": r"$\dot{\varPsi}\;(\unit{\degree\per\second})$",
    "gamma": r"$\gamma\;(-)$",
    "gamma_dot": r"$\dot{\gamma}\;(\unit{\per\second})$",
    "gamma_deg": r"$\gamma\;(\unit{\degree})$",
    "gamma_deg_dot": r"$\dot{\gamma}\;(\unit{\degree\per\second})$",
    "gamma_A": r"$\gamma_A\;(-)$",
    "gamma_A_deg": r"$\gamma_A\;(\unit{\degree})$",
    "gamma_A_dot": r"$\dot{\gamma}_A\;(\unit{\per\second})$",
    "gamma_A_dot_deg": r"$\dot{\gamma}_A\;(\unit{\degree\per\second})$",
    "chi": r"$\chi\;(-)$",
    "chi_deg": r"$\chi\;(\mathrm{^\circ})$",
    "chi_dot": r"$\dot{\chi}\;(\mathrm{1/s})$",
    "chi_dot_deg": r"$\dot{\chi}\;(\mathrm{^\circ/s})$",
    "chi_A": r"$\chi_A\;(-)$",
    "chi_A_deg": r"$\chi_A\;(\mathrm{^\circ})$",
    "chi_A_dot": r"$\dot{\chi}_A\;(\mathrm{1/s})$",
    "chi_A_dot_deg": r"$\dot{\chi}_A\;(\mathrm{^\circ/s})$",
    "V_A": r"$V_{A}\;(\unit{\metre\per\second})$",
    "V_A_dot": r"$\dot{V}_{A}\;(\unit{\metre\per\second\squared})$",
    "V_A_xy": r"$V_{A,xy}\;(\unit{\metre\per\second})$",
    "V_A_xy_dot": r"$\dot{V}_{A,xy}\;(\unit{\metre\per\second\squared})$",
    "V_K": r"$V_{K}\;(\unit{\metre\per\second})$",
    "V_K_dot": r"$\dot{V}_{K}\;(\unit{\metre\per\second\squared})$",
    "V_K_xy": r"$V_{K,xy}\;(\unit{\metre\per\second})$",
    "V_K_xy_dot": r"$\dot{V}_{K,xy}\;(\unit{\metre\per\second})$",
    "t": r"$t\;(s)$",
    "w": r"$w\;(\unit{\metre\per\second})$",
    "w_dot": r"$\dot{w}\;(\unit{\metre\per\second\squared})$",
    "w_A_g": r"$w_{Ag}\;(\unit{\metre\per\second})$",
    "w_A_g_dot": r"$\dot{w}_{Ag}\;(\unit{\metre\per\second\squared})$",
    "d_E": r"$d_E\;(\unit{\metre})$",
    "d_E_xy": r"$d_{E,xy}\;(\unit{\metre})$",
    "d_E_z": r"$d_{E,z}\;(\unit{\metre})$",
    "T": r"$T\;(\unit{\newton})$",
    "T_bracket": r"$\Delta T\;(\unit{\newton})$",
    "T_current": r"$T\;(\unit{\newton})$",
    "P": r"$P\;(\unit{\watt})$",
    "E": r"$E\;(\unit{\joule})$",
    "n": r"$n$",
    "C_L": r"$C_L$",
    "residual": r"$\Delta\,{}$",
}

labels_de_latex = labels_en_latex.copy()
labels_de_latex["3D view"] = labels_de_text["3D view"]
labels_de_latex["top view"] = labels_de_text["top view"]
labels_de_latex["vert. profile"] = labels_de_text["vert. profile"]
labels_de_latex["3d-dubins-path"] = labels_de_text["3d-dubins-path"]
labels_de_latex["timeline"] = labels_de_text["timeline"]
labels_de_latex["timelines"] = labels_de_text["timelines"]

# Default configuration
default_config = {
    "labels": labels_en_text,
    # "labels": labels_en_latex,
    "figure_size": figure_sizes_print["A4_full"],
    "style": None,
    "show": True,
    "filename": None,
}

# Active configuration
config = default_config.copy()


def show(**kwargs):
    plt.show(**kwargs)


def set_style(style=None):
    """Set a style for matplotlib plots.

    If style is None or "dark", the style will be configured using the
    qbstyles package, if installed, or the matplotlib's default or
    "dark_background" style respectively.

    If style is a dictionary, it will be directly set as the new style.
    """
    if style is None:
        try:
            from qbstyles import mpl_style
            mpl_style(dark=False)
        except ImportError:
            print(
                "Could not import qbstyles. "
                "Falling back to matplotlib's 'default' style")
        plt.style.use("default")
    if style == "dark":
        try:
            from qbstyles import mpl_style
            mpl_style(dark=True)
        except ImportError:
            print(
                "Could not import qbstyles. "
                "Falling back to matplotlib's 'dark_background' style")
            color_cycle = mpl.rcParams['axes.prop_cycle']
            plt.style.use(['dark_background'])
            mpl.rcParams['axes.prop_cycle'] = color_cycle
    else:
        plt.style.use(style)


# Configure latex plotting mode (for publication ready plots)
def enable_latex_mode(
        pageformat="A4_full",
        language="en",
        fontsize=8,
        screen_dpi=150,
        dpi=300,
        **kwargs):
    mpl_config_latex = {
        "font.family": "serif",
        "text.usetex": True,
        "text.latex.preamble": r"""
        \usepackage{amssymb}
        \usepackage{amsmath}
        \usepackage{gensymb}
        \usepackage{siunitx}
        \sisetup{per-mode=fraction}
        """,
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "lines.linewidth": 1.0,
        "legend.fontsize": fontsize,
        "axes.formatter.useoffset": False
    }
    plt.rcParams.update(mpl_config_latex)
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['figure.dpi'] = screen_dpi
    if pageformat not in figure_sizes_print:
        raise ValueError(
            f"Unknown page format '{pageformat}'. Available formats: "
            f"{sorted(figure_sizes_print)}")
    config["figure_size"] = figure_sizes_print[pageformat]
    if language == "en":
        config["labels"] = labels_en_latex
    elif language == "de":
        config["labels"] = labels_de_latex


# Configure screen (i.e. jupyter notebook) plotting mode
def enable_screen_mode(
        text_labels=False,
        language="en",
        fontsize=6,
        dpi=150,
        screen_size="SD",
        **kwargs):
    mpl_config_screen = {
        'text.usetex': False,
        'font.family': "sans-serif",
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'lines.linewidth': 1.0,
        'legend.fontsize': fontsize,
        "axes.formatter.useoffset": False
    }
    plt.rcParams.update(mpl_config_screen)
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['figure.dpi'] = dpi
    if text_labels:
        if language == "en":
            config["labels"] = labels_en_text
        elif language == "de":
            config["labels"] = labels_de_text
    else:
        if language == "en":
            config["labels"] = labels_en_mathtext
        elif language == "de":
            config["labels"] = labels_de_mathtext
    config["figure_size"] = [px/dpi for px in figure_sizes_screen[screen_size]]


# Configure plotting
def configure(mode, **kwargs):
    plt.ioff()
    if mode == "screen":
        enable_screen_mode(**kwargs)
    elif mode == "latex":
        enable_latex_mode(**kwargs)
    else:
        raise ValueError(f"Unsupported plotting mode '{mode}'")


# Plot configuration context manager
class PlotConfig():

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        # Backup configuration
        self.config_backup = config.copy()
        self.rcParams_backup = plt.rcParams.copy()
        config.update(self.kwargs)
        configure(**config)

    def __exit__(self, *args):
        # Save figure
        if config["filename"] is not None:
            plt.savefig(config["filename"], bbox_inches='tight', pad_inches=0)
        # Show or hide figure
        if config["show"]:
            plt.show()
        else:
            plt.close()
        # Restore configuration
        config.update(self.config_backup)
        plt.rcParams.update(self.rcParams_backup)


def get_label_residual(key, length):
    var_key = key.split("_residual")[0]
    var_label = get_label(var_key, length)
    res_label = get_label("residual")
    if var_label.startswith("$") and var_label.endswith("$"):
        # The base label is already math, and the math/latex templates own the
        # delimiters ("$\\Delta\\,{}$"), so drop the base label's own ones.
        var_label = var_label[1:-1]
    elif "$" in res_label:
        # A plain label (unknown key) substituted into a math template: escape
        # the underscores, which mathtext would otherwise read as subscripts
        # ("z_g_tf" -> double subscript).
        var_label = var_label.replace("_", r"\_")
    return res_label.format(var_label)


def get_label(key, length=None):
    latex_mode = plt.rcParams['text.usetex'] is True
    if key.endswith("_residual"):
        return get_label_residual(key, length)
    if key not in config["labels"]:
        # print("No label found for key: {}".format(key))
        if latex_mode:
            label = key.replace('_', ' ')
        else:
            label = key
    else:
        label = config["labels"][key]
    # If a length is given and if non latex mode, wrap label
    if length and not latex_mode:
        return '\n'.join(textwrap.wrap(label, length))
    else:
        return label


def limit_axis_scale(ax, col):
    """Limit axis scale to reasonable values to avoid extreme scaling in case
    of noisy data"""
    ylim = list(ax.get_ylim())
    ylim[0] = min(ylim[0], -1e-3)
    ylim[1] = max(ylim[1], 1e-3)
    ax.set_ylim(ylim)


def rotate_ylabels(axes, only_if_necessary=True):
    """Rotate ylabels if necessary to avoid overlapping with other subplots"""
    # Check if any y-labels exceed the subplot height
    rotate = False
    max_height = 0
    for ax in axes:
        ylabel_extents = ax.yaxis.label.get_window_extent()
        subplot_extents = ax.get_window_extent()
        max_height = max(max_height, ylabel_extents.height)
        if ylabel_extents.height > subplot_extents.height*0.9:
            rotate = True
    # Rotate all or none of the y-labels
    if rotate or not only_if_necessary:
        for ax in axes:
            ylabel_extents = ax.yaxis.label.get_window_extent()
            # Rotate label and add label padding
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_verticalalignment("center")
            ax.yaxis.label.set_horizontalalignment("right")
            # ax.yaxis.labelpad = max_height*0.5 + ylabel_extents.width*2


def plot_obstacles(ax, obstacles):
    for obs in obstacles:
        obs_patch = Polygon(obs, facecolor='#99ccff', edgecolor='#6699cc')
        ax.add_patch(obs_patch)


def plot_wind_vectors(ax, wind_speed=0, wind_dir=None, grid_scale=150.0, n=8):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_grid = (xlim[1] - xlim[0])/n
    y_grid = (ylim[1] - ylim[0])/n
    if wind_speed == 0:
        return
    arrow_color = mcolors.to_rgba('lightblue', alpha=0.6)
    for row in range(-n, n + 1):
        for column in range(-n, n + 1):
            arrow = [column*grid_scale, row*grid_scale] + [
                x_grid/2*math.sin(wind_dir), y_grid/2*math.cos(wind_dir)
            ]

            ax.arrow(
                *arrow,
                fc=arrow_color,
                ec=arrow_color,
                head_width=max(x_grid, y_grid)/10,
                head_length=max(x_grid, y_grid)/10)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_limits(
        ax,
        lower=None,
        upper=None,
        fill_color="#dddddd",
        preserve_ylimits=False):
    initial_ylimits = ax.get_ylim()
    if isinstance(lower, dict):
        lower_x = lower.keys()
        lower_y = lower.values()
    elif isinstance(lower, pandas.Series):
        lower_x = lower.index.values
        lower_y = lower.values
    else:
        lower_x = None
        lower_y = lower
    if isinstance(upper, dict):
        upper_x = upper.keys()
        upper = upper.values()
    elif isinstance(upper, pandas.Series):
        upper_x = upper.index.values
        upper_y = upper.values
    else:
        upper_x = None
        upper_y = upper
    if upper_y is not None:
        if upper_x is not None:
            ax.step(upper_x, upper_y, color=fill_color, where='post')
            ylimits = ax.get_ylim()
            ax.fill_between(
                upper_x,
                upper_y,
                ylimits[1],
                step="post",
                color=fill_color,
                alpha=0.5)
            ax.set_ylim(ylimits)
        else:
            # ylimits = ax.get_ylim()
            # print("upper_y", upper_y, "ylimits", ylimits)
            ax.axhline(upper_y, color=fill_color)
            ylimits = ax.get_ylim()
            ax.axhspan(upper_y, ylimits[1], color=fill_color, alpha=0.5)
            ax.set_ylim(ylimits)
    if lower_y is not None:
        if lower_x is not None:
            ax.step(lower_x, lower_y, color=fill_color, where='post')
            ylimits = ax.get_ylim()
            ax.fill_between(
                lower_x,
                lower_y,
                ylimits[0],
                step="post",
                color=fill_color,
                alpha=0.5)
            ax.set_ylim(ylimits)
        else:
            ax.axhline(lower_y, color=fill_color)
            ylimits = ax.get_ylim()
            ax.axhspan(ylimits[0], lower_y, color=fill_color, alpha=0.5)
            ax.set_ylim(ylimits)
    if preserve_ylimits:
        ax.set_ylim(initial_ylimits)


def set_3d_axes_equal(ax, z_scale=1, z_min=None, z_max=0):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    ax.set_box_aspect((1, 1, 1/z_scale))

    if z_min is not None:
        ax.set_zlim3d([z_min, ax.get_zlim3d()[1]])
    elif z_max is not None:
        ax.set_zlim3d([ax.get_zlim3d()[0], z_max])


def get_legend_handles(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    return by_label.values(), by_label.keys()


def get_default_colors():
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # colors = mcolors.TABLEAU_COLORS.values()
    return colors


def get_color_names():
    return list(mcolors.BASE_COLORS.keys()) \
        + list(mcolors.CSS4_COLORS.keys()) \
        + list(mcolors.TABLEAU_COLORS.keys())


def get_default_styles(n=5):
    colors = get_default_colors()
    line_styles = ["-", "--", ":", "-."]
    default_styles = [
        {
            "linewidth": 1,
            "color": colors[i % len(colors)],
            "linestyle": line_styles[i % len(line_styles)]
        } for i in range(n)
    ]
    return default_styles


def resolve_plot_style_shorthand(style):
    """Get a plot style dictionary from a shorthand string or list thereof."""
    if isinstance(style, dict):
        return style
    elif isinstance(style, (list, tuple)):
        resolved_items = [resolve_plot_style_shorthand(item) for item in style]
        if all(isinstance(item, dict) for item in resolved_items):
            all_keys = [k for item in resolved_items for k in item.keys()]
            if len(set(all_keys)) == len(all_keys):
                # Unique keys -> merge into one style dict
                return {
                    k: v
                    for item in resolved_items
                    for k, v in item.items()
                }
            else:
                # Non-unique keys -> return as list of style dicts
                return resolved_items
        else:
            unresolved_items = [
                item for item in resolved_items if not isinstance(item, dict)
            ]
            resolved_items = [
                item for item in resolved_items if isinstance(item, dict)
            ]
            for item in unresolved_items:
                resolved_items.append(resolve_plot_style_shorthand(item))
            return resolved_items
    else:
        color_names = get_color_names()
        try:
            if style in ['-', '--', ':', '-.']:
                return {"linestyle": style}
            elif style in color_names:
                return {"color": style}
            elif float(style) > 0:
                return {"linewidth": float(style)}
        except ValueError:
            print("Failed to interpret plot style shorthand: {}".format(style))
            raise


def get_linestyles(user_styles, default_styles=[], n=5):
    user_styles = [resolve_plot_style_shorthand(sty) for sty in user_styles]
    styles = get_default_styles(n)
    for i in range(min(len(styles), len(user_styles))):
        styles[i].update(user_styles[i])
    return styles


def merge_styles(default_styles, user_styles):
    styles = [style.copy() for style in default_styles]
    [styles[i].update(user_styles[i]) for i in range(len(user_styles))]
    return styles


def plot_3D_view(
        trajs,
        styles=[],
        x_col='x_g',
        y_col='y_g',
        z_col='z_g',
        s_col="V_A",
        z_scale=1,
        xy_projection=True,
        title=True,
        POIs=[],
        fig=None,
        axis=None):
    # Pre-process user input
    if isinstance(trajs, dict):
        traj_labels = list(trajs.keys())
        trajs = list(trajs.values())
    else:
        trajs = utils.listify(trajs)
        traj_labels = [f"traj. {idx}" for idx in range(len(trajs))]

    if fig is None or axis is None:
        fig, axis = plt.subplots(
            1, figsize=config["figure_size"], subplot_kw={"projection": "3d"})

    default_styles = get_default_styles()
    styles = merge_styles(default_styles, styles)
    if title:
        axis.set_title(get_label("3D view"), y=1, pad=-2)

    # Draw xy projection
    if xy_projection:
        for idx, traj in enumerate(trajs):
            for t, row in traj.iloc[::2].iterrows():
                # for t, row in traj.iterrows():
                axis.plot(
                    [row[x_col]]*2, [row[y_col]]*2, [0, row[z_col]],
                    linewidth=2,
                    color="k",
                    alpha=0.1)

    if s_col is not None:
        # Multi-colored line plot (https://matplotlib.org/examples/pylab_examples/multicolored_line.html)
        cmap = plt.cm.plasma
        z_min = min([traj[s_col].min() for idx, traj in enumerate(trajs)])
        z_max = max([traj[s_col].max() for idx, traj in enumerate(trajs)])
        z_norm = mpl.colors.Normalize(vmin=z_min*0.8, vmax=z_max*1.2)

        for idx, traj in enumerate(trajs):
            # Make a regular plot first, to have the axis scaled properly
            axis.plot(
                traj[x_col].values,
                traj[y_col].values,
                traj[z_col].values,
                linewidth=.1)
            points = numpy.array(
                [traj[x_col].values, traj[y_col].values,
                 traj[z_col].values]).T.reshape(-1, 1, 3)
            segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
            lc = Line3DCollection(segments, cmap=cmap, norm=z_norm, zorder=10)
            lc.set_array(traj[s_col].values)
            axis.add_collection(lc)
        # Add colorbar
        sc_mappable = ScalarMappable(cmap=cmap, norm=z_norm)
        colorbar = fig.colorbar(sc_mappable, ax=axis)
        colorbar.set_label(get_label(s_col), rotation=90)
    else:
        for idx, traj in enumerate(trajs):
            # Regular plot
            axis.plot(
                traj[x_col].values,
                traj[y_col].values,
                traj[z_col].values,
                label=get_label(traj_labels[idx]) if len(trajs) > 1 else None,
                **styles[idx % len(styles)])

    # Draw markers for POIs
    for poi in POIs:
        poi_kwargs = {
            k: v
            for k, v in poi.items() if k not in (x_col, y_col, z_col)
        }
        axis.scatter(
            poi[x_col], poi[y_col], poi[z_col], **poi_kwargs, depthshade=False)
    if POIs:
        # axis.legend(loc="lower right", bbox_to_anchor=(0.8, 0.25), ncol=2)
        # axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    # Axis scaling and direction
    set_3d_axes_equal(axis, z_scale)
    # Rotate to get NED frame
    axis.view_init(elev=-20, azim=140, roll=180)
    # FOV = 0 deg, default is "persp" with focal length = 1
    axis.set_proj_type('ortho')

    # Axis label
    axis.xaxis.set_rotate_label(False)
    axis.yaxis.set_rotate_label(False)
    axis.zaxis.set_rotate_label(False)
    # ax1.xaxis.labelpad=50
    # ax1.yaxis.labelpad=50
    # ax1.zaxis.labelpad=-8
    axis.set_xlabel(get_label(x_col), rotation=0)
    axis.set_ylabel(get_label(y_col), rotation=0)
    axis.set_zlabel(get_label(z_col), rotation=90)
    # Reduce label padding
    axis.xaxis.labelpad = 4
    axis.yaxis.labelpad = 4
    axis.zaxis.labelpad = 4

    # # Work-around to move the whole plot to the left to avoid z-axis label cut of on the right
    # x0, y0, w, h = axis.get_position().bounds  # returns (x0, y0, w, h)
    # axis.set_position([x0 - 0.03, y0 + 0.15, w, h])

    return fig, axis


def plot_topview_with_airpath(
        trajs,
        styles=[],
        x_col='x_g',
        y_col='y_g',
        s_col=None,
        wind_speed=None,
        wind_dir=None,
        POIs=[],
        title=True,
        fig=None,
        axis=None):
    """Plot top view and airpath side-by-side."""
    # Listify input parameters.
    trajs = utils.listify(trajs)
    styles = utils.listify(styles)
    # x_col = utils.listify(x_col, len(trajs))
    # y_col = utils.listify(y_col, len(trajs))
    s_col = utils.listify(s_col, repeat=len(trajs))
    # Get a list of styles from default and user-provided styles.
    default_styles = get_default_styles()
    styles = merge_styles(default_styles, styles)
    # Create figure and axes if none are given.
    if fig is None or axis is None:
        fig, axes = plt.subplots(1, 2, figsize=config["figure_size"])
    # Plot topview into left plot
    plot_topview(
        trajs, styles, x_col, y_col, s_col, wind_speed, wind_dir, POIs, title,
        fig, axes[0])
    plot_topview(
        trajs, styles, "x_a", "y_a", s_col, wind_speed, wind_dir, POIs,
        "air-relative path", fig, axes[1])

    fig.tight_layout()
    fig.align_ylabels()
    # Add legend if multiple datasets were given
    if len(trajs) > 0:
        axes[-1].legend(
            bbox_to_anchor=(0.5, -0.6),
            loc="lower center",
            borderaxespad=0.,
            ncol=min(len(trajs), 3),
            frameon=False,
            handletextpad=0.5,
            handlelength=1.5)
        # fig.subplots_adjust(bottom=0.12)
    return fig, axes


def plot_topview(
        trajs,
        styles=[],
        x_col='x_g',
        y_col='y_g',
        s_col=None,
        wind_speed=None,
        wind_dir=None,
        POIs=[],
        title=True,
        fig=None,
        axis=None,
        legend=True):
    # Pre-process user input
    if isinstance(trajs, dict):
        trajs = list(trajs.values())
    else:
        trajs = utils.listify(trajs)
    styles = utils.listify(styles)
    # x_col = utils.listify(x_col, len(trajs))
    # y_col = utils.listify(y_col, len(trajs))
    s_col = utils.listify(s_col, repeat=len(trajs))
    # Create figure and axes if none are given.
    if fig is None or axis is None:
        fig, axis = plt.subplots(1, figsize=config["figure_size"])
    # Get a list of styles from default and user-provided styles.
    default_styles = get_default_styles()
    styles = merge_styles(default_styles, styles)
    if title:
        if isinstance(title, str):
            axis.set_title(title)
        else:
            axis.set_title(get_label("top view"))
    if s_col[0] is not None:
        # Multi-colored line plot (https://matplotlib.org/examples/pylab_examples/multicolored_line.html)
        cmap = plt.cm.plasma
        z_min = min([traj[s_col[idx]].min() for idx, traj in enumerate(trajs)])
        z_max = max([traj[s_col[idx]].max() for idx, traj in enumerate(trajs)])
        z_norm = mpl.colors.Normalize(vmin=z_min*0.8, vmax=z_max*1.2)
        for idx, traj in enumerate(trajs):
            # Make a regular plot first, to have the axis scaled properly
            axis.plot(traj[y_col].values, traj[x_col].values, linewidth=.1)
            points = numpy.array([traj[y_col].values,
                                  traj[x_col].values]).T.reshape(-1, 1, 2)
            segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
            lc = mpl.collections.LineCollection(
                segments, cmap=cmap, norm=z_norm, zorder=10)
            lc.set_array(traj[s_col[idx]].values)
            axis.add_collection(lc)
        # Add colorbar
        colorbar = fig.colorbar(
            lc, ax=axis, location="right", fraction=0.046, pad=0.04)
        colorbar.set_label(get_label(s_col[0]), rotation=90)
    else:
        # Regular plot
        for idx, traj in enumerate(trajs):
            axis.plot(
                traj[y_col].values, traj[x_col].values,
                **styles[idx % len(styles)])
    # Draw markers for POIs
    for poi in POIs:
        poi_kwargs = {
            k: v
            for k, v in poi.items() if k not in (x_col, y_col, "z_g")
        }
        axis.scatter(poi[y_col], poi[x_col], **poi_kwargs)
        # if "marker" in poi:
        #     axis.scatter(
        #         poi[y_col],
        #         poi[x_col],
        #         marker=poi["marker"],
        #         label=poi["label"])
        # else:
        #     axis.plot(poi[y_col], poi[x_col], poi["style"], label=poi["label"])
        # arrow_length = min(
        #     axis.get_xlim()[1] - axis.get_xlim()[0],
        #     axis.get_ylim()[1] - axis.get_ylim()[0])*0.05
        # dx = math.cos(poi["chi"])*arrow_length
        # dy = math.sin(poi["chi"])*arrow_length
        # axis.arrow(
        #     poi[y_col],
        #     poi[x_col],
        #     dy,
        #     dx,
        #     head_width=arrow_length,
        #     head_length=arrow_length,
        #     fc="g",
        #     ec="g")
    if POIs and legend:
        axis.legend()
    if wind_speed and wind_dir:
        xlim, ylim = axis.get_xlim(), axis.get_ylim()
        plot_wind_vectors(
            axis, wind_speed, wind_dir, grid_scale=0.15*(xlim[1] - xlim[0]))
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
    # Make sure the plot is at least square or wider
    ylim_dist = axis.get_ylim()[1] - axis.get_ylim()[0]
    xlim_dist = max(ylim_dist, axis.get_xlim()[1] - axis.get_xlim()[0])
    x_axis_center = axis.get_xlim()[0] + (
        axis.get_xlim()[1] - axis.get_xlim()[0])*0.5
    axis.set_xlim(x_axis_center - 0.5*xlim_dist, x_axis_center + 0.5*xlim_dist)
    axis.set_aspect(1, adjustable="box")
    # axis.set_aspect(1)
    axis.set_xlabel(get_label(y_col[0]))
    axis.set_ylabel(get_label(x_col[0]))
    # fig.tight_layout()
    return fig, axis


def plot_vert_profile(
        trajs, styles=[], dxy_col='l_xy', z_col='h_g', fig=None, axis=None):
    # Pre-process user input
    if isinstance(trajs, dict):
        trajs = list(trajs.values())
    else:
        trajs = utils.listify(trajs)
    styles = utils.listify(styles)
    dxy_col = utils.listify(dxy_col, repeat=len(trajs))
    z_col = utils.listify(z_col, repeat=len(trajs))
    # Create figure and axis if none are given.
    if fig is None or axis is None:
        fig, axis = plt.subplots(1, figsize=config["figure_size"])
    # Get a list of styles from default and user-provided styles.
    default_styles = get_default_styles()
    styles = merge_styles(default_styles, styles)
    axis.set_title(get_label("vert. profile"))
    for idx, traj in enumerate(trajs):
        axis.plot(
            traj[dxy_col[idx]].values, traj[z_col[idx]].values,
            **styles[idx % len(styles)])
    axis.set_xlabel(get_label(dxy_col[0]))
    axis.set_ylabel(get_label(z_col[0]))
    # fig.tight_layout()
    return fig, axis


def plot_topview_vert_profile(
        trajs,
        styles=[],
        x_col='x_g',
        y_col='y_g',
        z_col='h_g',
        dxy_col='l_xy',
        wind_vec=None,
        fig=None,
        axes=None):
    """top view + vertical profile in one square plot."""
    # Create figure and axes if none are given.
    if fig is None or axes is None:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=config["figure_size"],
            gridspec_kw={
                "height_ratios": [3, 1],
                "hspace": 0.35
            })
    plot_topview(trajs, styles, x_col, y_col, None, wind_vec, fig, axes[0])
    plot_vert_profile(trajs, styles, dxy_col, z_col, fig, axes[1])
    # fig.tight_layout()
    fig.align_ylabels()
    # Add legend if multiple datasets were given
    if len(trajs) > 0:
        axes[-1].legend(
            bbox_to_anchor=(0.5, -0.6),
            loc="lower center",
            borderaxespad=0.,
            ncol=min(len(trajs), 3),
            frameon=False,
            handletextpad=0.5,
            handlelength=1.5)
        # fig.subplots_adjust(bottom=0.12)
    return fig, axes


def get_ax_size(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return bbox.width, bbox.height


def plot_timeline(
        trajectories,
        subplots,
        plot_styles=[],
        time_col='index',
        bounds=[],
        t0=None,
        fig=None,
        axes=None,
        traj_legend_y_pos=-0.53,
        traj_legend_n_col=3,
        title=False,
        columns=2,
        legend=True,
        verbose=True,
        **plot_kwargs):
    """ Create a timeline plot.
    trajectories: A single dataframe or dict with labels as keys and dataframes as values

    subplots: List of N subplot specifications. Each subplot specification is either
        - a single column to be plotted for each trajectory
        - a list of columns to be plotted for each trajectory
        - a subplot specification dict with keys:
            - columns: List of columns to plot from the trajectory dataframes.
            - (optional) ylabel: Y-axis label
            - (optional) styles: List of plot styles (shorthands) for each column to plot.
            - (optional) plot_kwargs: List of additional keyword arguments to be passed to the plot function.
        - a dict of subplot specification per trajectory with traj. labels as keys

    NOTE: subplots will be processed into a dict (N trajectory labels as keys) of lists (K subplots)
    of subplot spec dictionaries.

    TODO: This needs to be refactored and simplified, possibly split up into multiple functions.
    """
    # Pre-process user input
    if isinstance(trajectories, dict):
        traj_labels = list(trajectories.keys())
        trajectories = list(trajectories.values())
    else:
        trajectories = utils.listify(trajectories)
        traj_labels = [f"traj. #{idx}" for idx in range(len(trajectories))]
    n_traj = len(trajectories)

    # Make subplots a dict with traj. labels as keys
    if not isinstance(subplots, dict) or not all(k in traj_labels
                                                 for k in subplots):
        subplots = {traj_label: subplots for traj_label in traj_labels}

    # Convert all subsplot specifications to lists of dicts
    n_timelines = 0
    for traj_label in traj_labels:
        subplots[traj_label] = utils.listify(subplots[traj_label])
        n_timelines = max(n_timelines, len(subplots[traj_label]))
        for idx in range(len(subplots[traj_label])):
            # If not a dict, make it one
            if not isinstance(subplots[traj_label][idx], dict):
                subplot_spec = {
                    "columns": utils.listify(subplots[traj_label][idx])
                }
            else:
                subplot_spec = subplots[traj_label][idx]
            # Make sure all required keys are present
            if "ylabel" not in subplot_spec:
                subplot_spec["ylabel"] = utils.listify(
                    subplot_spec["columns"])[0]
            if "styles" not in subplot_spec:
                subplot_spec["styles"] = {}
            if "plot_kwargs" not in subplot_spec:
                subplot_spec["plot_kwargs"] = {}
            # Make sure columns, styles and plot_kwargs are lists
            subplot_spec["columns"] = utils.listify(subplot_spec["columns"])
            n_cols = len(subplot_spec["columns"])
            subplot_spec["styles"] = utils.listify(
                subplot_spec["styles"], repeat=n_cols)
            subplot_spec["plot_kwargs"] = utils.listify(
                subplot_spec["plot_kwargs"], repeat=n_cols)
            # Set result
            subplots[traj_label][idx] = subplot_spec

    # Get default plot styles as baseline and apply user styles on top
    plot_styles = utils.listify(plot_styles)
    user_styles = plot_styles
    plot_styles = get_default_styles()
    for i in range(len(user_styles)):
        plot_styles[i].update(user_styles[i])

    # Create figure and axes if none are given.
    if fig is None or axes is None:
        fig, axes = plt.subplots(
            math.ceil(n_timelines/columns),
            columns,
            figsize=config["figure_size"],
            sharex=True,
            layout="constrained")
        # Make sure axes is a flat list of axes objects
        if n_timelines == 1:
            axes = [axes]
        if columns > 1:
            axes = [ax_list[idx] for idx in range(columns) for ax_list in axes]
            # axes = [ax for ax_list in axes for ax in ax_list]
        # Disable empty subplots
        if n_timelines < len(axes):
            for ax in axes[n_timelines:]:
                ax.axis("off")
    if title is True:
        if n_timelines > 1:
            axes[0].set_title(get_label("timelines"))
        else:
            axes[0].set_title(get_label("timeline"))
    elif title is not False:
        axes[0].set_title(get_label(title))
    assert (n_timelines <= len(axes))

    traj_plot_handles = []
    for traj_label, subplot_specs in subplots.items():
        if verbose:
            print(
                f"Creating {len(subplot_specs)} timeline plots for trajectory '{traj_label}'"
            )
        # Get dataframe from label
        traj_idx = traj_labels.index(traj_label)
        traj = trajectories[traj_idx]
        for subplot_idx, subplot_spec in enumerate(subplot_specs):
            if verbose:
                print(f"Subplot specification #{subplot_idx}: {subplot_spec}")
            # Get time values
            if time_col == 'index':
                t_values = traj.index.values
            else:
                t_values = traj[time_col].values
            if t0 is not None:
                t_values = t_values - (t_values[0] + t0)
            # Get axis to draw on
            axis = axes[subplot_idx]
            # Plot each column with its designated style
            plot_handles = []
            for col_idx, col in enumerate(subplot_spec["columns"]):
                handle = axis.plot(
                    t_values,
                    traj[col].values,
                    label=get_label(traj_label),
                    zorder=col_idx,
                    **{
                        **plot_styles[(
                            traj_idx*len(subplot_spec["columns"]) + col_idx) % len(
                                          plot_styles)],
                        **plot_kwargs,
                        **subplot_spec["styles"][col_idx],
                        **subplot_spec["plot_kwargs"][col_idx],
                    })
                plot_handles.append(handle)
                # Plot limits for this column
                col_upper = bounds[col +
                                   '_max'] if col + '_max' in bounds else None
                col_lower = bounds[col +
                                   '_min'] if col + '_min' in bounds else None
                if col_upper or col_lower:
                    plot_limits(axis, upper=col_upper, lower=col_lower)
            # Set y-axis label
            axis.set_ylabel(get_label(subplot_spec["ylabel"]))
            if "ylim" in subplot_spec:
                axis.set_ylim(subplot_spec["ylim"])
            # Make space for legend inside the plot by adjusting y-axis limits
            if len(plot_handles) > 1:
                axis_height = get_ax_size(fig, axis)[1]
                ylims = axis.get_ylim()
                axis_yrange = ylims[1] - ylims[0]
                legend_fs_inch = plt.rcParams["legend.fontsize"]/72
                axis_scale = axis_yrange/axis_height
                yrange_offset = 2.5*legend_fs_inch*axis_scale
                axis.set_ylim([ylims[0] - yrange_offset, ylims[1]])
                # Create subplot legend
                subplot_legend = axis.legend(
                    handles=plot_handles,
                    bbox_to_anchor=(0.0, 0.0, 1.0, 0.1),
                    loc="lower right",
                    borderaxespad=0.,
                    ncol=len(plot_handles),
                    frameon=False,
                    handletextpad=0.5,
                    handlelength=1.5)
                axis.add_artist(subplot_legend)
                traj_plot_handles.append(plot_handles[0][0])
            # # Ensure there are ticks for min/max values
            # y_min = min(
            #     [min(traj[col].values) for col in subplot_spec["columns"]])
            # y_max = max(
            #     [max(traj[col].values) for col in subplot_spec["columns"]])
            # current_yticks = list(axis.get_yticks())
            # all_yticks = sorted(list(set(current_yticks + [y_min, y_max])))
            # axis.set_yticks(all_yticks)
            # axis.set_yticks(current_yticks)

        traj_plot_handles.append(plot_handles[0][0])

    # Add shared x-label and set some decent layout defaults
    axes[-1].set_xlabel(get_label("time"))
    rotate_ylabels(axes)
    fig.align_ylabels()
    # Add trajectory legend if multiple datasets were given
    if n_traj > 1 and legend:
        # print(traj_plot_handles)
        # print(traj_labels)
        axes[-1].legend(
            handles=traj_plot_handles,
            labels=traj_labels,
            # bbox_to_anchor=(0.5, traj_legend_y_pos),
            # loc="lower center",
            # borderaxespad=0.,
            # ncol=min(len(traj_plot_handles), traj_legend_n_col),
            # frameon=False,
            handletextpad=0.5,
            handlelength=1.5)
        # fig.subplots_adjust(
        #     bottom=0.2*math.ceil(len(traj_plot_handles)/traj_legend_n_col))
    # NOTE: This should not be required with constrained layout
    # # Work-around to not include overlapping ylabels into layout:
    # # (https://github.com/matplotlib/matplotlib/issues/13650#issuecomment-579381380)
    # for idx, ylabel in ylabels.items():
    #     axes[idx].set_ylabel(ylabel)
    return fig, axes


def plot_ref_trajectory(
        paths,
        wind_speed=None,
        wind_dir=None,
        _3D_view=True,
        top_view=True,
        vert_profile=True,
        s_col="V_A",
        xy_projection=True,
        x_col='x_g',
        y_col='y_g',
        z_col='z_g',
        z_scale=2,
        timelines=[],
        POIs=[]):
    """One-page plot with 3D view, top view, vertical profile and timelines"""
    figsize = config["figure_size"]

    # Calculate height ratio and total figure height (1: figure width)
    height_ratios = []
    subplots_mosaic = []
    per_subplot_kw = {}
    # 3D view should be in a square grid cell
    if _3D_view:
        height_ratios.append(1)
        subplots_mosaic.append(["3D view", "3D view"])
        per_subplot_kw["3D view"] = {
            "projection": "3d",
            # "xmargin": 0.5,
            # "ymargin": 0.5
        }
    # top view and vertical profile should be in a 2x1 grid cell
    if top_view and vert_profile:
        height_ratios.append(.33)
        subplots_mosaic.append(["top view", "vert. profile"])
    elif top_view:
        height_ratios.append(.33)
        subplots_mosaic.append(["top view", "top view"])
    # Timelines should be in a 3x1 grid cell
    height_ratios += [0.15]*len(timelines)
    timeline_keys = [
        "".join(tl["columns"]) if isinstance(tl, dict) else tl
        for tl in timelines
    ]
    subplots_mosaic += [[tl, tl] for tl in timeline_keys]

    # Check if we exceeded the configured max. figure height
    vertical_space_left = config["figure_size"][1] - sum(
        height_ratios)*config["figure_size"][0]
    if vertical_space_left < 0:
        print("WARNING: Vertical space exceeded. Increasing figure height.")
        figsize[1] = sum(height_ratios)*config["figure_size"][0]

    # Create the figure, grid and axes
    fig, axes = plt.subplot_mosaic(
        subplots_mosaic,
        gridspec_kw={"height_ratios": height_ratios},
        per_subplot_kw=per_subplot_kw,
        layout="constrained",
        figsize=figsize)

    styles = get_default_styles()

    # 3D plot
    if _3D_view:
        plot_3D_view(
            paths,
            styles,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            s_col=s_col,
            xy_projection=xy_projection,
            fig=fig,
            axis=axes["3D view"],
            z_scale=z_scale,
            POIs=POIs)

    # top view and vertical profile
    if top_view:
        plot_topview(
            paths,
            styles,
            x_col=x_col,
            y_col=y_col,
            s_col=s_col,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            POIs=POIs,
            fig=fig,
            axis=axes["top view"],
            legend=False)
    if vert_profile:
        plot_vert_profile(paths, styles, fig=fig, axis=axes["vert. profile"])

    # Timelines
    if timelines:
        plot_timeline(
            paths,
            timelines,
            styles,
            fig=fig,
            axes=[axes[tk] for tk in timeline_keys],
            title=True,
            legend=False)

    # # Layout tuning
    # fig.tight_layout()
    # fig.align_ylabels()
    # fig.subplots_adjust(bottom=0.05)
    # # Adjust position/size of 3D view (must be done after call to 'tight_layout()')
    # orig_pos = axes["3D view"].get_position()
    # orig_pos.x0 -= .13
    # orig_pos.x1 += .1
    # axes["3D view"].set_position(orig_pos)

    return fig, axes


def plot_stats(datasets, col, xlabels, title="", fig=None, axis=None):
    """Bar plot over statistics of a series."""
    # Create figure and axes if none are given.
    if fig is None or axis is None:
        fig, axis = plt.subplots(1, 1, figsize=config["figure_size"])
    colors = plt.cm.Set3(numpy.linspace(0, 1, 12))
    axis.set_title(title)
    box = axis.boxplot(
        [dataset[col] for dataset in datasets],
        notch=False,
        patch_artist=True,
        whis=1e9)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    axis.set_ylabel(get_label(col))
    if xlabels:
        axis.tick_params(axis='x', which='major', pad=10)
        #        axis.tick_params(direction='out', pad=5)
        axis.set_xticklabels(
            xlabels,
            rotation=60,
            rotation_mode="anchor",
            ha="right",
            va="center")
    fig.tight_layout()


def plot_iteration_values(
        traces,
        x_col="step",
        y_col="value",
        group_col="span_id",
        title=None,
        xlabel="iteration",
        ylabel=None,
        logy=True,
        absolute=True,
        median=True,
        median_label="median",
        runs_label="individual run",
        median_min_support=0.05,
        alpha=0.5,
        color=None,
        grid_alpha=0.3,
        median_color="black",
        fig=None,
        axis=None,
        **plot_kwargs):
    """Plot value traces over multiple runs of an iterative algorithm from
    traces recorded by htrace module.

    traces: A dataframe with one row per recorded sample. It must contain
        ``x_col``, ``y_col`` and ``group_col``.

    Each entry of ``group_col`` identifies one *individual invocation* of the
    loop (one root solve, one bisection) and is drawn as one curve. With htrace
    traces, this is usually the "span_id".

    The ensemble is drawn semi-transparent so that its density stays readable,
    with a bold median curve on top. The median curve may be truncated where
    fewer than the mediam_min_support runs extend to higher values on x.

    The x-axis is shifted by one when the samples start at zero: htrace
    records one value per *iteration*, so index 0 already reflects the first
    iteration, and labelling it "iteration 0" would imply no work was done.
    """
    # Create figure and axes if none are given.
    if fig is None or axis is None:
        fig, axis = plt.subplots(
            1, 1, figsize=config["figure_size"], layout="constrained")
    if traces is None or len(traces) == 0:
        axis.axis("off")
        return fig, axis
    if absolute:
        traces = traces.assign(**{y_col: traces[y_col].abs()})
    # Label the abscissa by iterations performed rather than by sample index
    x_offset = 1 if traces[x_col].min() == 0 else 0
    # Get plot color
    if color is None:
        color = get_default_colors()[0]
    # Draw one semi-transparent curve per invocation of the loop. The first
    # curve carries the legend entry for all of them.
    plot_fn = axis.semilogy if logy else axis.plot
    runs_handle = None
    for idx, (_, curve) in enumerate(traces.groupby(group_col, sort=False)):
        curve = curve.sort_values(x_col)
        values = curve[y_col].to_numpy(dtype=float)
        if logy:
            values = numpy.clip(values, 1e-18, None)
        handle, = plot_fn(
            curve[x_col].to_numpy(dtype=float) + x_offset,
            values,
            color=color,
            linewidth=0.8,
            alpha=alpha,
            label=runs_label if idx == 0 else None,
            **plot_kwargs)
        if runs_handle is None:
            runs_handle = handle
    if median:
        # One median value per step, valid only while enough runs are still
        # sampled there.
        wide = traces.pivot_table(
            index=x_col, columns=group_col, values=y_col,
            aggfunc="first").sort_index()
        supported = wide.count(axis=1) >= max(
            1, int(median_min_support*traces[group_col].nunique()))
        median_curve = wide[supported].median(axis=1)
        plot_fn(
            median_curve.index + x_offset,
            median_curve.to_numpy(dtype=float),
            color=median_color,
            linewidth=1.8,
            label=median_label)
    if runs_handle is not None or median:
        axis.legend()
    axis.grid(True, which="both", alpha=grid_alpha)
    axis.set_axisbelow(True)
    # Snap the x-ticks to integers
    axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Name the y-axis after the recorded quantity. The frame's "variable"
    # column (htrace) names the quantity; get_label() resolves it, including
    # the "*_residual" suffix via get_label_residual().
    if ylabel is None:
        ylabel = y_col
        if "variable" in traces.columns and traces["variable"].nunique() == 1:
            ylabel = traces["variable"].iloc[0]
        ylabel = get_label(ylabel)
    axis.set_xlabel(get_label(xlabel))
    axis.set_ylabel(ylabel)
    if title is not None:
        axis.set_title(title)
    return fig, axis


def plot_iteration_histogram(
        rt_traces,
        span=None,
        iterations_counter="iterations",
        title=None,
        xlabel="iterations",
        ylabel="count",
        bins=None,
        color=None,
        grid_alpha=0.3,
        fig=None,
        axis=None,
        **hist_kwargs):
    """Histogram of how many iterations an iterative algorithm performed per run
    based on recorded htrace runtime traces.

    rt_traces: A runtime-trace frame as returned by
        ``logger.get_rt_traces()``, with one row per span.

    span: Optional span name to restrict to.

    iterations_counter: Name of the column holding the count.

    ``bins`` defaults to integer bins spanning the observed range.

    An empty frame, a missing column or a ``span`` with no rows switches the
    axis off rather than raising: a loop may legitimately record no counts.
    """
    # Create figure and axes if none are given.
    if fig is None or axis is None:
        fig, axis = plt.subplots(
            1, 1, figsize=config["figure_size"], layout="constrained")
    if rt_traces is None or len(rt_traces) == 0:
        axis.axis("off")
        return fig, axis
    if span is not None:
        rt_traces = rt_traces[rt_traces["span"] == span]
    if iterations_counter not in rt_traces.columns:
        axis.axis("off")
        return fig, axis
    values = rt_traces[iterations_counter].dropna()
    if values.empty:
        axis.axis("off")
        return fig, axis
    if bins is None:
        explicit_bins = False
        lo, hi = int(values.min()), int(max(values.max(), 1))
        bins = numpy.arange(lo - 0.5, hi + 1.5, 1)
    else:
        explicit_bins = True
    if color is None:
        color = get_default_colors()[0]
    axis.hist(values, bins=bins, color=color, **hist_kwargs)
    # Handle cases where iteration count is always the same
    if not explicit_bins and values.nunique() == 1:
        only = int(values.iloc[0])
        axis.set_xlim(only - 1.0, only + 1.0)
        axis.set_xticks([only - 0.5, only + 0.5])
    axis.grid(True, axis="y", alpha=grid_alpha)
    axis.set_axisbelow(True)
    # Counts are whole numbers, so snap the ticks to integers
    axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axis.set_xlabel(get_label(xlabel))
    axis.set_ylabel(get_label(ylabel))
    axis.set_title(title if title is not None else iterations_counter)
    return fig, axis


def plot_thrust_over_airspeed(
        ac,
        gamma_A=0,
        n=None,
        V_A_range=None,
        V_A_ref=None,
        fig=None,
        axis=None,
        show_V_A_C_L_max=True):
    gamma_A_values = utils.listify(gamma_A)
    n_values = utils.listify(n)
    # Create figure and axes if none are given.
    if fig is None or axis is None:
        fig, axis = plt.subplots(
            1, 1, figsize=config["figure_size"], layout="constrained")
    if V_A_range is None:
        V_A_range = [ac.V_A_max, ac.V_A_max]
        for gamma_A in gamma_A_values:
            V_A_range[0] = min(
                V_A_range[0],
                ac.airspeed_bounds(gamma_A=gamma_A)[0])
    V_A_values = numpy.linspace(V_A_range[0], V_A_range[1], 100)
    for gamma_A in gamma_A_values:
        for n in n_values:
            T_values = [
                utils.error2constant(ac.thrust)(V_A=V_A, gamma_A=gamma_A, n=n)
                for V_A in V_A_values
            ]
            axis.plot(
                V_A_values,
                T_values,
                label=f"gamma_A={math.degrees(gamma_A):.1f}, n={n}")

    # Draw thrust limits
    axis.axhline(ac.T_min, color="r")
    axis.axhline(ac.T_max, color="r")

    axis.legend()
    return fig, axis


def plot_doubleint_phase_plane(
        ax, traj, problem=None, ylabels=True, arrow_grid=None, **kwargs):
    ax.set_title(get_label("phase plane"))
    lines = ax.plot(traj.x, traj.v, **kwargs)
    # Plot arrows
    if arrow_grid:
        arrow_dx = arrow_grid[0]
        arrow_dv = arrow_grid[1]
        # Find a descent upper bound, j, not too close to the goal state
        x_last = traj.x.iloc[-1]
        v_last = traj.v.iloc[-1]
        for j in reversed(range(len(traj.x))):
            if abs(traj.x.iloc[j]-x_last) >= arrow_dx/2 \
                    or abs(traj.v.iloc[j]-v_last) >= arrow_dv/2:
                break
        # Draw circles
        x_last = traj.x.iloc[0]
        v_last = traj.v.iloc[0]
        for i in range(j):
            if abs(traj.x.iloc[i]-x_last) >= arrow_dx \
                    or abs(traj.v.iloc[i]-v_last) >= arrow_dv:
                #                print("arrow_grid",arrow_grid,"dx",abs(traj.x.iloc[i]-x_last),"dv",abs(traj.v.iloc[i]-v_last) )
                ax.quiver(
                    traj.x.iloc[i],
                    traj.v.iloc[i],
                    traj.x.iloc[i + 1] - traj.x.iloc[i],
                    traj.v.iloc[i + 1] - traj.v.iloc[i],
                    angles='xy',
                    width=.02,
                    color=lines[-1].get_color())
                x_last = traj.x.iloc[i]
                v_last = traj.v.iloc[i]
    ax.set_xlabel(get_label("z_g"))
    if ylabels:
        ax.set_ylabel(get_label("w_K_g"), horizontalalignment="left")
    if problem:
        ax.scatter(
            [problem.x0], [problem.v0],
            marker="o",
            s=20,
            facecolor='none',
            label=get_label("start"),
            color="dimgray")
        ax.scatter(
            [problem.xg], [problem.vg],
            marker="x",
            s=20,
            label=get_label("goal"),
            color="dimgray")


def plot_doubleint_timeline(axes, traj, ylabels=True, **kwargs):
    axes[0].set_title(get_label("timelines"))
    axes[0].plot(traj.index.values, traj.x, **kwargs)
    axes[1].plot(traj.index.values, traj.v, **kwargs)
    axes[2].plot(traj.index.values, traj.u, **kwargs)
    axes[2].set_xlabel("t (s)")
    # axes[2].get_shared_x_axes().join(axes[0], axes[1])
    if ylabels:
        axes[0].set_ylabel(get_label("z_g"))
        axes[1].set_ylabel(get_label("w_K_g"))
        axes[2].set_ylabel(get_label("w_K_g_dot"))
        # align_ylabels(axes)


def plot_doubleint(
        topt_problem,
        topt_traj,
        fixed_problem=None,
        fixed_traj=None,
        fig=None,
        subplot_spec=None,
        ylabels=True):
    # Configure subplots and axes
    if not fig:
        fig = plt.figure(figsize=(3.5, 5))
    if not subplot_spec:
        outer_grid = gridspec.GridSpec(
            2, 1, height_ratios=[1, 1.25], bottom=0.2, left=0.2, hspace=0.5)
    else:
        outer_grid = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec, height_ratios=[1, 1], hspace=0.5)
    timeline_grid = gridspec.GridSpecFromSubplotSpec(3, 1, outer_grid[1, 0])
    phase_plane_axis = fig.add_subplot(outer_grid[0, 0])
    timeline_axes = [fig.add_subplot(timeline_grid[0, 0])]
    plt.setp(timeline_axes[-1].get_xticklabels(), visible=False)
    timeline_axes.append(
        fig.add_subplot(timeline_grid[1, 0], sharex=timeline_axes[-1]))
    plt.setp(timeline_axes[-1].get_xticklabels(), visible=False)
    timeline_axes.append(
        fig.add_subplot(timeline_grid[2, 0], sharex=timeline_axes[-1]))
    # Time-opt. plot
    arrow_grid = (
        max(
            topt_traj.x.max() - topt_traj.x.min(), (
                fixed_traj.x.max() -
                fixed_traj.x.min() if fixed_traj is not None else 0))/3,
        max(
            topt_traj.v.max() - topt_traj.v.min(), (
                fixed_traj.v.max() -
                fixed_traj.v.min() if fixed_traj is not None else 0))/3)
    topt_kwargs = {
        "label": get_label("time-optimal solution"),
        "linestyle": "-"
    }
    plot_doubleint_phase_plane(
        phase_plane_axis, topt_traj, topt_problem, ylabels, arrow_grid,
        **topt_kwargs)
    plot_doubleint_timeline(timeline_axes, topt_traj, ylabels, **topt_kwargs)
    # (Optional) fixed time plot
    if fixed_problem is not None and fixed_traj is not None:
        fixed_kwargs = {
            "label": get_label("fixed final time solution"),
            "linewidth": 1.5,
            "linestyle": ":"
        }
        plot_doubleint_phase_plane(
            phase_plane_axis, fixed_traj, fixed_problem, ylabels, arrow_grid,
            **fixed_kwargs)
        plot_doubleint_timeline(
            timeline_axes, fixed_traj, ylabels, **fixed_kwargs)
    # Invert y-axis to have increasing climb speed and altitude up
    phase_plane_axis.invert_yaxis()
    for t_axis in timeline_axes:
        t_axis.invert_yaxis()
    # Make a legend if this is not a subplot
    legend_handles, legend_labels = get_legend_handles(phase_plane_axis)
    if not subplot_spec:
        timeline_axes[-1].legend(
            legend_handles,
            legend_labels,
            loc=9,
            bbox_to_anchor=(0.0, -0.5, 1.0, -0.5),
            ncol=2,
            mode="expand")
    return fig, legend_handles, legend_labels


def plot_doubleint_multi(
        topt_problems=None,
        topt_trajs=None,
        fixed_problems=None,
        fixed_trajs=None):
    fig = plt.figure(figsize=(5, 5))
    grid = gridspec.GridSpec(
        1, len(topt_problems), bottom=0.2, left=0.15, wspace=0.3)
    for i in range(len(topt_problems)):
        fig, legend_handles, legend_labels = plot_doubleint(
            topt_problems[i] if topt_problems else None,
            topt_trajs[i] if topt_trajs else None,
            fixed_problems[i] if fixed_problems else None,
            fixed_trajs[i] if fixed_trajs else None,
            fig,
            grid[0, i],
            ylabels=(i == 0))
    plt.figlegend(legend_handles, legend_labels, loc=8, ncol=2)
    fig.tight_layout()
    return fig

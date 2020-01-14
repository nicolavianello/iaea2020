# import matplotlib.gridspec as gridspec
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import json
import seaborn as sns
import gpr1dfusion
from collections import OrderedDict
from scipy.interpolate import UnivariateSpline, interp1d
from scipy import constants
import bottleneck as bn

try:
    import MDSplus as mds
except:
    print("MDSplus not loaded")
try:
    import dd
except:
    print("dd library loaded")
try:
    from jetlib import (
        JETProbe,
        #        LiB as JETLiB,
        elm_detection,
        tespec,
        #        JETProfiles,
        bin_by,
    )
    from jet.data import sal
    from cherab.jet.equilibrium import JETEquilibrium
    from cherab.core.math import sample2d
except:
    warnings.warn("Jet library not loaded, JET figures can't be produced")
try:
    from tcv import (
        langmuir as tcvlangmuir,
        baratrons as tcvbaratrons,
        tcvgeom,
        gas as tcvgas,
    )
except:
    warnings.warn("TCV library not loaded, TCV figures can't be produced")
try:
    from aug import myThbObject, langmuir as auglangmuir, libes as AUGLiB, geomaug
except:
    warnings.warn("AUG library not loaded, AUG figures can't be produced")

mpl.rcParams["font.family"] = "sans-serif"
mpl.rc("font", size=22)
mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Tahoma"]})


def print_menu():
    print(30 * "-", "MENU", 30 * "-")
    print("1. Equilibria explored")
    print("2. Upstream profiles and target conditions")
    print("99: End")
    print(67 * "-")


loop = True

while loop:
    print_menu()
    selection = int(input("Enter your choice [1-99] "))
    if selection == 1:
        fig, ax = mpl.pylab.subplots(figsize=(13, 5), nrows=1, ncols=3)
        fig.subplots_adjust(bottom=0.15, wspace=0.25, right=0.98)
        # -----------------
        # JET
        # -----------------
        interior_contours = 10
        exterior_contours = 6
        # generate interior levels (evenly spaced between axis and boundary psi values)
        delta = 1.0 / (interior_contours + 2)
        interior_min = delta
        interior_max = 1.0 - delta
        interior_levels = np.linspace(interior_min, interior_max, interior_contours)
        # generate exterior levels
        exterior_levels = np.arange(
            1.0 + delta, 1.0 + (exterior_contours + 1) * delta, delta
        )

        shotList = (96543, 96543, 96542)
        tList = (48.65, 50.61, 49.66)
        colorList = ("#1EA8D1", "#4ABD22", "#D92534")
        for shotidx, (shot, t0, col) in enumerate(zip(shotList, tList, colorList)):
            Eq = JETEquilibrium(shot, user="jetppf", dda="EFTP")
            EqA = Eq.time(t0)
            limiter = np.append(
                EqA.limiter_polygon, [EqA.limiter_polygon[0, :]], axis=0
            )
            if shotidx == 0:
                ax[0].plot(limiter[:, 0], limiter[:, 1], "k", lw=2)
            #
            rmin, rmax = EqA.r_range
            zmin, zmax = EqA.z_range
            resolution = 0.01
            nr = round((rmax - rmin) / resolution)
            nz = round((zmax - zmin) / resolution)
            r, z, psi_sampled = sample2d(
                EqA.psi_normalised, (rmin, rmax, nr), (zmin, zmax, nz)
            )
            ax[0].contour(
                r,
                z,
                psi_sampled.transpose(),
                levels=interior_levels,
                colors=col,
                linewidths=0.6,
                linestyles="-",
            )
            ax[0].contour(
                r,
                z,
                psi_sampled.transpose(),
                levels=exterior_levels,
                colors=col,
                linewidths=0.5,
                linestyles="--",
            )
            ax[0].contour(
                r,
                z,
                psi_sampled.transpose(),
                levels=[1.0],
                colors=col,
                linewidths=1.0,
                linestyles="-",
            )
            ax[0].set_xlabel(r"R [m]")
            ax[0].set_ylabel(r"Z [m]")
            ax[0].set_aspect("equal")
            ax[0].autoscale(tight=True)
            ax[0].set_title(r"JET", fontsize=16)

        # -------
        # AUG
        # -------
        Eq = equilibrium.equilibrium(gfile="../data/AUG/g36574.3000")
        ax[1].plot(Eq.wall["R"], Eq.wall["Z"], "k", lw=2)
        ax[1].contour(
            Eq.R,
            Eq.Z,
            Eq.psiN,
            levels=interior_levels,
            colors="#1EA8D1",
            linewidths=0.6,
            linestyles="-",
        )
        ax[1].contour(
            Eq.R,
            Eq.Z,
            Eq.psiN,
            levels=exterior_levels,
            colors="#1EA8D1",
            linewidths=0.5,
            linestyles="--",
        )
        ax[1].contour(
            Eq.R,
            Eq.Z,
            Eq.psiN,
            levels=[1.0],
            colors="#1EA8D1",
            linewidths=1.0,
            linestyles="-",
        )
        ax[1].set_xlabel(r"R [m]")
        ax[1].set_ylabel(r"Z [m]")
        ax[1].set_aspect("equal")
        ax[1].autoscale(tight=True)
        ax[1].set_title(r"AUG", fontsize=16)

        # ----------
        # TCV
        # ----------
        Eq = equilibrium.equilibrium(gfile="../data/TCV/TCV64957_1.1.g")
        ax[2].plot(Eq.wall["R"], Eq.wall["Z"], "k", lw=2)
        ax[2].contour(
            Eq.R,
            Eq.Z,
            Eq.psiN,
            levels=interior_levels,
            colors="#1EA8D1",
            linewidths=0.6,
            linestyles="-",
        )
        ax[2].contour(
            Eq.R,
            Eq.Z,
            Eq.psiN,
            levels=exterior_levels,
            colors="#1EA8D1",
            linewidths=0.5,
            linestyles="--",
        )
        ax[2].contour(
            Eq.R,
            Eq.Z,
            Eq.psiN,
            levels=[1.0],
            colors="#1EA8D1",
            linewidths=1.0,
            linestyles="-",
        )
        ax[2].set_xlabel(r"R [m]")
        ax[2].set_ylabel(r"Z [m]")
        ax[2].set_aspect("equal")
        ax[2].autoscale(tight=True)
        ax[2].set_title(r"TCV", fontsize=16)
        fig.savefig("../pdfbox/AllEquilibria.pdf", bbox_to_inches="tight")

    elif selection == 2:
        pass

    elif selection == 99:
        loop = False
    else:
        input("Unknown Option Selected!")

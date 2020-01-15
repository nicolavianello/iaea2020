import warnings
import numpy as np
import matplotlib as mpl
import json
import seaborn as sns

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
    # from aug import myThbObject, langmuir as auglangmuir, libes as AUGLiB, geomaug
    from aug import myThbObject
except:
    warnings.warn("AUG library not loaded, AUG figures can't be produced")

mpl.rcParams["font.family"] = "sans-serif"
mpl.rc("font", size=22)
mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Tahoma"]})


def print_menu():
    print(30 * "-", "MENU", 30 * "-")
    print("1. Fig 1 synopsis:Equilibria explored")
    print("2. Fig 2 synopsis: Upstream profiles at two different recycling state")
    print("3. Fig 3 synopsis: Fluctuation studies")
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
        # create the figure
        fig, ax = mpl.pylab.subplots(figsize=(13, 5), nrows=1, ncols=3, sharey="row")
        fig.subplots_adjust(bottom=0.15, wspace=0.05, right=0.98)
        for _idax, (_ax, exp) in enumerate(zip(ax, ("JET", "AUG", "TCV"))):
            _ax.set_yscale("log")
            _ax.set_xlabel(r"$\rho$")
            _ax.set_ylim([0.05, 3])
            _ax.set_xlim([0.95, 1.15])
            if _idax != 0:
                _ax.tick_params(labelleft=False)
            else:
                _ax.set_ylabel(r"n$_e$/n$_e^{\rho=1}$")
            _ax.set_title(exp)

        # ---------------------
        # --- JET -------------
        # ---------------------
        shotList = (95504, 95641)
        colorList = ("#0568A6", "#F29F05")
        label = ("Low Rec", "High Rec")
        for _idxshot, (shot, col, lb) in enumerate(zip(shotList, colorList, label)):
            # determine the corresponding average Te
            jsonfile = "/Users/nicolavianello/Desktop/M18-41/Analysis/data/shot{}-VT5C.json".format(
                shot
            )
            Dictionary = {}
            with open(jsonfile, "r") as f:
                Dictionary = json.load(f)
            SpecTe = tespec.tespec(shot, configuration="VT5C", dtype="telo")
            t, y = SpecTe.selectradius()
            te = np.nanmean(
                y[np.logical_and(t >= Dictionary["tmin"], t <= Dictionary["tmax"])]
            )
            # now load the saved profiles
            File = "/Users/nicolavianello/Desktop/M18-41/Analysis/data/Shot{}_VT5C.npz".format(
                shot
            )
            Data = np.load(File)
            _norm = Data["enfit"][np.argmin(np.abs(Data["rhofit"] - 1))]
            ax[0].errorbar(
                Data["rho"],
                Data["ne"] / _norm,
                yerr=Data["dne"] / _norm,
                fmt="o",
                color=col,
                alpha=0.3,
                ms=5,
                errorevery=5,
            )
            ax[0].plot(Data["rhofit"], Data["enfit"] / _norm, "-", lw=2, color=col)
            ax[0].fill_between(
                Data["rhofit"],
                (Data["enfit"] - Data["denfit"]) / _norm,
                (Data["enfit"] + Data["denfit"]) / _norm,
                alpha=0.05,
                color=col,
            )
            ax[0].text(
                0.45, 0.9 - 0.1 * _idxshot, lb, transform=ax[0].transAxes, color=col
            )
            # ax[0].text(
            #     0.45,
            #     0.9 - 0.1 * _idxshot,
            #     r"$\langle$T$_e\rangle$ = {:.1f} eV".format(te),
            #     transform=ax[0].transAxes,
            #     color=col,
            # )

        # ---------------------
        # --- AUG -------------
        # ---------------------
        FileList = (
            "../data/AUG/ProfileShot34276_t2.25-2.35.npz",
            "../data/AUG/ProfileShot34276_t5.40-5.50.npz",
        )
        for _idxshot, (file, col, lb) in enumerate(zip(FileList, colorList, label)):
            Data = np.load(file)
            _norm = Data["rawEn"][np.argmin(np.abs(Data["rawRho"] - 1))]
            if _idxshot == 1:
                _e = Data["rawEn_err"] / 2.0
            else:
                _e = Data["rawEn_err"]
            ax[1].errorbar(
                Data["rawRho"],
                Data["rawEn"] / _norm,
                xerr=Data["rawRho_err"],
                yerr=_e / _norm,
                fmt="-o",
                color=col,
                alpha=0.5,
                ms=5,
            )
            ax[1].text(
                0.45, 0.9 - 0.1 * _idxshot, lb, transform=ax[1].transAxes, color=col
            )
            # ax[1].text(
            #     0.45,
            #     0.9 - 0.1 * _idxshot,
            #     r"$\langle$T$_e\rangle$ = {:.1f} eV".format(Data["teosp"]),
            #     transform=ax[1].transAxes,
            #     color=col,
            # )

        # ---------------------
        # --- TCV -------------
        # ---------------------
        FileList = (
            "../data/TCV/ProfileShot64494_t0.88-0.96.npz",
            "../data/TCV/ProfileShot64947_t1.50-1.75.npz",
        )
        for _idxshot, (file, col, lb) in enumerate(zip(FileList, colorList, label)):
            Data = np.load(file)
            _norm = Data["fitEn"][np.argmin(np.abs(Data["fitRho"] - 1))]
            ax[2].errorbar(
                Data["rawRho"],
                Data["rawEn"] / _norm,
                yerr=Data["rawEn_err"] / _norm,
                fmt="o",
                color=col,
                alpha=0.5,
                ms=5,
            )
            ax[2].plot(Data["fitRho"], Data["fitEn"] / _norm, "-", color=col, lw=2)
            ax[2].fill_between(
                Data["fitRho"],
                (Data["fitEn"] - Data["fitEn_err"]) / _norm,
                (Data["fitEn"] + Data["fitEn_err"]) / _norm,
                color=col,
                alpha=0.3,
            )

            ax[2].text(
                0.45, 0.9 - 0.1 * _idxshot, lb, transform=ax[2].transAxes, color=col
            )
            # ax[2].text(
            #     0.45,
            #     0.9 - 0.1 * _idxshot,
            #     r"$\langle$T$_e\rangle$ = {:.1f} eV".format(Data["teosp"]),
            #     transform=ax[2].transAxes,
            #     color=col,
            # )
        for _ax, l in zip(ax, ("(a)", "(b)", "(c)")):
            _ax.text(0.1, 0.1, l, transform=_ax.transAxes)

        fig.savefig("../pdfbox/AllUpstreamProfiles_synopsis.pdf")

    elif selection == 3:
        fig, ax = mpl.pylab.subplots(figsize=(13, 5), nrows=1, ncols=3)
        fig.subplots_adjust(bottom=0.15, wspace=0.25, right=0.98, left=0.08)
        colorList = ("#0568A6", "#F29F05")

        # ---------------------
        # --- JET -------------
        # ---------------------

        DictionaryShot = {
            "shotlist": (95502, 95642),
            "tlist": ([47.6, 48.6], [47.4, 48]),
            "color": ("#0568A6", "#F29F05"),
        }
        for _idxshot, (shot, col, trange, label) in enumerate(
            zip(
                DictionaryShot["shotlist"],
                DictionaryShot["color"],
                DictionaryShot["tlist"],
                ("Low rec", "High rec"),
            )
        ):
            Probe = JETProbe.limiterprobe(shot)
            num = 11
            rho = 0.5
            Probe.load_data(num)
            js, time, v = Probe.get_jsat(
                trange=trange, inter_elm=True, elm_range=[0.7, 0.9], rho=rho
            )
            jsN = (js - js.mean()) / js.std()
            sns.distplot(
                jsN,
                kde=True,
                hist=True,
                hist_kws={
                    "range": [jsN.min(), np.quantile(jsN, 0.98)],
                    "histtype": "step",
                },
                kde_kws={"clip": [jsN.min(), np.quantile(jsN, 0.98)]},
                color=col,
                ax=ax[0],
            )
            ax[0].text(
                0.6, 0.9 - _idxshot * 0.1, label, transform=ax[0].transAxes, color=col
            )
        ax[0].set_yscale("log")
        ax[0].set_xlim([-2.5, 4])
        ax[0].set_ylim([5e-2, 3])
        ax[0].set_xlabel(r"$\delta$J$_s$/$\sigma$")
        ax[0].set_title("JET")

        # ---------------------
        # --- TCV -------------
        # ---------------------
        shot = 64948
        Data = np.load(
            "/Users/nicolavianello/Dropbox/Work/Collaborations/ITPA/28th-Div-SOL/Data/TCV/jsatFluctuation64948.npz"
        )
        jsWall = Data["jsWall"]
        trange = ((0.9, 1.1), (1.65, 1.9))
        t = Data["t"]
        DataD = np.load("../data/TCV/DalphaShot64948.npz")
        Dalpha = DataD["y"]
        tDalpha = DataD["t"]
        for _idxshot, (tr, col, label) in enumerate(
            zip(trange, colorList, ("Low rec", "High rec"))
        ):
            _idx = np.where((t >= tr[0]) & (t <= tr[1]))[0]
            # ---- WALL ----
            # (jsWall[_idx] - jsWall[_idx].mean()) / (jsWall[_idx].std())
            jsN = jsWall[_idx]
            tN = t[_idx]
            _idxDalpha = np.where((tDalpha >= tr[0]) & (tDalpha <= tr[1]))[0]
            _x, _y = tDalpha[_idxDalpha], Dalpha[_idxDalpha]
            ED = elm_detection.elm_detection(_x, _y, rho=0.82, width=0.01)
            ED.run()
            mask = ED.filter_signal(tN, inter_elm_range=[0.5, 0.9])
            jsN = jsN[mask]
            jsN = (jsN - jsN.mean()) / (jsN.std())
            sns.distplot(
                jsN,
                kde=True,
                hist=True,
                hist_kws={
                    "range": [jsN.min(), np.quantile(jsN, 0.99)],
                    "histtype": "step",
                },
                kde_kws={"clip": [jsN.min(), np.quantile(jsN, 0.99)]},
                color=col,
                ax=ax[2],
            )
            ax[2].text(
                0.6, 0.9 - _idxshot * 0.1, label, transform=ax[2].transAxes, color=col
            )
        ax[2].set_yscale("log")
        ax[2].set_xlim([-2.5, 4])
        ax[2].set_ylim([6e-3, 1])
        ax[2].set_xlabel(r"$\delta$J$_s$/$\sigma$")
        ax[2].set_title("TCV")
        ax[2].legend(loc="best", frameon=False, numpoints=1)
        # ---------------------
        # --- AUG -------------
        # ---------------------
        Data = np.load("../data/AUG/fluctuationTHB36574.npz")
        # np.savez(Filesave, time=time, blob_freq=blob_freq_trace, velocity=Velocity, amplitude=amplitude,
        #          peakstart=np.arra
        #          ...: y(Data.peak_start_time)[_refChannel])

        peakStart = Data["peakstart"]
        _idxA = np.where(peakStart <= 3)[0]
        _idxB = np.where(peakStart >= 5.5)[0]
        Velocity = Data["velocity"]
        Amplitude = Data["amplitude"]
        for _idxshot, (_i, _l, col) in enumerate(
            zip((_idxA, _idxB), ("Low rec", "High rec"), colorList)
        ):
            _y = Velocity[_i]
            _y = _y[np.logical_and(_y > 0, _y < 8)]

            sns.distplot(
                _y,
                kde=True,
                hist=True,
                hist_kws={"range": [0, 8], "histtype": "step", "label": _l,},
                kde_kws={"clip": [0, 8]},
                color=col,
                ax=ax[1],
            )
            ax[1].text(
                0.6, 0.9 - _idxshot * 0.1, _l, transform=ax[1].transAxes, color=col
            )
        ax[1].set_xlabel(r"v$_r$ [km/s]")
        ax[1].set_title("AUG")
        for _ax, l in zip(ax, ("(a)", "(b)", "(c)")):
            _ax.text(0.1, 0.9, l, transform=_ax.transAxes)
        fig.savefig("../pdfbox/SynopsisFluctuationCombined.pdf", bbox_to_inches="tight")

    elif selection == 99:
        loop = False
    else:
        input("Unknown Option Selected!")

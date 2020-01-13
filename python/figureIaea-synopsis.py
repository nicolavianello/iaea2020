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
except ImportWarning:
    print("MDSplus not loaded")
try:
    import dd
except ImportWarning:
    print("dd library loaded")
try:
    from jetlib import (
        JETProbe,
        LiB as JETLiB,
        elm_detection,
        equilibrium as JETequilibrium,
        tespec,
        JETProfiles,
        bin_by,
    )
except ImportWarning:
    warning.warns("Jet library not loaded")
try:
    from tcv import (
        langmuir as tcvlangmuir,
        baratrons as tcvbaratrons,
        tcvgeom,
        gas as tcvgas,
    )
except ImportWarning:
    warnings.warn("TCV library not loaded")
try:
    from aug import myThbObject, langmuir as auglangmuir, libes as AUGLiB, geomaug
except ImportWarning:
    warnings.warn("AUG library not loaded")

mpl.rcParams["font.family"] = "sans-serif"
mpl.rc("font", size=22)
mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Tahoma"]})


def print_menu():
    print(30 * "-", "MENU", 30 * "-")
    print("1. AUG power scan behaviour")
    print("2. AUG power scan target behaviour")
    print("3. THB analysis")
    print("4. General plot for shot 34276, 36574, 36605")
    print("5. General profiles for shot 34276, 36574, 36605")
    print("6. Example and zooms of ELM behavior for shot 36574")
    print("7. Compute the profile of Lambda for shot 34276, 36574, 36605")
    print("8. Blob frequency in far SOL and neutrals shot 36574")
    print("99: End")
    print(67 * "-")


loop = True

while loop:
    print_menu()
    selection = int(input("Enter your choice [1-99] "))
    if selection == 1:
        shotList = (36342, 36343, 36345, 36346)
        colorList = ("#BC1AF0", "#324D5C", "#F29E38", "#FF3B30")
        lineStyle = (":", "-.", "--", "-")
        thrListAll = (
            (1200, 900, 700, 700),
            (1200, 700, 700, 700),
            (1000, 700, 700, 500),
            (1200, 800, 800, 600),
        )
        trList = ((2.8, 2.9), (5.0, 5.1), (5.8, 5.9), (6.85, 6.95))
        try:
            c = mds.Connection("localhost:8001")
            _hasMds = True
        except:
            _hasMds = False
            print("Using dd local library")
        OuterTarget = OrderedDict(
            [
                ("ua1", {"R": 1.582, "z": -1.199, "s": 1.045}),
                ("ua2", {"R": 1.588, "z": -1.175, "s": 1.070}),
                ("ua3", {"R": 1.595, "z": -1.151, "s": 1.094}),
                ("ua4", {"R": 1.601, "z": -1.127, "s": 1.126}),
                ("ua5", {"R": 1.608, "z": -1.103, "s": 1.158}),
                ("ua6", {"R": 1.614, "z": -1.078, "s": 1.189}),
                ("ua7", {"R": 1.620, "z": -1.054, "s": 1.213}),
                ("ua8", {"R": 1.627, "z": -1.030, "s": 1.246}),
                ("ua9", {"R": 1.640, "z": -0.982, "s": 1.276}),
            ]
        )

        # build a figure with only the upstream profiles
        # with color coding according to shots and for 4
        # different timing
        figProfiles, axProfiles = mpl.pylab.subplots(
            figsize=(12, 3), nrows=1, ncols=4, sharey="row"
        )
        figProfiles.subplots_adjust(
            left=0.13, right=0.98, wspace=0.05, bottom=0.25, top=0.85
        )
        for _i, (ax, tr) in enumerate(zip(axProfiles, trList)):
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.set_xlabel(r"$\rho$")
            ax.set_title(r"t = {:.2f} s".format(np.average(tr)))
            ax.set_xlim([0.98, 1.11])
            ax.set_ylim([9e-2, 2])
            ax.yaxis.set_major_locator(
                mpl.ticker.LogLocator(
                    base=10.0, numticks=15, subs=(1, 2, 3, 4, 5, 6, 7, 8, 9)
                )
            )
            ax.set_yscale("log")
            if _i != 0:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(r"n$_e$/n$_e^{\rho =1}$")

        # fig profiles for each shot
        figProfilesShot, axProfilesShot = mpl.pylab.subplots(
            figsize=(12, 3), nrows=1, ncols=4, sharey="row"
        )
        figProfilesShot.subplots_adjust(
            left=0.13, right=0.98, wspace=0.05, bottom=0.25, top=0.85
        )
        for _i, (ax, tr) in enumerate(zip(axProfilesShot, shotList)):
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.set_xlabel(r"$\rho$")
            ax.set_title(r"# {}".format(tr))
            ax.set_xlim([0.98, 1.11])
            ax.set_ylim([9e-2, 2])
            ax.yaxis.set_major_locator(
                mpl.ticker.LogLocator(
                    base=10.0, numticks=15, subs=(1, 2, 3, 4, 5, 6, 7, 8, 9)
                )
            )
            ax.set_yscale("log")
            if _i != 0:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(r"n$_e$/n$_e^{\rho =1}$")

        # build the figure with NBI, fueling and total ion flux
        # build the figure
        figTime, axTime = mpl.pylab.subplots(
            figsize=(5, 7), nrows=3, ncols=1, sharex="col"
        )
        figTime.subplots_adjust(left=0.2, right=0.98, hspace=0.05, bottom=0.1, top=0.95)
        label = (
            r"P$_{\mathrm{NBI}}$ [MW]",
            r"[10$^{23}$molecules/m$^{2}$/s]",
            r"ion flux [10$^{23}$ /s]",
        )

        for _i, (ax, lb) in enumerate(zip(axTime, label)):
            if _i != 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"t [s]")
            ax.text(0.05, 0.85, lb, transform=ax.transAxes)
            ax.set_xlim([0, 8])
        # now the cycle for the plot
        for shotid, (shot, _c, _lst, thrList, _axP) in enumerate(
            zip(shotList, colorList, lineStyle, thrListAll, axProfilesShot)
        ):
            # read the NBI and plot it
            if _hasMds:
                _s = 'augsignal({},"NIS","PNI")'.format(shot)
                Data = c.get(_s).data()
                _st = 'dim_of(augsignal({},"NIS","PNI"))'.format(shot)
                axTime[0].plot(
                    c.get(_st).data(),
                    Data / 1e6,
                    _lst,
                    color=_c,
                    alpha=0.7,
                    label=r"# {}".format(shot),
                    lw=2.5,
                )
            else:
                Data = dd.shotfile("NIS", shot)("PNI")
                axTime[0].plot(
                    Data.time,
                    Data.data / 1e6,
                    _lst,
                    color=_c,
                    alpha=0.7,
                    label=r"# {}".format(shot),
                    lw=2.5,
                )

            # read the Pressure and plot it
            if _hasMds:
                _s = 'augsignal({},"IOC","F01")'.format(shot)
                Data = c.get(_s).data() / 1e21
                _st = 'dim_of(augsignal({},"IOC","F01"))'.format(shot)
                axTime[1].plot(
                    c.get(_st).data(),
                    Data,
                    _lst,
                    color=_c,
                    alpha=0.7,
                    label=r"# {}".format(shot),
                    lw=2.5,
                )
            else:
                Data = dd.shotfile("IOC", shot)("F01")
                axTime[1].plot(
                    Data.time,
                    Data.data / 1e23,
                    _lst,
                    color=_c,
                    alpha=0.7,
                    label=r"# {}".format(shot),
                    lw=2.5,
                )
            # now compute the ion flux to the target
            outDivSignal = np.asarray([])
            if _hasMds:
                neTime = c.get(
                    'dim_of(augsignal({}, "LSD", "ne-{}"))'.format(shot, "ua1")
                ).data()
                for i, s in enumerate(OuterTarget.keys()):
                    _ne = c.get('augsignal({}, "LSD", "ne-{}")'.format(shot, s)).data()
                    _te = c.get('augsignal({}, "LSD", "te-{}")'.format(shot, s)).data()
                    _an = c.get('augsignal({}, "LSD", "ang-{}")'.format(shot, s)).data()
                    _cs = np.sqrt(constants.e * 4 * _te / (2 * constants.m_p))
                    # this is the ion flux
                    _s = _ne * _cs * np.abs(np.sin(np.radians(_an)))

                    if i == 0:
                        outDivSignal = _s
                    else:
                        outDivSignal = np.vstack((outDivSignal, _s))
            else:
                neTime = dd.shotfile("LSD", shot)("ne-ua1").time
                for i, s in enumerate(OuterTarget.keys()):
                    _ne = dd.shotfile("LSD", shot)("ne-{}".format(s)).data
                    _te = dd.shotfile("LSD", shot)("te-{}".format(s)).data
                    _an = dd.shotfile("LSD", shot)("ang-{}".format(s)).data
                    _cs = np.sqrt(constants.e * 4 * _te / (2 * constants.m_p))
                    # this is the ion flux
                    _s = _ne * _cs * np.abs(np.sin(np.radians(_an)))

                    if i == 0:
                        outDivSignal = _s
                    else:
                        outDivSignal = np.vstack((outDivSignal, _s))

            outTarget = np.zeros(outDivSignal.shape[1])
            for i in range(neTime.size):
                _x = np.asarray([OuterTarget[k]["s"] for k in OuterTarget.keys()])
                _r = np.asarray([OuterTarget[k]["R"] for k in OuterTarget.keys()])
                _y = outDivSignal[:, i]
                _dummy = np.vstack((_x, _y)).transpose()
                _dummy = _dummy[~np.isnan(_dummy).any(1)]
                _x = _dummy[:, 0]
                _y = _dummy[:, 1][np.argsort(_x)]
                _x = np.sort(_x)
                outTarget[i] = 2 * np.pi * _r.mean() * np.trapz(_y, x=_x)
            outTarget = bn.move_mean(outTarget, 600) / 1e23
            axTime[2].plot(neTime, outTarget, "-", color=_c, alpha=0.8)
            # this is for the plot of the profiles
            # collect also the data of LiBes and Ipol
            if _hasMds:
                LiBes = c.get('augsignal({}, "LIN", "ne")'.format(shot)).data() / 1e19
                LiRho = c.get('dim_of(augsignal({},"LIN","ne"),1)'.format(shot)).data()
                LiTime = c.get('dim_of(augsignal({},"LIN","ne"))'.format(shot)).data()
                # read the IpolSolA per il masking del
                Ipol = c.get('augsignal({}, "MAC", "Ipolsola")'.format(shot)).data()
                IpolTime = c.get(
                    'dim_of(augsignal({}, "MAC", "Ipolsola"))'.format(shot)
                ).data()
            else:
                LiBes = dd.shotfile("LIN", shot)("ne").data.transpose() / 1e19
                LiRho = dd.shotfile("LIN", shot)("ne").area.data.transpose()
                LiTime = dd.shotfile("LIN", shot)("ne").time
                Ipol = dd.shotfile("MAC", shot)("Ipolsola").data
                IpolTime = dd.shotfile("MAC", shot)("Ipolsola").time

            for i, (tr, _thr, _ax, _cp) in enumerate(
                zip(
                    trList,
                    thrList,
                    axProfiles,
                    ("#3E9DD3", "#9A8D84", "#2BFF00", "#3B4859"),
                )
            ):
                axTime[0].axvline(np.average(tr), ls="-", lw=2, color=_cp)
                axTime[1].axvline(np.average(tr), ls="-", lw=2, color=_cp)
                axTime[2].axvline(np.average(tr), ls="-", lw=2, color=_cp)

                _idxT = np.where(((IpolTime >= tr[0]) & (IpolTime <= tr[1])))[0]
                # now create an appropriate savgolfile
                IpolS = savgol_filter(Ipol[_idxT], 301, 3)
                IpolT = IpolTime[_idxT]
                IpolO = Ipol[_idxT]
                # we generate an UnivariateSpline object
                _dummyTime = LiTime[np.where((LiTime >= tr[0]) & (LiTime <= tr[1]))[0]]
                _dummyLib = LiBes[:, np.where((LiTime >= tr[0]) & (LiTime <= tr[1]))[0]]
                _dummyRho = LiRho[:, np.where((LiTime >= tr[0]) & (LiTime <= tr[1]))[0]]
                IpolSp = UnivariateSpline(IpolT, IpolS, s=0)(_dummyTime)
                # on these we choose a threshold
                # which can be set as also set as keyword
                ElmMask = np.zeros(IpolSp.size, dtype="bool")
                ElmMask[np.where(IpolSp > _thr)[0]] = True
                _interElm = np.where(ElmMask == False)[0]
                # in this way they are
                _dummyLib = _dummyLib[:, _interElm].ravel()
                _dummyRho = _dummyRho[:, _interElm].ravel()
                yOut, bins, bin_means, bin_width, xOut = bin_by.bin_by(
                    _dummyRho, _dummyLib, nbins=40
                )
                xB = np.asarray([np.nanmean(k) for k in xOut])
                xBE = np.asarray([np.nanstd(k) for k in xOut])
                yB = np.asarray([np.nanmean(k) for k in yOut])
                yBE = np.asarray([np.nanstd(k) for k in yOut])
                enS = yB[np.argmin(np.abs(xB - 1))]
                _ax.errorbar(
                    xB,
                    yB / enS,
                    xerr=xBE,
                    yerr=yBE / enS,
                    fmt="-",
                    color=_c,
                    alpha=0.7,
                    lw=2.5,
                )
                _axP.errorbar(
                    xB,
                    yB / enS,
                    xerr=xBE,
                    yerr=yBE / enS,
                    fmt="-",
                    color=_cp,
                    alpha=0.7,
                    lw=2.5,
                )

        figTime.savefig("../pdfbox/AUG_PowerScanTime.pdf", bbox_to_inches="tight")
        figProfiles.savefig(
            "../pdfbox/AUG_PowerScanProfile.pdf", bbox_to_inches="tight"
        )
        figProfilesShot.savefig(
            "../pdfbox/AUG_PowerScanProfileEachShot.pdf", bbox_to_inches="tight"
        )

    elif selection == 2:
        shotList = (36342, 36343, 36345, 36346)
        colorList = ("#BC1AF0", "#324D5C", "#F29E38", "#FF3B30")
        lineStyle = (":", "-.", "--", "-")
        thrListAll = (
            (1200, 900, 700, 700),
            (1200, 700, 700, 700),
            (1000, 700, 700, 500),
            (1200, 800, 800, 600),
        )
        trList = ((2.8, 2.9), (5.0, 5.1), (5.8, 5.9), (6.85, 6.95))
        # fig profiles for each shot it also save the npz data to
        # be further collected
        figProfiles, axProfiles = mpl.pylab.subplots(
            figsize=(12, 8), nrows=2, ncols=4, sharey="row", sharex="col"
        )
        figProfiles.subplots_adjust(
            left=0.13, right=0.98, wspace=0.05, bottom=0.18, top=0.9
        )

        # fig profiles for each shot
        figProfilesShot, axProfilesShot = mpl.pylab.subplots(
            figsize=(12, 8), nrows=2, ncols=4, sharey="row", sharex="col"
        )
        figProfilesShot.subplots_adjust(
            left=0.13, right=0.98, wspace=0.05, bottom=0.18, top=0.9
        )

        axProfiles[0, 0].set_ylabel(r"n$_e [10^{19}$m$^{-3}$]")
        axProfiles[1, 0].set_ylabel(r"T$_e$ [eV]")
        axProfilesShot[0, 0].set_ylabel(r"n$_e [10^{19}$m$^{-3}$]")
        axProfilesShot[1, 0].set_ylabel(r"T$_e$ [eV]")

        for i, (shot, tr) in enumerate(zip(shotList, trList)):
            if i != 0:
                axProfiles[0, i].tick_params(labelleft=False, labelbottom=False)
                axProfilesShot[0, i].tick_params(labelleft=False, labelbottom=False)
                axProfiles[1, i].tick_params(labelleft=False)
                axProfilesShot[1, i].tick_params(labelleft=False)
            axProfiles[1, i].set_xlabel(r"$\rho$")
            axProfilesShot[1, i].set_xlabel(r"$\rho$")
            axProfiles[1, i].set_xlim([0.95, 1.1])
            axProfilesShot[1, i].set_xlim([0.95, 1.1])
            axProfilesShot[0, i].set_title(r"#{}".format(shot))
            axProfiles[0, i].set_title(r"t={:2f}".format(np.average(tr)))

        # now the cycle for the plot
        for shotid, (shot, _c, thrList) in enumerate(
            zip(shotList, colorList, thrListAll)
        ):
            Target = langmuir.Target(shot)
            for i, (tr, _thr, _cp) in enumerate(
                zip(trList, thrList, ("#3E9DD3", "#9A8D84", "#2BFF00", "#3B4859"))
            ):
                rhoNe, Ne, errNe = Target.PlotEnProfile(
                    trange=tr, interelm=True, threshold=_thr, Plot=False
                )
                rhoTe, Te, errTe = Target.PlotTeProfile(
                    trange=tr, interelm=True, threshold=_thr, Plot=False
                )
                axProfilesShot[0, shotid].errorbar(
                    rhoNe,
                    Ne / 1e19,
                    yerr=errNe / 1e19,
                    fmt="o",
                    ms=10,
                    color=_cp,
                    alpha=0.5,
                )
                axProfilesShot[1, shotid].errorbar(
                    rhoTe, Te, yerr=errTe, fmt="o", ms=10, color=_cp, alpha=0.5
                )

                axProfiles[0, i].errorbar(
                    rhoNe,
                    Ne / 1e19,
                    yerr=errNe / 1e19,
                    fmt="o",
                    ms=10,
                    color=_c,
                    alpha=0.5,
                )
                axProfiles[1, i].errorbar(
                    rhoTe, Te, yerr=errTe, fmt="o", ms=10, color=_c, alpha=0.5
                )
        figProfiles.savefig(
            "../pdfbox/AUG_PowerScanTargetProfile.pdf", bbox_to_inches="tight"
        )
        figProfilesShot.savefig(
            "../pdfbox/AUG_PowerScanTargetProfileEachShot.pdf", bbox_to_inches="tight"
        )
    elif selection == 3:
        # plot a comparison of the distribution of the amplitude and of the radial
        # velocities for t<= 3 and t >= 5.5 s
        shot = 36574
        File = "../Data/AUG/36574_t_2.000_7.500_PMT1"
        Data = myThbObject.myThbObject(File)
        _refChannel = 22  # channel 22
        peakStart = np.array(Data.peak_start_time)[_refChannel]
        _idxA = np.where(peakStart <= 3)[0]
        _idxB = np.where(peakStart >= 5.5)[0]
        Velocity = np.array(Data.velocity_all) / 1e3
        Amplitude = np.array(Data.fil_amplitude)[_refChannel]
        colorL = ("#2679B1", "#F2811D")
        fig, ax = mpl.pylab.subplots(figsize=(6, 8), nrows=2, ncols=1)
        fig.subplots_adjust(hspace=0.3, left=0.2, bottom=0.12, top=0.96, right=0.95)
        fig2, ax2 = mpl.pylab.subplots(figsize=(6, 4), nrows=1, ncols=1)
        fig2.subplots_adjust(hspace=0.3, left=0.2, bottom=0.2, top=0.96, right=0.95)
        for _i, _l, _col in zip(
            (_idxA, _idxB), ("Type-I inter-ELM", "small-ELM"), colorL
        ):
            _y = Amplitude[_i]
            _y = _y[_y > 1]
            sns.distplot(
                _y[~np.isnan(_y)], bins=None, color=_col, kde=True, ax=ax[0], label=_l
            )
            _y = Velocity[_i]
            _y = _y[np.logical_and(_y > 0, _y < 8)]
            sns.distplot(
                _y[~np.isnan(_y)], bins=None, color=_col, kde=True, ax=ax[1], label=_l
            )
            sns.distplot(
                _y[~np.isnan(_y)], bins=None, color=_col, kde=True, ax=ax2, label=_l
            )

        ax[0].set_xlabel("Amplitude [a.u.]")
        ax[1].set_xlabel(r"v$_r$ [km/s]")
        ax[0].legend(loc="best", numpoints=1, frameon=False, fontsize=14)
        ax[1].legend(loc="best", numpoints=1, frameon=False, fontsize=14)
        ax2.legend(loc="best", numpoints=1, frameon=False, fontsize=14)

        ax2.set_xlabel(r"v$_r$ [km/s]")
        fig.savefig(
            "../pdfbox/amplitudeVelocityBlobShot36574.pdf", bbox_to_inches="tight"
        )
        fig2.savefig("../pdfbox/VelocityBlobShot36574.pdf", bbox_to_inches="tight")

    elif selection == 4:
        shotList = (34276, 36574, 36605)
        stList = {
            "34276": {
                "tlist": ((2.245, 2.345), (3.0, 3.1), (4.63, 4.73), (5.4, 5.5)),
                "thresh": [-99.0, 1200, 500, 500],
            },
            "36574": {
                "tlist": ((1.9, 2.1), (3.9, 4.1), (4.9, 5.1), (6.15, 6.35)),
                "thresh": (3000, 1000, 1000, -99),
            },
            "36605": {
                "tlist": ((2.1, 2.2), (4.1, 4.2), (4.9, 5.0), (6.25, 6.35)),
                "thresh": (3000, 1000, 1000, 1000),
            },
        }
        colorList = ("#2679B1", "#BC2764", "#00805E", "#F2811D")
        try:
            c = mds.Connection("localhost:8001")
            _hasMds = True
        except:
            _hasMds = False
            print("Using dd local library")
        for k in stList.keys():
            shot = int(k)
            tList = stList[k]["tlist"]
            thresL = stList[k]["thresh"]
            # this is the plot for time traces
            figTime, axTime = mpl.pylab.subplots(
                figsize=(8, 11), nrows=4, ncols=1, sharex="col"
            )
            figTime.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.15)
            # Hide shared x-tick labels
            for ax in axTime[:-1]:
                ax.tick_params(labelbottom=False)
                ax.set_xlim([0, 8])

            if _hasMds:
                _s = 'augsignal({},"NIS","PNI")'.format(shot)
                Data = c.get(_s).data()
                _st = 'dim_of(augsignal({},"NIS","PNI"))'.format(shot)
                _sEc = 'augsignal({},"ECS","PECRH")'.format(shot)
                _sEct = 'dim_of(augsignal({},"ECS","PECRH"))'.format(shot)
                _SpEcr = interp1d(
                    c.get(_sEct).data(), c.get(_sEc).data(), fill_value="extrapolate"
                )
                axTime[0].plot(
                    c.get(_st).data(),
                    (Data + _SpEcr(c.get(_st).data())) / 1e6,
                    color="k",
                    lw=2.5,
                )
            else:
                Data = dd.shotfile("NIS", shot)("PNI")
                _sEc = dd.shotfile("ECS", shot)("PECRH")
                _SpEcr = interp1d(_sEc.time, _sEc.data, fill_value="extrapolate")
                axTime[0].plot(
                    Data.time, (Data.data + _SpEcr(Data.time)) / 1e6, color="k", lw=2.5
                )

            axTime[0].set_ylabel(r"P$_{\mathrm{heat}}$ [MW]")
            axTime[0].set_title(r"#{}".format(shot))
            # now the divertor pressure
            if _hasMds:
                _s = 'augsignal({},"IOC","F01")'.format(shot)
                Data = c.get(_s).data() / 1e23
                _st = 'dim_of(augsignal({},"IOC","F01"))'.format(shot)
                axTime[1].plot(
                    c.get(_st).data(), bn.move_mean(Data, 100), color="k", lw=2.5
                )
            else:
                _s = dd.shotfile("IOC", shot)("F01")
                axTime[1].plot(
                    _s.time, bn.move_mean(_s.data / 1e23, 100), color="k", lw=2.5
                )

            axTime[1].text(
                0.05,
                0.83,
                r"[10$^{23}$molecules/m$^{2}$/s]",
                transform=axTime[1].transAxes,
                fontsize=20,
            )

            # also the fueling
            if _hasMds:
                Data = c.get('augsignal({}, "UVS", "D_tot")'.format(shot)).data() / 1e23
                _st = 'dim_of(augsignal({}, "UVS", "D_tot")'.format(shot) + ")"
                axTime[1].plot(c.get(_st).data(), Data, color="k", lw=2.5)
            else:
                _s = dd.shotfile("UVS", shot)("D_tot")
                axTime[2].plot(_s.time, _s.data / 1e23, color="k", lw=2.5)
            if shot == 36574:
                axTime[2].set_ylim([0, 1.2])
            axTime[2].text(
                0.05,
                0.83,
                r"D$_2$ [10$^{23}$el/s]",
                transform=axTime[2].transAxes,
                fontsize=20,
            )

            # now compute the target ion flux
            OuterTarget = OrderedDict(
                [
                    ("ua1", {"R": 1.582, "z": -1.199, "s": 1.045}),
                    ("ua2", {"R": 1.588, "z": -1.175, "s": 1.070}),
                    ("ua3", {"R": 1.595, "z": -1.151, "s": 1.094}),
                    ("ua4", {"R": 1.601, "z": -1.127, "s": 1.126}),
                    ("ua5", {"R": 1.608, "z": -1.103, "s": 1.158}),
                    ("ua6", {"R": 1.614, "z": -1.078, "s": 1.189}),
                    ("ua7", {"R": 1.620, "z": -1.054, "s": 1.213}),
                    ("ua8", {"R": 1.627, "z": -1.030, "s": 1.246}),
                    ("ua9", {"R": 1.640, "z": -0.982, "s": 1.276}),
                ]
            )
            outDivSignal = np.asarray([])
            if _hasMds:
                neTime = c.get(
                    'dim_of(augsignal({}, "LSD", "ne-{}"))'.format(shot, "ua1")
                ).data()
                for i, s in enumerate(OuterTarget.keys()):
                    _ne = c.get('augsignal({}, "LSD", "ne-{}")'.format(shot, s)).data()
                    _te = c.get('augsignal({}, "LSD", "te-{}")'.format(shot, s)).data()
                    _an = c.get('augsignal({}, "LSD", "ang-{}")'.format(shot, s)).data()
                    _cs = np.sqrt(constants.e * 4 * _te / (2 * constants.m_p))
                    # this is the ion flux
                    _s = _ne * _cs * np.abs(np.sin(np.radians(_an)))

                    if i == 0:
                        outDivSignal = _s
                    else:
                        outDivSignal = np.vstack((outDivSignal, _s))
            else:
                neTime = dd.shotfile("LSD", shot)("ne-ua1").time
                for i, s in enumerate(OuterTarget.keys()):
                    _ne = dd.shotfile("LSD", shot)("ne-{}".format(s)).data
                    _te = dd.shotfile("LSD", shot)("te-{}".format(s)).data
                    _an = dd.shotfile("LSD", shot)("ang-{}".format(s)).data
                    _cs = np.sqrt(constants.e * 4 * _te / (2 * constants.m_p))
                    # this is the ion flux
                    _s = _ne * _cs * np.abs(np.sin(np.radians(_an)))

                    if i == 0:
                        outDivSignal = _s
                    else:
                        outDivSignal = np.vstack((outDivSignal, _s))
            # now we compute the total integrate ion flux
            outTarget = np.zeros(neTime.size)
            for i in range(neTime.size):
                _x = np.asarray([OuterTarget[k]["s"] for k in OuterTarget.keys()])
                _r = np.asarray([OuterTarget[k]["R"] for k in OuterTarget.keys()])
                _y = outDivSignal[:, i]
                _dummy = np.vstack((_x, _y)).transpose()
                _dummy = _dummy[~np.isnan(_dummy).any(1)]
                _x = _dummy[:, 0]
                _y = _dummy[:, 1][np.argsort(_x)]
                _x = np.sort(_x)
                outTarget[i] = 2 * np.pi * _r.mean() * np.trapz(_y, x=_x)
            outerTarget = bn.move_mean(outTarget, 300)
            axTime[3].plot(neTime, outerTarget / 1e23, lw=2.5, color="k")
            axTime[3].set_xlabel(r"t [s]")
            axTime[3].set_ylabel(r"$[10^{23}$ion/s$]$")
            for ax in axTime:
                for t, _c in zip(tList, colorList):
                    ax.axvline(np.mean(t), ls="-", lw=2, color=_c)
            figTime.savefig(
                "../pdfbox/GeneralTimeShot{}.pdf".format(shot), bbox_to_inches="tight"
            )

    elif selection == 5:
        shotList = (34276, 36574, 36605)
        stList = {
            "34276": {
                "tlist": ((2.245, 2.345), (3.0, 3.1), (4.63, 4.73), (5.4, 5.5)),
                "thresh": [-99.0, 1200, 500, 500],
            },
            "36574": {
                "tlist": ((1.9, 2.1), (3.9, 4.1), (4.9, 5.1), (6.15, 6.35)),
                "thresh": (3000, 1000, 1000, -99),
            },
            "36605": {
                "tlist": ((2.1, 2.2), (4.1, 4.2), (4.9, 5.0), (6.25, 6.35)),
                "thresh": (3000, 1000, 1000, 1000),
            },
        }
        colorList = ("#2679B1", "#BC2764", "#00805E", "#F2811D")
        try:
            c = mds.Connection("localhost:8001")
            _hasMds = True
        except:
            _hasMds = False
            print("Using dd local library")
        for k in stList.keys():
            shot = int(k)
            tList = stList[k]["tlist"]
            thresL = stList[k]["thresh"]
            # this is the plot with upstream and downstream profile
            figProfile = mpl.pylab.figure(figsize=(11, 9), tight_layout=True)
            gs_top = mpl.pylab.GridSpec(2, 2, wspace=0.3, top=0.95)
            # Bottom axes with the same x and y axis
            upAxes = figProfile.add_subplot(gs_top[0, :])
            upAxes.yaxis.set_major_locator(
                mpl.ticker.LogLocator(
                    base=10.0, numticks=15, subs=(1, 2, 3, 4, 5, 6, 7, 8, 9)
                )
            )
            upAxes.set_xlabel(r"$\rho$")
            upAxes.set_ylim([3e-2, 4])
            upAxes.set_xlim([0.98, 1.1])
            upAxes.set_ylabel(r"n$_e$/n$_e^{\rho=1}$")
            upAxes.set_yscale("log")
            # The shared in time axes
            axNe = figProfile.add_subplot(gs_top[1, 0])
            axTe = figProfile.add_subplot(gs_top[1, 1])
            # Hide shared x-tick labels
            for ax in (axNe, axTe):
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.set_xlim([0.98, 1.06])
                ax.set_xlabel(r"$\rho$")
            axNe.set_ylabel(r"n$_e [10^{19}$m$^{-3}$]")
            axTe.set_ylabel(r"T$_e$ [eV]")

            # read the Li-Beam data
            if _hasMds:
                LiBes = c.get('augsignal({}, "LIN", "ne")'.format(shot)).data() / 1e19
                LiRho = c.get('dim_of(augsignal({},"LIN","ne"),1)'.format(shot)).data()
                LiTime = c.get('dim_of(augsignal({},"LIN","ne"))'.format(shot)).data()
                # read the IpolSolA per il masking del
                # LiB
                Ipol = c.get('augsignal({}, "MAC", "Ipolsola")'.format(shot)).data()
                IpolTime = c.get(
                    'dim_of(augsignal({}, "MAC", "Ipolsola"))'.format(shot)
                ).data()
            else:
                LiBes = dd.shotfile("LIN", shot)("ne").data.transpose() / 1e19
                LiRho = dd.shotfile("LIN", shot)("ne").area.data.transpose()
                LiTime = dd.shotfile("LIN", shot)("ne").time
                Ipol = dd.shotfile("MAC", shot)("Ipolsola").data
                IpolTime = dd.shotfile("MAC", shot)("Ipolsola").time

            Target = langmuir.Target(shot)
            # now the profiles which is more subtle and long
            for _idx, (_tr, _thr, _col) in enumerate(zip(tList, thresL, colorList)):
                if _thr != -99:
                    _idxT = np.where(((IpolTime >= _tr[0]) & (IpolTime <= _tr[1])))[0]
                    # now create an appropriate savgolfile
                    IpolS = savgol_filter(Ipol[_idxT], 301, 3)
                    IpolT = IpolTime[_idxT]
                    IpolO = Ipol[_idxT]
                    _dummyTime = LiTime[
                        np.where((LiTime >= _tr[0]) & (LiTime <= _tr[1]))[0]
                    ]
                    _dummyLib = LiBes[
                        :, np.where((LiTime >= _tr[0]) & (LiTime <= _tr[1]))[0]
                    ]
                    _dummyRho = LiRho[
                        :, np.where((LiTime >= _tr[0]) & (LiTime <= _tr[1]))[0]
                    ]
                    IpolSp = UnivariateSpline(IpolT, IpolS, s=0)(_dummyTime)
                    # on these we choose a threshold
                    # which can be set as also set as keyword
                    ElmMask = np.zeros(IpolSp.size, dtype="bool")
                    ElmMask[np.where(IpolSp > _thr)[0]] = True
                    _interElm = np.where(ElmMask == False)[0]
                    # in this way they are
                    _dummyLib = _dummyLib[:, _interElm].ravel()
                    _dummyRho = _dummyRho[:, _interElm].ravel()
                    rhoNe, Ne, errNe = Target.PlotEnProfile(
                        trange=_tr, interelm=True, threshold=_thr, Plot=False
                    )
                    rhoTe, Te, errTe = Target.PlotTeProfile(
                        trange=_tr, interelm=True, threshold=_thr, Plot=False
                    )
                else:
                    _dummyLib = LiBes[
                        :, np.where((LiTime >= _tr[0]) & (LiTime <= _tr[1]))[0]
                    ].ravel()
                    _dummyRho = LiRho[
                        :, np.where((LiTime >= _tr[0]) & (LiTime <= _tr[1]))[0]
                    ].ravel()
                    rhoNe, Ne, errNe = Target.PlotEnProfile(
                        trange=_tr, interelm=False, Plot=False
                    )
                    rhoTe, Te, errTe = Target.PlotTeProfile(
                        trange=_tr, interelm=False, Plot=False
                    )
                yOut, bins, bin_means, bin_width, xOut = bin_by.bin_by(
                    _dummyRho, _dummyLib, nbins=20
                )
                xB = np.asarray([np.nanmean(k) for k in xOut])
                xBE = np.asarray([np.nanstd(k) for k in xOut])
                yB = np.asarray([np.nanmean(k) for k in yOut])
                yBE = np.asarray([np.nanstd(k) for k in yOut])
                enS = yB[np.argmin(np.abs(xB - 1))]
                upAxes.errorbar(
                    xB, yB / enS, xerr=xBE, yerr=yBE / enS, fmt="-", color=_col, lw=2.5
                )
                axNe.errorbar(
                    rhoNe, Ne / 1e19, yerr=errNe / 1e19, fmt="o", ms=14, color=_col
                )
                axTe.errorbar(rhoTe, Te, yerr=errTe, fmt="o", ms=14, color=_col)
                np.savez(
                    "../Data/AUG/LastTargetShot{}_t{:.2f}".format(
                        shot, np.average(_tr)
                    ),
                    rhoNe=rhoNe,
                    Ne=Ne,
                    errNe=errNe,
                    rhoTe=rhoTe,
                    Te=Te,
                    errTe=errTe,
                )
            figProfile.savefig(
                "../pdfbox/GeneralProfilesShot{}.pdf".format(shot),
                bbox_to_inches="tight",
            )

    elif selection == 6:
        shot = 36574
        try:
            c = mds.Connection("localhost:8001")
            _hasMds = True
        except:
            _hasMds = False
            print("Using dd local library")
        colorList = ("#2679B1", "#F2811D")
        trangeList = ((1.9, 1.95), (6.15, 6.35))
        if _hasMds:
            _s = 'augsignal({},"MAC","Ipolsola")'.format(shot)
            Data = -c.get(_s).data()
            time = c.get('dim_of(augsignal({}, "MAC", "Ipolsola"))'.format(shot)).data()
        else:
            Data = -dd.shotfile("MAC", shot)("Ipolsola").data
            time = dd.shotfile("MAC", shot)("Ipolsola").time
        fig, ax = mpl.pylab.subplots(figsize=(10, 5), nrows=1, ncols=1)
        fig.subplots_adjust(bottom=0.15, left=0.17)
        ax.set_xlabel(r"t [s]")
        ax.set_ylabel(r"I$_{shunt}$ [kA]")
        ax.set_title(r"#{}".format(shot))
        ax.set_xlim([0, 8])
        ax.plot(time, Data / 1e3, "-", color="gray", alpha=0.5)
        fig.savefig(
            "../pdfbox/ExampleELMShot{}".format(shot) + ".pdf", bbox_to_inches="tight"
        )
        for tr, _col in zip(trangeList, colorList):
            fig, ax = mpl.pylab.subplots(figsize=(10, 5), nrows=1, ncols=1)
            fig.subplots_adjust(bottom=0.15, left=0.17)
            ax.set_xlabel(r"t [s]")
            ax.set_ylabel(r"I$_{shunt}$ [kA]")
            ax.set_title(r"#{}".format(shot))
            ax.set_xlim([0, 8])
            ax.plot(time, Data / 1e3, "-", color="gray", alpha=0.5)
            # inset axes....
            axins = ax.inset_axes([0.05, 0.6, 0.47, 0.37])
            axins.plot(time, Data / 1e3, "-", color=_col)
            # sub region of the original image
            ymx = Data[np.logical_and(time >= tr[0], time <= tr[1])].max() / 1e3
            x1, x2, y1, y2 = tr[0], tr[1], 0, ymx
            axins.set_xlim([x1, x2])
            axins.set_ylim([y1, y2])
            axins.set_xticklabels("")
            axins.set_yticklabels("")
            axins.spines["top"].set_visible(False)
            axins.spines["right"].set_visible(False)
            ax.indicate_inset_zoom(axins)
            fig.savefig(
                "../pdfbox/ExampleELMShot{}_t{:.1f}".format(shot, tr[0]) + ".pdf",
                bbox_to_inches="tight",
            )

    elif selection == 7:
        rho, Lpar = np.loadtxt("../Data/AUG/Lparallel.txt", unpack=True)
        stList = {
            "34276": {
                "tlist": ((2.245, 2.345), (3.0, 3.1), (4.63, 4.73), (5.4, 5.5)),
                "thresh": [-99.0, 1200, 500, 500],
            },
            "36574": {
                "tlist": ((1.9, 2.1), (3.9, 4.1), (4.9, 5.1), (6.15, 6.35)),
                "thresh": (3000, 1000, 1000, -99),
            },
            "36605": {
                "tlist": ((2.1, 2.2), (4.1, 4.2), (4.9, 5.0), (6.25, 6.35)),
                "thresh": (3000, 1000, 1000, 1000),
            },
        }
        colorList = ("#2679B1", "#BC2764", "#00805E", "#F2811D")
        for k in stList.keys():
            fig, ax = mpl.pylab.subplots(figsize=(6, 4), nrows=1, ncols=1)
            fig.subplots_adjust(bottom=0.2, left=0.19, top=0.9, right=0.98)
            shot = int(k)
            tList = stList[k]["tlist"]

            for _tr, _col in zip(tList, colorList):
                File = "../Data/AUG/LastTargetShot{}_t{:.2f}.npz".format(
                    shot, np.average(_tr)
                )
                Data = np.load(File)
                S = UnivariateSpline(rho, Lpar, s=0)
                # limit to data in the main SOL
                _idx = np.where(Data["rhoNe"] >= 1)
                nuEi = 5e-11 * Data["Ne"][_idx] / (Data["Te"][_idx] ** 1.5)
                Cs = np.sqrt(
                    2 * constants.e * Data["Te"][_idx] / (2 * constants.proton_mass)
                )
                Lambda = (
                    nuEi
                    * S(Data["rhoNe"][_idx])
                    * constants.electron_mass
                    / (constants.proton_mass * Cs)
                )
                ax.semilogy(Data["rhoNe"][_idx], Lambda)
            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(r"$\Lambda_{div}$")
            ax.set_title(r"#{}".format(shot))
            fig.savefig(
                "../pdfbox/LambdaProfile{}.pdf".format(shot), bbox_to_inches="tight"
            )
    elif selection == 8:
        shot = 36574
        # plot also the evolution of blob-filaments together with the pressure
        try:
            c = mds.Connection("localhost:8001")
            _hasMds = True
        except:
            _hasMds = False
            print("Using dd local library")
        fig, ax = mpl.pylab.subplots(figsize=(6, 8), nrows=2, ncols=1, sharex="col")
        fig.subplots_adjust(hspace=0.05, left=0.2, bottom=0.12, top=0.96, right=0.95)
        if _hasMds:
            _s = 'augsignal({},"IOC","F01")'.format(shot)
            Data = c.get(_s).data() / 1e23
            _st = 'dim_of(augsignal({},"IOC","F01"))'.format(shot)
            ax[0].plot(c.get(_st).data(), bn.move_mean(Data, 100), color="k", lw=2.5)
        else:
            _s = dd.shotfile("IOC", shot)("F01")
            ax[0].plot(_s.time, bn.move_mean(_s.data / 1e23, 100), color="k", lw=2.5)
        ax[0].text(
            0.25,
            0.43,
            r"[10$^{23}$molecules/m$^{2}$/s]",
            transform=ax[0].transAxes,
            fontsize=20,
        )
        File = "../Data/AUG/36574_t_2.000_7.500_PMT1"
        Data = myThbObject.myThbObject(File)
        _idx = np.where(Data["rho_pol_mean"] >= 1.076)[0]
        colorList = ("#2C3E50", "#E74C3C", "#01A2A6", "#FF8C00")
        for _i, _c, rho in zip(_idx, colorList, Data["rho_pol_mean"][_idx]):
            ax[1].plot(
                Data["int_time"],
                Data["blob_freq_trace"][_i, :],
                "-",
                lw=2,
                color=_c,
                label=r"$\rho$ = {:.3f}".format(rho),
            )
        ax[1].set_xlim([1.8, 6.5])
        ax[1].set_xlabel(r"t [s]")
        ax[1].set_ylabel(r"filament frequency [Hz]")
        ax[1].legend(loc="best", numpoints=1, frameon=False, fontsize=12)
        fig.savefig("../pdfbox/BlobFrequencyFarSOL.pdf", bbox_to_inches="tight")

    elif selection == 99:
        loop = False
    else:
        input("Unknown Option Selected!")

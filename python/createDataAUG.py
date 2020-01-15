import numpy as np
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator

try:
    import MDSplus as mds
except:
    print("MDSplus not loaded")
try:
    import dd
except:
    print("dd library loaded")
from collections import OrderedDict
import bin_by
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter
from scipy import constants
import bottleneck as bn

try:
    import langmuir
except:
    print("langmuir class not loaded")
from aug import myThbObject
import seaborn as sns


def print_menu():
    print(30 * "-", "MENU", 30 * "-")
    print("1. AUG profile for synopsis")
    print("99: End")
    print(67 * "-")


loop = True

while loop:
    print_menu()
    selection = int(input("Enter your choice [1-99] "))
    if selection == 1:
        shot= 34276
        tList = ((2.245, 2.345), (5.4, 5.5))
        threshL = (-99.0, 500)
        LiBes = dd.shotfile("LIN", shot)("ne").data.transpose() / 1e19
        LiRho = dd.shotfile("LIN", shot)("ne").area.data.transpose()
        LiTime = dd.shotfile("LIN", shot)("ne").time
        Ipol = dd.shotfile("MAC", shot)("Ipolsola").data
        IpolTime = dd.shotfile("MAC", shot)("Ipolsola").time
        F01 = dd.shotfile("IOC", shot)("F01")
        En = dd.shotfile("TOT", shot)("n/nGW")
        # now the profiles which is more subtle and long
        for _idx, (_tr, _thr) in enumerate(zip(tList, threshL)):
            _idx = np.where(np.logical_and(F01.time >= _tr[0], F01.time <= _tr[1]))
            PrLabel = F01.data[_idx].mean()
            _idx = np.where(np.logical_and(En.time >= _tr[0], En.time <= _tr[1]))
            gW = 0.8/(np.pi*np.power(0.5, 2))
            EnLabel = En.data[_idx].mean()*gW
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
            else:
                _dummyLib = LiBes[
                    :, np.where((LiTime >= _tr[0]) & (LiTime <= _tr[1]))[0]
                ].ravel()
                _dummyRho = LiRho[
                    :, np.where((LiTime >= _tr[0]) & (LiTime <= _tr[1]))[0]
                ].ravel()
            yOut, bins, bin_means, bin_width, xOut = bin_by.bin_by(
                _dummyRho, _dummyLib, nbins=20
            )
            rawRho = np.asarray([np.nanmean(k) for k in xOut])
            rawRho_err = np.asarray([np.nanstd(k) for k in xOut])
            rawEn = np.asarray([np.nanmean(k) for k in yOut])
            rawEn_err = np.asarray([np.nanstd(k) for k in yOut])
            if _idx == 1:
                rawEn_err /= 2.5
            Data = np.load(
                "/afs/ipp/home/n/nvianell/analisi/28itpa-div-sol/Data/AUG/LastTargetShot{}_t{:.2f}".format(
                    shot, np.average(_tr)
                )+'.npz'
            )
            teOSP = Data["Te"][np.argmin(np.abs(Data["rhoTe"] - 1))]
            File = "../data/AUG/ProfileShot{}_t{:.2f}-{:.2f}".format(shot, _tr[0], _tr[1])
            np.savez(
                File,
                rawRho=rawRho,
                rawRho_err=rawRho_err,
                rawEn=rawEn,
                rawEn_err=rawEn_err,
                enAvg=EnLabel,
                pdiv=PrLabel,
                teosp=teOSP,
            )

    elif selection == 99:
        loop = False
    else:
        input("Unknown Option Selected!")

import profiletools
import MDSplus as mds
import langmuir as tcvlangmuir, baratrons as tcvbaratrons, tcvgeom, gas as tcvgas, tcvProfiles
import numpy as np
import gpr1dfusion
from scipy.interpolate import UnivariateSpline, interp1d
from scipy import constants


def print_menu():
    print(30 * "-", "MENU", 30 * "-")
    print("1. save profile for sinopsys figure")
    print("99: End")
    print(67 * "-")


loop = True

while loop:
    print_menu()
    selection = int(input("Enter your choice [1-99] "))
    if selection == 1:
        # create the profiles together with the fit and save in appropriate npz file
        shotlist = (64494, 64947)
        trlist = ([0.88, 0.96], [1.5, 1.75])
        for _idxshot, (shot, tr) in enumerate(zip(shotlist, trlist)):
            Tree = mds.Tree("tcv_shot", shot)
            Data = Tree.getNode(r"\base::pd:pd_001")
            En = Tree.getNode(r"\results::fir:n_average")
            tEn = En.getDimensionAt().data()
            En = En.data()
            Pr = tcvbaratrons.pressure(shot).divertor
            tPr = Pr.t.values
            Pr = Pr.values
            _idx = np.where((tEn >= tr[0]) & (tEn <= tr[1]))[0]
            enLabel = En[_idx].mean() / 1e19
            _idx = np.where((tPr >= tr[0]) & (tPr <= tr[1]))[0]
            PrLabel = Pr[_idx].mean() * 1e3
            P = tcvProfiles.tcvProfiles(shot)
            En = P.profileNe(
                t_min=tr[0],
                t_max=tr[1],
                abscissa="sqrtpsinorm",
                interelm=True,
                rho=0.82,
                width=0.05,
            )
            _ = En.remove_points(
                np.logical_and(En.X.ravel() <= 0.9, En.X.ravel() >= 1.12)
            )
            fitRho = np.linspace(En.X.ravel().min(), 1.12, 120)
            fitEn, fitEn_err, _, _ = P.gpr_robustfit(
                fitRho, density=True, temperature=False, regularization_parameter=6
            )
            rawRho = En.X.ravel()
            rawEn = En.y
            rawEn_err = En.err_y
            Target = tcvlangmuir.LP(shot)
            teOSP = np.asarray(
                [
                    Target.te[i, np.argmin(np.abs(Target.Rho[i, :] - 1))]
                    for i in range(Target.t.size)
                ]
            )
            _idx = np.where(np.logical_and(Target.t >= tr[0], Target.t <= tr[1]))
            teOSPm = np.mean(teOSP)
            File = "../data/TCV/ProfileShot{}_t{:.2f}-{:.2f}".format(shot, tr[0], tr[1])
            np.savez(
                File,
                rawRho=rawRho,
                rawEn=rawEn,
                rawEn_err=rawEn_err,
                fitRho=fitRho,
                fitEn=fitEn,
                fitEn_err=fitEn_err,
                enAvg=enLabel,
                pdiv=PrLabel,
                teosp=teOSPm,
            )
    elif selection == 2:
        pass

    elif selection == 99:
        loop = False
    else:
        input("Unknown Option Selected!")

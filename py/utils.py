import os
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

plt.style.use(["science", "ieee"])
import numpy as np
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, num2date
from matplotlib import patches
import matplotlib.patches as mpatches
import matplotlib.colors as colors

import matplotlib.dates as dates
from scipy.signal import medfilt


import os
import datetime as dt
import pandas as pd
import numpy as np

import aacgmv2

import plots

def smooth(x, window_len=51, window="hanning"):
    if x.ndim != 1: raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len: raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = numpy.ones(window_len,"d")
    else: w = eval("np."+window+"(window_len)")
    y = np.convolve(w/w.sum(),s,mode="valid")
    d = window_len - 1
    y = y[int(d/2):-int(d/2)]
    return y

def read_omni_gs():
    import bezpy
    magfiles = ["data/zhs20120616dmin.min", "data/maw20120616dmin.min", "data/paf20120616dmin.min", 
                "data/her20120616dmin.min", "data/aae20120616dmin.min"]
    mags_name = [
            {"name": "ZHO", "mlat": -75.0, "mlt": 11.5},
            {"name": "MAW", "mlat": 73.2, "mlt": 11.2},
            {"name": "PAF", "mlat": -58.63, "mlt": 13.4},
            {"name": "HER", "mlat": -42.08, "mlt": 10.8},
            {"name": "AAE", "mlat": 5.32, "mlt": 12.6}
        ]
    mags = []
    for mf in magfiles:
        o = bezpy.mag.read_iaga(mf)
        mags.append(o)
    with open("data/symh20120616.min", "r") as f: lines = f.readlines()
    sh = []
    for l in lines:
        l = list(filter(None, l.replace("\n", "").split(" ")))
        date = dt.datetime(int(l[0]),1,1) + dt.timedelta(int(l[1])-1) + dt.timedelta(hours=int(l[2])) +\
                dt.timedelta(minutes=int(l[3]))
        sh.append({"time":date, "symh": float(l[4])})
    sh = pd.DataFrame.from_records(sh)
    mags.append(sh)
    with open("data/WI_H0_MFI_25079.txt", "r") as f: lines = f.readlines()
    imf = []
    for l in lines[102:-4]:
        l = list(filter(None, l.replace("\n", "").split(" ")))
        date = dt.datetime.strptime(l[0]+"T"+l[1].split(".")[0], "%d-%m-%YT%H:%M:%S")
        imf.append({"time": date, "B":float(l[2]), "Bx":float(l[3]), "By":float(l[4]), "Bz":float(l[5])})
    imf = pd.DataFrame.from_records(imf)
    with open("data/WI_K0_SWE_31798.txt", "r") as f: lines = f.readlines()
    omni = []
    for l in lines[70:-4]:
        l = list(filter(None, l.replace("\n", "").split(" ")))
        date = dt.datetime.strptime(l[0]+"T"+l[1].split(".")[0], "%d-%m-%YT%H:%M:%S")
        omni.append({"time": date, "np":float(l[2]), "pdy":float(l[3]), 
                    "vs":np.sqrt(float(l[5])**2+float(l[6])**2+float(l[7])**2)})
    omni = pd.DataFrame.from_records(omni)
    plots.create_figure2(imf, omni, mags, mags_name)
    return

def get_gridded_parameters(q, xparam="beam", yparam="slist", zparam="v", r=0, rounding=False):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[ [xparam, yparam, zparam] ]
    if rounding:
        plotParamDF.loc[:, xparam] = np.round(plotParamDF[xparam].tolist(), r)
        plotParamDF.loc[:, yparam] = np.round(plotParamDF[yparam].tolist(), r)
    else:
        plotParamDF[xparam] = plotParamDF[xparam].tolist()
        plotParamDF[yparam] = plotParamDF[yparam].tolist()
    plotParamDF = plotParamDF.groupby( [xparam, yparam] ).mean().reset_index()
    plotParamDF = plotParamDF[ [xparam, yparam, zparam] ].pivot( xparam, yparam )
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y  = np.meshgrid( x, y )
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
            np.isnan(plotParamDF[zparam].values),
            plotParamDF[zparam].values)
    return X,Y,Z

def read_plot_velocity():
    Zdvl = []
    with open("data/ZS36R_2012168.DVL", "r") as f: lines = f.readlines()
    for l in lines:
        l = list(filter(None, l.replace("\n", "").split(" ")))
        date = dt.datetime.strptime(l[6]+"T"+l[8], "%Y/%m/%dT%H:%M:%S")
        Zdvl.append({"date": date, "x":float(l[9]), "xe":float(l[10]), 
                     "y":float(l[11]), "ye":float(l[12]), "z":float(l[13]), "ze":float(l[14])})
    Zdvl = pd.DataFrame.from_records(Zdvl)
    Zdvl = Zdvl.set_index("date")
    Zdvl = Zdvl.rolling('1200s', min_periods=3).median()
    Zdvl = Zdvl.reset_index()
    Zdvl = Zdvl[(Zdvl.date>=dt.datetime(2012,6,16,9,30)) & (Zdvl.date<=dt.datetime(2012,6,16,10,30))]
    print(Zdvl.head())
    plots.create_figure4(Zdvl)
    return


def read_create_fig9x():
    
    files = [
        "data/POES_combinedSpectrum_n15_00_20120616.nc",
        "data/POES_combinedSpectrum_n16_00_20120616.nc",
        "data/POES_combinedSpectrum_n17_00_20120616.nc",
        "data/POES_combinedSpectrum_n18_00_20120616.nc",
        "data/POES_combinedSpectrum_n19_00_20120616.nc"
    ]
    
    dxs = pd.DataFrame()
    import xarray as xr
    for f in files:
        ds = xr.open_dataset(f)
    print(dxs.head())
    
    ds0 = [
        dt.datetime(2012,6,16,8,5), 
        dt.datetime(2012,6,16,9,45)
    ]
    
    mpl.rcParams.update({"xtick.labelsize":12, "ytick.labelsize":12, "font.size":12})
    fig = plt.figure(figsize=(10, 4), dpi=300)
    for i, d0 in enumerate(ds0):
        dates = [d0, d0 + dt.timedelta(hours=0.5)]
        for f in files:
            ds = xr.open_dataset(f)
            lat, lon, flat, flon, mlt, dts, e, p = [], [], [], [], [], [], [], []
            date_m = [dt.datetime(2012,6,16) + dt.timedelta(hours=t) for t in ds.variables["time"][:]]
            i, j = date_m.index(dates[0]), date_m.index(dates[-1])
            o = pd.DataFrame()
            energy = ds.variables["energy"][:]
            Ecounts, Pcounts = ds.variables["Ecounts"][:], ds.variables["Pcounts"][:]
            Ecounts, Pcounts = Ecounts[i:j+1,:], Pcounts[i:j+1,:]
            i, j = o.index.tolist()[0], o.index.tolist()[-1]
            Ecounts = Ecounts[i:j+1,:]
            Pcounts = Pcounts[i:j+1,:]
            Ecounts[Ecounts==0] = np.nan
            Pcounts[Pcounts==0] = np.nan
            eng = []
            for c in range(i, j+1):
                lat.append([ds.variables["geogLat"][c]]*len(energy))
                lon.append([ds.variables["geogLon"][c]]*len(energy))
                flat.append([ds.variables["foflLat"][c]]*len(energy))
                flon.append([ds.variables["foflLon"][c]]*len(energy))
                mlt.append([ds.variables["MLT"][c]]*len(energy))
                dts.append([date_m[c]]*len(energy))
                ## ---TODO---
#                 for f in range():
                    
            e = Ecounts.ravel()
            p = Pcounts.ravel()
        
        ax0 = fig.add_subplot(221+i)
        ax1 = fig.add_subplot(222+i)
        ax0.xaxis.set_major_formatter(DateFormatter(r"${%H}^{%M}$"))
        ax1.xaxis.set_major_formatter(DateFormatter(r"${%H}^{%M}$"))
        hours = mdates.MinuteLocator(range(0, 60, 5))
        ax0.xaxis.set_major_locator(hours)
        ax0.set_ylabel("Energy Channels, eV", fontdict={"size":8, "fontweight": "bold"})
        ax0.set_yscale("log")
        ax1.xaxis.set_major_locator(hours)
        ax1.set_xlabel("Time, UT\n Latitude (deg)\n Foot Latitude (deg)", fontdict={"size":8, "fontweight": "bold"})
        ax1.set_ylabel("Energy Channels, eV", fontdict={"size":8, "fontweight": "bold"})
        ax1.set_yscale("log")
        ax0.set_xlim(dates)
        ax1.set_xlim(dates)
        
    return

def read_create_fig9():
    d0 = dt.datetime(2012,6,16,8,5)
    d0 = dt.datetime(2012,6,16,9,45)
    dates = [d0, d0+dt.timedelta(hours=0.5)]
    mpl.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize":8, "font.size":8})
    fig = plt.figure(figsize=(5, 4), dpi=300 )
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    ax0.xaxis.set_major_formatter(DateFormatter(r"${%H}^{%M}$"))
    ax1.xaxis.set_major_formatter(DateFormatter(r"${%H}^{%M}$"))
    hours = mdates.MinuteLocator(range(0, 60, 5))
    ax0.xaxis.set_major_locator(hours)
    ax0.set_ylabel("Energy Channels, eV", fontdict={"size":8, "fontweight": "bold"})
    ax0.set_yscale("log")
    ax1.xaxis.set_major_locator(hours)
    ax1.set_xlabel("Time, UT\n Latitude (deg)\n Foot Latitude (deg)", fontdict={"size":8, "fontweight": "bold"})
    ax1.set_ylabel("Energy Channels, eV", fontdict={"size":8, "fontweight": "bold"})
    ax1.set_yscale("log")
    ax0.set_xlim(dates)
    ax1.set_xlim(dates)
    
    files = [
        "data/POES_combinedSpectrum_n15_00_20120616.nc",
        "data/POES_combinedSpectrum_n16_00_20120616.nc",
        "data/POES_combinedSpectrum_n17_00_20120616.nc",
        "data/POES_combinedSpectrum_n18_00_20120616.nc",
        "data/POES_combinedSpectrum_n19_00_20120616.nc"
    ]
    from netCDF4 import Dataset
    for m, f in enumerate(files):
        ds = Dataset(f)
        lat, lon = ds.variables["geogLat"][:], ds.variables["geogLon"][:]
        flat, flon = ds.variables["foflLat"][:], ds.variables["foflLon"][:]
        mlt = ds.variables["MLT"][:]
        energy = ds.variables["energy"][:]
        Ecounts, Pcounts = ds.variables["Ecounts"][:], ds.variables["Pcounts"][:]
        date = [dt.datetime(2012,6,16) + dt.timedelta(hours=t) for t in ds.variables["time"][:]]
        o = pd.DataFrame()
        o["lat"], o["lon"] = lat, lon
        o["flat"], o["flon"] = flat, flon
        o["mlt"] = mlt
        o["date"] = date
        o = o[(o.date>=dates[0]) & (o.date<=dates[1])]
        xdates = [d0 + dt.timedelta(minutes=x*5) for x in range(7)]
        if m==3:            
            i, j = o.index.tolist()[0], o.index.tolist()[-1]
            Ecounts = Ecounts[i:j+1,:]
            Pcounts = Pcounts[i:j+1,:]
            Ecounts[Ecounts==0] = np.nan
            Pcounts[Pcounts==0] = np.nan
            p = ax0.pcolormesh(o.date, energy, Ecounts.T, cmap="Reds", 
                               norm=colors.LogNorm(vmin=0.1, vmax=100), shading="auto")
            print(energy, Ecounts)
            ax0.axvline(dt.datetime(2012,6,16,9,56), lw=0.6, color="k", ls="--")
            labels = [item.get_text() for item in ax0.get_xticklabels()]
            tags = []
            for i, lx in enumerate(xdates):
                u = o[(o.date>=lx-dt.timedelta(minutes=1)) & (o.date<=lx+dt.timedelta(minutes=1))]
                tags.append(
                    lx.strftime(r"$%H^{%M}$") 
                    + "\n" + r"%d$^\circ$"%u.lat.mean() + "\n"
                    + r"%d$^\circ$"%u.flat.mean()
                )
            ax0.set_xticklabels(labels)
            if m==len(files)-1:
                cb = fig.colorbar(p, ax=ax0, extend="max", shrink=0.7)
                cb.set_label(r"\#$e^-$, $N/cm^2/sr/keV$")
            p = ax1.pcolormesh(o.date, energy, Pcounts.T, cmap="Blues", 
                               norm=colors.LogNorm(vmin=0.1, vmax=1000), shading="auto")
            ax1.axvline(dt.datetime(2012,6,16,9,56), lw=0.6, color="k", ls="--")
            ax1.set_xticklabels(tags)
            if m==len(files)-1: 
                cb = fig.colorbar(p, ax=ax1, extend="max", shrink=0.7)
                cb.set_label(r"\#$p^+$, $N/cm^2/sr/keV$")
            break
        ds.close()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.savefig("figures/Figure9.png")
    return

def read_cdfs_fig10():
    d0 = dt.datetime(2012,6,16,8)
    dates = [d0, d0+dt.timedelta(hours=4)]
    mpl.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize":8, "font.size":8})
    fig = plt.figure(figsize=(5, 4), dpi=240 )
    ax0 = fig.add_subplot(211)
    #ax1 = fig.add_subplot(212)
    ax0.xaxis.set_major_formatter(DateFormatter(r"${%H}^{%M}$"))
    #ax1.xaxis.set_major_formatter(DateFormatter(r"{%H}^{%M}"))
    hours = mdates.MinuteLocator(range(0, 60, 5))
    ax0.xaxis.set_major_locator(hours)
    ax0.set_ylabel("$<Energy>$, eV", fontdict={"size":8, "fontweight": "bold"})
    ax0.set_yscale("log")
    ax0.set_ylim(1e-2,1e2)
    #ax1.xaxis.set_major_locator(hours)
    ax0.set_xlabel("Time, UT\nLatitude (deg)", fontdict={"size":8, "fontweight": "bold"})
    #ax1.set_ylabel("Energy Channels, eV", fontdict={"size":8, "fontweight": "bold"})
    #ax1.set_yscale("log")
    ax0.set_xlim(dates)
    #ax1.set_xlim(dates)
    xdates = [d0 + dt.timedelta(minutes=x*20) for x in range(13)]
    import cdflib
    files = [
        "data/dmsp-f16_ssj_precipitating-electrons-ions_20120616_v1.1.2.cdf",
        "data/dmsp-f17_ssj_precipitating-electrons-ions_20120616_v1.1.2.cdf",
        "data/dmsp-f18_ssj_precipitating-electrons-ions_20120616_v1.1.2.cdf"
    ]
    for m, f in enumerate(files):
        cf = cdflib.CDF(f)
        print(cf.cdf_info())
        time = cdflib.cdfepoch.breakdown(cf.varget("Epoch"))
        time = [dt.datetime(x[0], x[1], x[2], x[3], x[4], x[5]) for x in time]
        Eflux = cf.varget("ELE_DIFF_ENERGY_FLUX")
        Iflux = cf.varget("ION_DIFF_ENERGY_FLUX")
        energy = cf.varget("CHANNEL_ENERGIES")
        i, j = time.index(dates[0]), time.index(dates[1])
        Eflux = Eflux[i:j+1, :]
        Iflux = Iflux[i:j+1, :]
        time = time[i:j+1]
        o = pd.DataFrame()
        o["time"] = time
        o["lat"], o["lon"] = cf.varget("SC_GEOCENTRIC_LAT")[i:j+1], cf.varget("SC_GEOCENTRIC_LON")[i:j+1]
#         p0 = ax0.pcolormesh(time, energy, Eflux.T, cmap="jet", 
#                            norm=colors.LogNorm(vmin=1e5, vmax=1e10), shading="auto")
#         p1 = ax1.pcolormesh(time, energy, Iflux.T, cmap="jet", 
#                            norm=colors.LogNorm(vmin=1e5, vmax=1e10), shading="auto")
        if m==0:
            ax0.semilogy(time, smooth(cf.varget("ELE_TOTAL_ENERGY_FLUX")[i:j+1]/1e12), "ro", ms=0.5, alpha=0.4, label=r"$e^-$")
            ax0.semilogy(time, smooth(cf.varget("ION_TOTAL_ENERGY_FLUX")[i:j+1]/1e12), "bs", ms=0.5, alpha=0.4, label=r"$p^+$")
        else:
            ax0.semilogy(time, smooth(cf.varget("ELE_TOTAL_ENERGY_FLUX")[i:j+1]/1e12), "ro", ms=0.5, alpha=0.4)
            ax0.semilogy(time, smooth(cf.varget("ION_TOTAL_ENERGY_FLUX")[i:j+1]/1e12), "bs", ms=0.5, alpha=0.4)
        if m==0: ax0.axvline(dt.datetime(2012,6,16,9,56), lw=0.6, color="k", ls="--")
        labels = [item.get_text() for item in ax0.get_xticklabels()]
        tags = []
        for i, lx in enumerate(xdates):
            u = o[(o.time>=lx-dt.timedelta(minutes=1)) & (o.time<=lx+dt.timedelta(minutes=1))]
            tags.append(
                lx.strftime(r"$%H^{%M}$") 
                + "\n" + r"%d$^\circ$"%u.lat.mean()
            )
        ax0.set_ylim(0, 2)
        ax0.set_xticks(xdates)
        ax0.set_xticklabels(tags)
        #ax1.set_xticks(xdates)
        #ax1.set_xticklabels(tags)

#         if m==len(files)-1: 
#             cb = fig.colorbar(p0, ax=ax0, extend="max", shrink=0.7)
#             cb.set_label(r"\#$e^-$, $eV/cm^2/sr/\Delta eV/s$")
#             cb = fig.colorbar(p1, ax=ax1, extend="max", shrink=0.7)
#             cb.set_label(r"\#$p^+$, $eV/cm^2/sr/\Delta eV/s$")
    ax0.legend(loc=2)
    fig.savefig("figures/Figure10.png")
    return

def read_cdfs_fig11():
    mpl.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize":8, "font.size":8})
    fig = plt.figure(figsize=(9, 2.5), dpi=240 )
    files = [
        "data/zs36r_20120616(168)094850_sky.txt",
        "data/zs36r_20120616(168)095620_sky.txt",
        "data/zs36r_20120616(168)100350_sky.txt",
    ]
    lab = ["(a)", "(b)", "(c)"]
    times = ["09:48 UT", "09:56 UT", "10:03 UT"]
    for m, fx in enumerate(files):
        ax0 = fig.add_subplot(131+m, projection="polar")
        K = []
        with open(fx, "r") as f: lines = f.readlines()
        for l in lines[6:]:
            l = np.array(list(filter(None, l.replace("\n", "").split(" ")))).astype(float)
            K.append({"F":l[0],"R":l[1],"Z":l[2],"Az":l[3],"D":l[4],"E":l[5],"A":l[6]})
        K = pd.DataFrame.from_records(K)
        p = ax0.scatter(K.Az, np.cos(np.deg2rad(90-K.Z))*K.R, s=0.4, c=K.D, alpha=0.6, 
                        vmax=2.5, vmin=-2.5, cmap="jet_r")
        ax0.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax0.set_xticklabels([r"$N_{m}$", r"$W_{m}$", r"$S_{m}$", r"$E_{m}$"])
        ax0.set_theta_zero_location("N")
        ax0.set_yticks([])
        ax0.text(-0.05, 0.9, lab[m]+" "+times[m], ha="left", va="center", transform=ax0.transAxes)
        Vv = 3*1e8*np.cos(np.deg2rad(K.Z))*np.array(K.D*1e-3/K.F)
        Vh = 3*1e8*np.sin(np.deg2rad(K.Z))*np.array(K.D*1e-3/K.F)
        txt = r"$V_z=%.1f \pm %.1f$ m/s"%(Vv.mean(), Vv.std()/10)# + "\n"\
            #r"$V_h=%.1f \pm %.1f$ m/s"%(Vh.mean(), Vh.std()/10) 
        ax0.text(0.01, 0.2, txt, ha="left", va="center", transform=ax0.transAxes, fontdict={"size":6})
        if m==2:
            fig.subplots_adjust(right=0.8)
            cax = fig.add_axes([0.85, 0.15, 0.01, 0.4])
            cb = fig.colorbar(p, cax=cax, shrink=0.6)
            cb.set_label(r"Doppler, Hz")
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig("figures/Figure11a.png", bbox_inches="tight")
    
    
    es = pd.read_csv("data/Es.txt", parse_dates=["date"])
    mpl.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize":8, "font.size":8})
    fig = plt.figure(figsize=(5, 2), dpi=180 )
    ax0 = fig.add_subplot(111)
    ax0.xaxis.set_major_formatter(DateFormatter(r"${%H}^{%M}$"))
    hours = mdates.MinuteLocator(range(0, 60, 15))
    ax0.xaxis.set_major_locator(hours)
    ax0.set_ylabel("$foE_s$, MHz", fontdict={"size":8, "fontweight": "bold", "color": "r"})
    ax0.set_xlabel("Time, UT", fontdict={"size":8, "fontweight": "bold"})
    ax0.axvline(dt.datetime(2012,6,16,9,56), lw=0.6, color="k", ls="--")
    ax0.set_ylim(2, 4)
    ax0.set_xlim(dt.datetime(2012,6,16,9,30),dt.datetime(2012,6,16,10,30))
    ax0.plot(es.date, es.foEs, "ro", ms=0.9, alpha=0.9)
    ax0 = ax0.twinx()
    ax0.xaxis.set_major_formatter(DateFormatter(r"${%H}^{%M}$"))
    ax0.xaxis.set_major_locator(hours)
    ax0.set_ylabel("$h'E_s$, km", fontdict={"size":8, "fontweight": "bold", "color": "b"})
    ax0.set_xlabel("Time, UT", fontdict={"size":8, "fontweight": "bold"})
    ax0.set_ylim(100, 150)
    ax0.plot(es.date, es.hEs, "bD", ms=0.9, alpha=0.9)
    ax0.text(0.05, 0.8, "(g)", ha="left", va="center", transform=ax0.transAxes)
    fig.savefig("figures/Figure11c.png", bbox_inches="tight")
    #plt.close()
    
    files = [
        ("data/9.52.41.I.txt", "data/9.52.41.S.txt"),
        ("data/9.58.30.I.txt", "data/9.58.30.S.txt"),
        ("data/10.00.11.I.txt", "data/10.00.11.S.txt")
    ]
    files = [
        ("data/09.36.00.I.txt", "data/09.36.00.S.txt"),
        ("data/09.37.41.I.txt", "data/09.37.41.S.txt"),
        ("data/09.43.41.I.txt", "data/09.43.41.S.txt")
    ]
    lab = ["(d)", "(e)", "(f)"]
    fig = plt.figure(figsize=(9,2.5), dpi=240 )
    m = 0
    for fI, fS in files:
        with open(fI, "r") as f: lines = f.readlines()
        I = []
        for l in lines[5:]:
            l = list(filter(None, l.replace("\n", "").split(" ")))
            I.append({"freq":float(l[0]), "height":float(l[1])})
        I = pd.DataFrame.from_records(I)
        with open(fS, "r") as f: lines = f.readlines()
        rxline = 0
        foF2, MUF, foF1, foEs, hEs = None, None, None, None, None
        for l in lines:
            lx = list(filter(None, l.replace("\n", "").split(" ")))
            if ("foF2" in l) and (foF2 is None): foF2 = None if lx[1] == "NoValue" else float(lx[1])
            if ("foF1" in l) and (foF1 is None): foF1 = None if lx[1] == "NoValue" else float(lx[1])
            if ("foEs" in l) and (foEs is None): foEs = None if lx[1] == "NoValue" else float(lx[1])
            if ("h`Es" in l) and (hEs is None): hEs = None if lx[1] == "NoValue" else float(lx[1])
            if rxline == 2: Fs, rxline = np.array(lx).astype(float), 3
            if rxline == 1: Hs, rxline = np.array(lx).astype(float), 2
            if "Regular true" in l: rxline = 1
        I = pd.DataFrame.from_records(I)
        ax0 = fig.add_subplot(131+m)
        ax0.set_ylabel("Height, km", fontdict={"size":8, "fontweight": "bold"})
        ax0.set_xlabel("Frequency, MHz", fontdict={"size":8, "fontweight": "bold"})
        ax0.plot(I.freq, medfilt(I.height, 11), "kD", ms=0.4, alpha=0.6)
        ax0.set_ylim(80, 400)
        ax0.set_xlim(0, 9)
        da = fI.split("/")[1].split(".")
        da = da[0]+":"+da[1]+" UT"
        ax0.text(0.01, 1.05, da, ha="left", va="center", transform=ax0.transAxes)
        txt = ""
        if foF2: 
            txt += r"$foF_2=%.1f$ MHz"%foF2 + "\n"
            ax0.axvline(foF2, color="r", ls="--", lw=0.6)
        else: txt += r"$foF_2=NA$ MHz" + "\n"
        if foF1: 
            txt += r"$foF_1=%.1f$ MHz"%foF1 + "\n"
            ax0.axvline(foF1, color="m", ls="--", lw=0.6)
        else: txt += r"$foF_1=NA$ MHz" + "\n"
        if foEs: 
            txt += r"$foE_s=%.1f$ MHz"%foEs + "\n"
            ax0.axvline(foEs, color="b", ls="--", lw=0.6)
        else: txt += r"$foE_s=NA$ MHz" + "\n"
        if hEs: 
            txt += r"$h`E_s=%.1f$ km"%hEs 
            ax0.axhline(hEs, color="darkblue", ls="-", lw=0.6)
        else: txt += r"$h`E_s=NA$ MHz" + "\n"
        ax0.text(0.8,0.2,txt,ha="center",va="center",transform=ax0.transAxes, fontdict={"size":6})
        ax0.plot(Fs, Hs, "ro",  ms=0.7, alpha=0.4)
        ax0.text(0.05, 0.8, lab[m], ha="left", va="center", transform=ax0.transAxes)
        m+=1
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig("figures/Figure11b.png", bbox_inches="tight")
    plt.close()
    return
             

if __name__ == "__main__":
    read_cdfs_fig11()
    #read_create_fig9()
    #read_cdfs_fig10()
    #read_create_fig9x()
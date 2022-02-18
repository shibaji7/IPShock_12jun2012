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
import random
import pytz

import matplotlib.dates as dates

import datetime as dt
import pandas as pd

import pydarn

import utils

CLUSTER_CMAP = plt.cm.gist_rainbow



def get_cluster_cmap(n_clusters, plot_noise=False):
    cmap = CLUSTER_CMAP
    cmaplist = [cmap(i) for i in range(cmap.N)]
    while len(cmaplist) < n_clusters:
        cmaplist.extend([cmap(i) for i in range(cmap.N)])
    cmaplist = np.array(cmaplist)
    r = np.array(range(len(cmaplist)))
    random.seed(10)
    random.shuffle(r)
    cmaplist = cmaplist[r]
    if plot_noise:
        cmaplist[0] = (0, 0, 0, 1.0)    # black for noise
    rand_cmap = cmap.from_list("Cluster cmap", cmaplist, len(cmaplist))
    return rand_cmap


class RangeTimePlot(object):
    """
    Create plots for IS/GS flags, velocity, and algorithm clusters.
    """
    def __init__(self, nrang, unique_times, fig_title, num_subplots=3):
        self.nrang = nrang
        self.unique_gates = np.linspace(1, nrang, nrang)
        self.unique_times = unique_times
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(8, 3*num_subplots), dpi=150) # Size for website
        plt.suptitle(fig_title, x=0.075, y=0.95, ha="left", fontweight="bold", fontsize=15)
        mpl.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize":12, "font.size":12})
        return
        
    def addParamPlot(self, df, beam, title, p_max=100, p_min=-100, p_step=25, xlabel="Time UT", ylabel="Range (km)", zparam="v",
                     label="Velocity ($m.s^{-1}$)"):
        ax = self._add_axis()
        df.slist = (df.slist*df.rsep) + df.frang
        df = df[df.bmnum==beam]
        X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam=zparam)
        bounds = list(range(p_min, p_max+1, p_step))
        cmap = plt.cm.jet_r
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # cmap.set_bad("w", alpha=0.0)
        # Configure axes
        ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
        hours = mdates.MinuteLocator(range(0, 60, 15))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_ylim([0, 3000])
        ax.set_ylabel(ylabel, fontdict={"size":12, "fontweight": "bold"})
        cax = ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap, norm=norm)
        self._add_colorbar(self.fig, ax, bounds, cmap, label=label)
        ax.text(0.05, 0.9, title, ha="left", va="center", transform=ax.transAxes, fontdict={"fontweight": "bold"})
        ax.set_xlim([dt.datetime(2012,6,16,9,30),
                               dt.datetime(2012,6,16,10,30)])
        ax.axvline(dt.datetime(2012,6,16,9,56), ls="--", lw=1.2)
        return ax

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")

    def close(self):
        self.fig.clf()
        plt.close()

    # Private helper functions

    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        ax.tick_params(axis="both", labelsize=12)
        return ax

    def _add_colorbar(self, fig, ax, bounds, colormap, label=""):
        """
        Add a colorbar to the right of an axis.
        :param fig:
        :param ax:
        :param bounds:
        :param colormap:
        :param label:
        :return:
        """
        import matplotlib as mpl
        pos = ax.get_position()
        cpos = [pos.x1 + 0.025, pos.y0 + 0.0125,
                0.015, pos.height * 0.9]                # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        norm = mpl.colors.BoundaryNorm(bounds, colormap.N)
        cb2 = mpl.colorbar.ColorbarBase(cax, cmap=colormap,
                                        norm=norm,
                                        ticks=bounds,
                                        spacing="uniform",
                                        orientation="vertical")
        cb2.set_label(label)
        return


def create_figure4():
    from get_fit_data import FetchData
    fdata_zho = FetchData( "zho", [dt.datetime(2012,6,16,9,25),
                               dt.datetime(2012,6,16,10,35)] )
    beams, _ = fdata_zho.fetch_data()
    zho = fdata_zho.convert_to_pandas(beams)
    zho.time = zho.time.apply(lambda x: dates.date2num(x))
    fdata_mcm = FetchData( "mcm", [dt.datetime(2012,6,16,9,25),
                               dt.datetime(2012,6,16,10,35)] )
    beams, _ = fdata_mcm.fetch_data()
    mcm = fdata_mcm.convert_to_pandas(beams)
    mcm.time = mcm.time.apply(lambda x: dates.date2num(x))
    
    rti = RangeTimePlot(80, np.unique(zho.time), "", num_subplots=2)
    rti.addParamPlot(zho, 7, "Beam 7-ZHO", p_max=400, p_min=-400, p_step=80, xlabel="", zparam="v")
    rti.addParamPlot(mcm, 4, "Beam 4-MCM", p_max=400, p_min=-400, p_step=80, xlabel="Time (UT)", zparam="v")
    rti.save("figures/Figure6.png")
    return

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


def create_figure8():
    mpl.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize":8, "font.size":8})
    from get_fit_data import FetchData
    fdata_zho = FetchData( "zho", [dt.datetime(2012,6,16,9,25),
                               dt.datetime(2012,6,16,10,35)] )
    beams, _ = fdata_zho.fetch_data()
    zho = fdata_zho.convert_to_pandas(beams)
    zho.time = zho.time.apply(lambda x: dates.date2num(x))
    zho = zho.groupby("time").size().reset_index()
    
    rio = pd.read_csv("data/riometer.csv")
    R = pd.DataFrame()
    o = rio[rio.time < 10.]
    tx = [(t-9.3) for t in o.time]
    tx = (tx - np.min(tx))/np.max(tx)*30
    o.time = [dt.datetime(2012,6,16,9,30) + dt.timedelta(minutes=t) for t in tx]
    R = o.copy()
    o = rio[rio.time > 10.]
    tx = [(10.3-t) for t in o.time]
    tx = (tx - np.min(tx))/np.max(tx)*30
    o.time = [dt.datetime(2012,6,16,10,30) - dt.timedelta(minutes=t) for t in tx]
    rio = pd.concat([R,o])
    rio.data = np.roll(rio.data,60)
    
    fig = plt.figure(figsize=(5, 2), dpi=180 )
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(DateFormatter(r"{%H}^{%M}"))
    hours = mdates.MinuteLocator(range(0, 60, 15))
    ax.xaxis.set_major_locator(hours)
    ax.set_xlabel("", fontdict={"size":8, "fontweight": "bold"})
    ax.set_ylabel("\# Echoes/beam", fontdict={"size":8, "fontweight": "bold"})
    ax.text(0.05, 1.05, "Radar: ZHO", ha="left", va="center", 
            transform=ax.transAxes, fontdict={"fontweight": "bold", "size":8})
    ax.plot(zho.time, zho[0], "ko", alpha=0.6, ms=1.2)
    ax.plot(zho.time, smooth(zho[0], 101), "b", ls="-", alpha=0.9, lw=1.2)
    ax.axvline(dt.datetime(2012,6,16,9,56), color="r", lw=0.8, ls="--")
    ax.set_ylim(0,60)
    ax.set_xlim([dt.datetime(2012,6,16,9,30), dt.datetime(2012,6,16,10,30)] )
    ax.set_xlabel("Time, UT", fontdict={"size":8, "fontweight": "bold"})
    
#     ax = fig.add_subplot(312)
#     ax.xaxis.set_major_formatter(DateFormatter(r"{%H}^{%M}"))
#     hours = mdates.MinuteLocator(range(0, 60, 15))
#     ax.xaxis.set_major_locator(hours)
#     ax.set_xlabel("Time, UT", fontdict={"size":8, "fontweight": "bold"})
#     ax.set_ylabel(r"Absorption ($\beta$), dB", fontdict={"size":8, "fontweight": "bold"})
#     ax.text(0.05, 1.05, "Riometer: ", ha="left", va="center", 
#             transform=ax.transAxes, fontdict={"fontweight": "bold", "size":8})
#     ax.plot(rio.time, rio.data, "ko", alpha=0.6, ms=1.2)
#     ax.plot(rio.time, smooth(rio.data,11),  "b", ls="-", alpha=0.9, lw=0.8)
#     ax.set_ylim(0,0.2)
#     ax.set_xlim([dt.datetime(2012,6,16,9,40), dt.datetime(2012,6,16,10,20)] )
#     ax.axvline(dt.datetime(2012,6,16,9,56), color="r", lw=0.8, ls="--")
    
    fig.savefig("figures/Figure8.png")
    return

if __name__ == "__main__":
    create_figure8()
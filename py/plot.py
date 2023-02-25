import matplotlib.pyplot as plt
plt.style.use(["science", "ieee"])

import matplotlib as mpl

import pandas as pd

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime as dt
import numpy as np

import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
import datetime as dt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def overlay_sw(params, fname):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans",
                                   "Lucida Grande", "Verdana"]
    fig = plt.figure(figsize=(8, 3*len(params)), dpi=300)
    for i in range(len(params)):
        param = params[i]
        data = param["data"]
        ax = fig.add_subplot(len(params), 1, i+1)
        ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
        hours = mdates.HourLocator(byhour=range(0, 24, 1))
        ax.xaxis.set_major_locator(hours)
        ax.tick_params(axis="both", labelsize=12)
        for pname, ls, col, lw, lab in zip(param["pnames"], param["linestyles"], 
                                      param["colors"], param["linewidths"], param["labels"]):
            ax.plot(data.dtime, data[pname], ls=ls, lw=lw, color=col, label=r"$%s$"%lab)
        ax.legend(loc=2)
        ax.set_ylabel(param["ylabel"], fontdict={"size":12, "fontweight": "bold"})
        ax.set_ylim(param["ylim"])
        ax.set_xlim(param["xlim"])
        ax.axvline(dt.datetime(2012,6,16,9,56), ls="--", lw=0.8, color="k")
    ax.set_xlabel("Time [UT]", fontdict={"size":12, "fontweight": "bold"})
    plt.suptitle("WIND IMF-SW/ 16 June 2012", x=0.075, y=0.92, ha="left", fontweight="bold", fontsize=15)
    fig.savefig(fname, bbox_inches="tight")
    return fig

def get_gridded_parameters(q, xparam="time", yparam="slist", zparam="v", round=False):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[ [xparam, yparam, zparam] ]
    if round:
        plotParamDF[xparam] = np.round(plotParamDF[xparam], 2)
        plotParamDF[yparam] = np.round(plotParamDF[yparam], 2)
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


class RangeTimePlot(object):
    """
    Create plots for IS/GS flags, velocity, and algorithm clusters.
    """
    def __init__(self, nrang, unique_times, fig_title, num_subplots=3, font="sans-serif"):
        plt.rcParams["font.family"] = font
        if font == "sans-serif":
            plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans",
                                   "Lucida Grande", "Verdana"]
        dpi = 180 #if num_subplots==2 else 180 - ( 40*(num_subplots-2) )
        self.nrang = nrang
        self.unique_times = unique_times
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(8, 3*num_subplots), dpi=dpi) # Size for website
        plt.suptitle(fig_title, x=0.075, y=0.95, ha="left", fontweight="bold", fontsize=15)
        mpl.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize":12, "font.size":12})
        return
    
    def addParamPlot(self, df, beam, title, p_max=100, p_min=-100, xlabel="Time UT",
             ylabel="Range gate", zparam="v", label="Velocity [m/s]", cmap="jet_r", 
             cbar=False, omni=None, add_gflg=False):
        ax = self._add_axis()
        df = df[df.bmnum==beam]
        X, Y, Z = get_gridded_parameters(df, xparam="time", yparam="slist", zparam=zparam)
        if add_gflg:
            Xg, Yg, Zg = get_gridded_parameters(df, xparam="time", yparam="slist", zparam="gflg")
        cmap = cmap
        # cmap.set_bad("w", alpha=0.0)
        # Configure axes
        ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
        hours = mdates.HourLocator(byhour=range(0, 24, 1))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_xlim([self.unique_times[0], self.unique_times[-1]])
        ax.set_ylim(self.nrang)
        ax.set_ylabel(ylabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        if add_gflg:
            Zx = np.ma.masked_where(Zg==0, Zg)
            ax.pcolormesh(Xg, Yg, Zx.T, lw=0.01, edgecolors="None", cmap="gray",
                        vmax=2, vmin=0, shading="nearest")
            Z = np.ma.masked_where(Zg==1, Z)
            im = ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap,
                        vmax=p_max, vmin=p_min, shading="nearest")
        else:
            im = ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap,
                        vmax=p_max, vmin=p_min, shading="nearest")
        if omni is None: 
            if cbar: self._add_colorbar(im, ax, cmap, label=label)
            return ax
        else: 
            if cbar: self._add_colorbar(im, ax, cmap, label=label, dx=0.15)
            t_ax = self.overlay_omni(ax, omni)
            return ax, t_ax
    
    def overlay_omni(self, ax, omni):
        t_ax = ax.twinx()
        t_ax.set_xlim([self.unique_times[0], self.unique_times[-1]])
        t_ax.set_ylim(0, 360)
        hours = mdates.HourLocator(byhour=range(0, 24, 1))
        t_ax.xaxis.set_major_locator(hours)
        t_ax.set_ylabel(r"IMF Clock Angle ($\theta_c$)"+"\n[in degrees]", 
            fontdict={"size":12, "fontweight": "bold"})
        t_ax.plot(
            omni.date, omni.Tc, 
            ls="-", lw=1.6, color="m"
        )
        t_ax.plot(
            omni.date, omni.Tc, 
            ls="--", lw=0.6, color="k"
        )
        return t_ax

    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        ax.tick_params(axis="both", labelsize=12)
        return ax

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")

    def close(self):
        self.fig.clf()
        plt.close()

    def _add_colorbar(self, im, ax, colormap, label="", dx=0.01):
        """
        Add a colorbar to the right of an axis.
        :param fig:
        :param ax:
        :param colormap:
        :param label:
        :return:
        """
        import matplotlib as mpl
        pos = ax.get_position()
        cpos = [pos.x1 + pos.width * dx, pos.y0 + pos.height*.1,
                0.01, pos.height * 0.8]                # this list defines (left, bottom, width, height
        cax = self.fig.add_axes(cpos)
        cb2 = self.fig.colorbar(im, cax,
                   spacing="uniform",
                   orientation="vertical", 
                   cmap=colormap)
        cb2.set_label(label)
        return

class CartoBase(object):
    """
    This class holds cartobase code for the
    SD, SMag, and GPS TEC dataset.
    """

    def __init__(self, date, xPanels=1, yPanels=1):
        self.date = date
        self.xPanels = xPanels
        self.yPanels = yPanels
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(4*yPanels, 4*xPanels), dpi=240) # Size for website
        mpl.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize":12, "font.size":12})
        self.proj = {
            "to": ccrs.Orthographic(-110, 90),
            "from": ccrs.PlateCarree(),
        }
        plt.suptitle(date.strftime("%d %b %Y, %H:%M UT"), 
            x=0.5, y=0.99, ha="center", va="bottom", fontweight="bold", fontsize=15)
        return

    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(
            self.xPanels, self.yPanels, 
            self._num_subplots_created,
            projection=self.proj["to"],
        )
        ax.tick_params(axis="both", labelsize=12)
        ax.add_feature(Nightshade(self.date, alpha=0.3))
        ax.set_global()
        ax.coastlines(color="k", alpha=0.5, lw=0.5)
        gl = ax.gridlines(crs=self.proj["from"], linewidth=0.3, 
            color="k", alpha=0.5, linestyle="--", draw_labels=True)
        gl.xlocator = mticker.FixedLocator(np.arange(-180,180,30))
        gl.ylocator = mticker.FixedLocator(np.arange(-90,90,20))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        return ax

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return

    def _fetch_axis(self):
        if not hasattr(self, "ax"): self.ax = self._add_axis()
        return

    def add_radars(self, radars):
        self._fetch_axis()
        for r in radars.keys():
            rad = radars[r]
            self.overlay_radar(rad)
            self.overlay_fov(rad)
            self.ovrlay_radar_data(rad)
        return

    def add_magnetometers(self, mags):
        self._fetch_axis()
        mags = mags[mags.tval==self.date]
        for i, mag in mags.iterrows():
            self.overlay_magnetometer(mag)
        self.ovrlay_magnetometer_data(mags)
        return

    def overlay_magnetometer(
        self, mag, marker="D", zorder=2, 
        markerColor="k", markerSize=2
    ):
        lat, lon = mag.glat, mag.glon
        self.ax.scatter([lon], [lat], s=markerSize, marker=marker,
            color=markerColor, zorder=zorder, transform=self.proj["from"], lw=0.8, alpha=0.4)
        return

    def ovrlay_magnetometer_data(
        self, mags, scalef=300, lkey=100
    ):
        lat, lon = np.array(mags.glat), np.array(mags.glon)
        # Extract H-component
        N, E = np.array(mags.N_geo), np.array(mags.E_geo)
        xyz = self.proj["to"].transform_points(self.proj["from"], lon, lat)
        x, y = xyz[:, 0], xyz[:, 1]
        # Add Quiver for H-Component
        self.ax.scatter(x, y, color="k", s=2)
        ql = self.ax.quiver(
            x,
            y,
            E,
            N,
            scale=scalef,
            headaxislength=0,
            linewidth=0.0,
            scale_units="inches",
        )
        self.ax.quiverkey(
            ql,
            0.95,
            0.95,
            lkey,
            str(lkey) + " nT",
            labelpos="N",
            transform=self.proj["from"],
            color="r",
            fontproperties={"size": 12},
        )
        return

    def overlay_radar(
        self, rad, marker="D", zorder=2, markerColor="k", 
        markerSize=2, fontSize="small", font_color="k", xOffset=-5, 
        yOffset=-1.5, annotate=True,
    ):
        """ Adding the radar location """
        lat, lon = rad.hdw.geographic.lat, rad.hdw.geographic.lon
        self.ax.scatter([lon], [lat], s=markerSize, marker=marker,
            color=markerColor, zorder=zorder, transform=self.proj["from"], lw=0.8, alpha=0.4)
        nearby_rad = [["adw", "kod", "cve", "fhe", "wal", "gbr", "pyk", "aze", "sys"],
                    ["ade", "ksr", "cvw", "fhw", "bks", "sch", "sto", "azw", "sye"]]
        if annotate:
            rad = rad.rad
            if rad in nearby_rad[0]: xOff, ha = -5 if not xOffset else -xOffset, -2
            elif rad in nearby_rad[1]: xOff, ha = 5 if not xOffset else xOffset, -2
            else: xOff, ha = xOffset, -1
            x, y = self.proj["to"].transform_point(lon+xOff, lat+ha, src_crs=self.proj["from"])
            self.ax.text(x, y, rad.upper(), ha="center", va="center", transform=self.proj["to"],
                        fontdict={"color":font_color, "size":fontSize}, alpha=0.8)
        return

    def overlay_fov(
        self, rad, maxGate=75, rangeLimits=None, beamLimits=None,
        model="IS", fov_dir="front", fovColor=None, fovAlpha=0.2,
        fovObj=None, zorder=1, lineColor="k", lineWidth=1.0, ls="-"
    ):
        """ Overlay radar FoV """
        from numpy import transpose, ones, concatenate, vstack, shape
        hdw = rad.hdw
        sgate = 0
        egate = hdw.gates if not maxGate else maxGate
        ebeam = hdw.beams
        if beamLimits is not None: sbeam, ebeam = beamLimits[0], beamLimits[1]
        else: sbeam = 0
        latFull, lonFull = rad.fov[0].T, rad.fov[1].T
        xyz = self.proj["to"].transform_points(self.proj["from"], lonFull, latFull)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        contour_x = concatenate((x[sbeam, sgate:egate], x[sbeam:ebeam, egate],
                    x[ebeam, egate:sgate:-1],
                    x[ebeam:sbeam:-1, sgate]))
        contour_y = concatenate((y[sbeam, sgate:egate], y[sbeam:ebeam, egate],
                y[ebeam, egate:sgate:-1],
                y[ebeam:sbeam:-1, sgate]))
        self.ax.plot(contour_x, contour_y, color=lineColor, 
            zorder=zorder, linewidth=lineWidth, ls=ls, alpha=1.0)
        return

    def ovrlay_radar_data(self, rad):
        data = rad.df[
            (rad.df.time>=self.date) &
            (rad.df.time<self.date+dt.timedelta(minutes=1)) &
            (rad.df.slist<=75)
        ]
        kwargs = {"rad": rad}
        # add a function to create GLAT/GLON in Data
        data = data.apply(self.convert_to_latlon, axis=1, **kwargs)
        # Grid based on GLAT/GLON
        X, Y, Z = get_gridded_parameters(data, "glon", "glat", "v")
        #print(np.max(X), np.min(X), np.max(Y), np.min(Y))
        # Plot based on transcript
        xyz = self.proj["to"].transform_points(self.proj["from"], X, Y)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        im = self.ax.pcolor(
            x, y, Z.T,
            cmap="jet_r",
            vmin=-300,
            vmax=300,
            transform=self.proj["to"],
        )
        self._add_colorbar(im, label="Velocity, m/s")
        return

    def convert_to_latlon(self, row, rad):
        row["glat"], row["glon"] = (
            rad.fov[0][row["slist"], row["bmnum"]],
            rad.fov[1][row["slist"], row["bmnum"]],
        )
        return row
    
    def add_TEC(self, tec):
        print(tec.head())
        self._fetch_axis()
        X, Y, Z = get_gridded_parameters(tec, "glon", "gdlat", "tec", round=True)
        # Plot based on transcript
        xyz = self.proj["to"].transform_points(self.proj["from"], X, Y)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        im = self.ax.pcolor(
            x, y, Z.T,
            cmap="jet_r",
            vmin=3,
            vmax=15,
            transform=self.proj["to"],
        )
        self._add_colorbar(im, label="TEC, TECu")
        return

    def _add_colorbar(self, im, colormap="jet_r", label=""):
        """Add a colorbar to the right of an axis."""
        pos = self.ax.get_position()
        cpos = [
            pos.x1 + 0.09,
            pos.y0 + 0.2 * pos.height,
            0.02,
            pos.height * 0.5,
        ]  # this list defines (left, bottom, width, height)
        cax = self.fig.add_axes(cpos)
        cb2 = self.fig.colorbar(
            im,
            cax=cax,
            cmap=colormap,
            spacing="uniform",
            orientation="vertical",
        )
        cb2.set_label(label)
        return
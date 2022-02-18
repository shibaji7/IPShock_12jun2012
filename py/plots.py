import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize":12, "font.size":12})
import matplotlib.pyplot as plt
plt.style.use(["science", "ieee"])
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import numpy as np
import datetime as dt

from sd_carto import *



def grid(nlats=180, nlons=360):
    """A global grid of the specified size"""
    _lats = np.linspace(-90, 90, nlats + 1)
    _lons = np.linspace(-180, 180, nlons + 1)
    coords = np.empty((_lats.size, _lons.size, 3))
    coords[:,:,1], coords[:,:,0] = np.meshgrid(_lons, _lats)
    coords[:,:,2] = 0 # height above WGS84 in km
    return coords

def _plot_contours(ax, x, y, z, *args, **kwargs):
    transform_before_plotting = kwargs.pop("transform_before_plotting", False)
    if transform_before_plotting:
        # transform coordinates *before* passing them to the plotting function
        tmp = ax.projection.transform_points(ccrs.PlateCarree(), x, y)
        x_t, y_t = tmp[..., 0], tmp[..., 1]
        return ax.contour(x_t, y_t, z, linewidths=0.4, alpha=0.7, *args, **kwargs)
    else:
        # ... transformation performed by the plotting function creates glitches at the antemeridian
        kwargs["transform"] = ccrs.PlateCarree()
        return ax.contour(x, y, z, linewidths=0.4, alpha=0.7, *args, **kwargs)

def plot_contours(ax, x, y, z, *args, **kwargs):
    fmt = kwargs.pop("fmt", r"%d$^\circ$")
    fontsize = kwargs.pop("fontsize", 4)
    #ax.add_feature(cfeature.LAND, facecolor=(1.0, 1.0, 0.9))
    #ax.add_feature(cfeature.OCEAN, facecolor=(0.9, 1.0, 1.0))
    #ax.add_feature(cfeature.COASTLINE, edgecolor='silver')
    #ax.gridlines()
    levels = [ 5, 10, 15,   20, 30, 60, 90, 150]
            #np.linspace(10*(int(np.min(z)/10)-1), 10*(int(np.max(z)/10)+1), 11)
    cs = _plot_contours(ax, x, y, z, levels, *args, **kwargs)
    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=fontsize, colors="darkred")
    return

def anotate(ax, x, ys, xlab, ylab, ylim, xlim, cols, labels=[], tag="", xtick=False, 
            vl=dt.datetime(2012,6,16,9,56), mgn=None):
    labels = [ylab] if len(ys) == 1 else labels
    ax.set_ylabel(ylab, fontdict={"size":12})
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.xaxis.set_major_formatter(DateFormatter(r"{%H}^{%M}"))
    hours = mdates.MinuteLocator(range(0, 60, 30))
    ax.xaxis.set_major_locator(hours)
    for y, c, l in zip(ys, cols, labels):
        ax.plot(x, y, c+"D", ls="None", ms=1, label=l)
    if len(ys) > 1: ax.legend(loc=3)
    if tag: ax.text(0.1, 0.9, tag, ha="left", va="center", transform=ax.transAxes, fontdict={"size":12})
    if not xtick: ax.set_xticklabels([])
    ax.axvline(vl, ls="--", lw=0.8, color="darkred")
    if mgn:
        txt = "{}\n$\Lambda_m={}^\circ$\nMLT={}".format(mgn["name"], mgn["mlat"], mgn["mlt"])
        ax.text(0.8,0.95, txt, ha="center", va="top", transform=ax.transAxes, fontdict={"size":7})
    return ax

def mean(x, key, start=dt.datetime(2012,6,16,9), end=dt.datetime(2012,6,16,9,30)):
    x = x[(x.index>=start) & (x.index<=end)]
    return np.mean(x[key])

def create_figure2(imf, omni, mag, mags_name):
    fig, axes = plt.subplots(nrows=6, ncols=2, dpi=150, figsize=(9, 14))
    xlim = [dt.datetime(2012,6,16,8), dt.datetime(2012,6,16,10)]
    vl = dt.datetime(2012,6,16,9,3)
    anotate(axes[0,0], imf.time, [imf.B], xlab="", ylab="B(nT)",  ylim=[0,12], xlim=xlim, cols=["k"],
            tag="(a-1)", vl=vl)
    anotate(axes[1,0], imf.time, [imf.Bx, imf.By], xlab="", ylab=r"$\text{B}_{\text{x,y}}$(nT)", 
            ylim=[-20,20], vl=vl, xlim=xlim, cols=["k", "r"], 
            labels=[r"$\text{B}_\text{x}$", r"$\text{B}_{y}$"], tag="(a-2)")
    anotate(axes[2,0], imf.time, [imf.Bz], xlab="", ylab=r"$\text{B}_\text{z}$(nT)", ylim=[-20,20], 
            xlim=xlim, cols=["k"], tag="(a-3)", vl=vl)
    anotate(axes[3,0], omni.time, [omni.np], xlab="", ylab=r"$\text{N}_{\text{sw}}$(\#/cc)",  
            ylim=[0,15], xlim=xlim, cols=["k"], tag="(a-4)", vl=vl)
    anotate(axes[4,0], omni.time, [omni.vs], xlab="", ylab=r"$\text{V}_{\text{sw}}$(km/s)",  
            ylim=[200,500], xlim=xlim, cols=["k"], tag="(a-5)", vl=vl)
    anotate(axes[5,0], omni.time, [omni.pdy], xlab="", ylab=r"$\text{P}_{\text{dyn}}$(nT)", 
            ylim=[0,5], xlim=xlim, cols=["k"], tag="(a-6)", xtick=True, vl=vl)
    axes[5,0].set_xlabel("Time, UT", fontdict={"size":12})
    xlim = [dt.datetime(2012,6,16,9,30), dt.datetime(2012,6,16,10,30)]
    anotate(axes[0,1], mag[0].index.tolist(), [-(mag[0].Y-mean(mag[0], "Y"))], 
            xlab="", ylab=r"H (nT)", ylim=[-100,200], xlim=xlim, cols=["k"], tag="(b-1)", mgn=mags_name[0])
    anotate(axes[1,1], mag[1].index.tolist(), [-(mag[1].Y-mean(mag[1], "Y"))], 
            xlab="", ylab=r"H (nT)", ylim=[-50,50], xlim=xlim, cols=["k"], tag="(b-2)", mgn=mags_name[1])
    anotate(axes[2,1], mag[2].index.tolist(), [-(mag[2].Y-mean(mag[2], "Y"))], xlab="",
            ylab=r"H (nT)", ylim=[-10,10], xlim=xlim, cols=["k"], tag="(b-3)", mgn=mags_name[2])
    anotate(axes[3,1], mag[3].index.tolist(), [-(mag[3].Y-mean(mag[3], "Y"))], xlab="",
            ylab=r"H (nT)", ylim=[-10,10], xlim=xlim, cols=["k"], tag="(b-4)", mgn=mags_name[3])
    anotate(axes[4,1], mag[4].index.tolist(), [-(mag[4].Y-mean(mag[4], "Y"))], xlab="",
            ylab=r"H (nT)", ylim=[-10,10], xlim=xlim, cols=["k"], tag="(b-5)", mgn=mags_name[4])
    anotate(axes[5,1], mag[5].time, [mag[5].symh], xlab="UT", ylab=r"Sym-H (nT)",
            ylim=[-0,50], xlim=xlim, cols=["k"], tag="(b-6)", xtick=True)
    axes[5,1].set_xlabel("Time, UT", fontdict={"size":12})
    #fig.autofmt_xdate()
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig("figures/Figure2.png", bbox_inches="tight")
    return


def create_figure4(Zdvl):
    fig, axes = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(5, 3))
    xlim = [dt.datetime(2012,6,16,9,30), dt.datetime(2012,6,16,10,30)]
    anotate(axes, Zdvl.date, [Zdvl.x, Zdvl.y, Zdvl.z], xlab="Time, UT", ylab=r"Velocity, $m.s^{-1}$",
            ylim=[-500,500], xlim=xlim, cols=["k", "r", "b"], tag="", xtick=True,
            labels=[r"$\text{V}_\text{x}$", r"$\text{V}_{y}$", r"$\text{V}_{z}$"])
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig("figures/Figure4.png", bbox_inches="tight")
    return
    

def create_figure1():
    from sd_carto import FanPlots
    fov = FanPlots()
    ax = fov.add_axes(dt.datetime(2012,6,16,10))
    ax.overlay_radar("zho", xOffset=10,markerColor="k",yOffset=-5, font_color="m")
    ax.overlay_radar("zho", zorder=3, markerColor="darkblue", markerSize=70, marker="+", annotate=False)
    ax.overlay_fov("zho")
    ax.overlay_fov("zho", beamLimits=[7,8], ls="--")
    ax.overlay_radar("mcm", xOffset=-5,markerColor="k", yOffset=5, font_color="m")
    ax.overlay_fov("mcm")
    ax.overlay_fov("mcm", beamLimits=[4,5], ls="--")
    fov.save("figures/Figure1.png")
    return

def eval_model(time=dt.datetime(2016, 6, 16), coords=grid(),
               shc_model=None):
    """Evaluate a .shc model at a fixed time

    Args:
        time (datetime)
        coords (ndarray)
        shc_model (str): path to file

    Returns:
        dict: magnetic field vector components:
            https://intermagnet.github.io/faq/10.geomagnetic-comp.html
    """
    import eoxmagmod
    shc_model = eoxmagmod.data.CHAOS_CORE_LATEST
    # Convert Python datetime to MJD2000
    epoch = eoxmagmod.util.datetime_to_decimal_year(time)
    mjd2000 = eoxmagmod.decimal_year_to_mjd2000(epoch)
    # Load model
    model = eoxmagmod.load_model_shc(shc_model)
    # Evaluate in North, East, Up coordinates
    height = eoxmagmod.GEODETIC_ABOVE_WGS84
    b_neu = model.eval(mjd2000, coords, height, height)
    # Inclination (I), declination (D), intensity (F)
    inc, dec, F = eoxmagmod.vincdecnorm(b_neu)
    return {"X": b_neu[:,:,0], "Y": b_neu[:,:,1], "Z": -b_neu[:,:,2],
            "I": inc, "D":dec, "F":F}

def create_figure7():
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import eoxmagmod
    
    proj=ccrs.SouthPolarStereo()#ccrs.Orthographic(130, -90)
    fig = plt.figure(figsize=(2.5,2.5), dpi=300)
    ax = fig.add_subplot(111, projection="sdcarto",\
                         map_projection = proj,\
                         coords="geo", plot_date=dt.datetime(2012,6,16,10))
    ax.overlay_radar("zho", xOffset=3,markerColor="k",yOffset=-5, font_color="m", annotate=True, fontSize=4)
    ax.overlay_fov("zho")
    #ax.set_global()
    ax.overaly_coast_lakes(lw=0.5, alpha=0.7)
    ax.set_extent([-180, 180, -90, -30], crs=cartopy.crs.PlateCarree())
    plt_lons = np.arange( 0, 361, 30 )
    mark_lons = np.arange( 0, 360, 30 )
    plt_lats = np.arange(-90,0,10)
    gl = ax.gridlines(crs=cartopy.crs.Geodetic(), linewidth=0.3)
    gl.xlocator = mticker.FixedLocator(plt_lons)
    gl.ylocator = mticker.FixedLocator(plt_lats)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.n_steps = 90
    ax.mark_latitudes(plt_lats, fontsize="xx-small", color="k")
    ax.mark_longitudes(plt_lons, fontsize="xx-small", color="k")   
    
    t = dt.datetime(2016, 6, 16)
    shc_model = eoxmagmod.data.CHAOS_CORE_LATEST
    coords = grid()
    mag_components = eval_model(t, coords, shc_model)
    south = (coords[:, 0, 0] < 0).nonzero()[0]
    
    plot_contours(ax, coords[south, :, 1], coords[south, :, 0], np.abs(mag_components["D"][south, :]),
                  transform_before_plotting=True)
    
    
    fig.savefig("figures/Figure7.png", bbox_inches="tight")
    return


if __name__ == "__main__":
    create_figure7()
    create_figure1()
"""
    This python module is used to analyze and plot the dataset.
"""
from math import radians
import os
import datetime as dt
import pandas as pd
import pydarn
import numpy as np


from read_fitacf import Radar, fetch_omni, fetch_wind
from plot import RangeTimePlot, overlay_sw

def generate_location_file(rad="pgr", beams=[]):
    os.makedirs("figures/", exist_ok=True)
    hdw = pydarn.read_hdw_file(rad)
    lat, lon = pydarn.Coords.GEOGRAPHIC(hdw.stid)
    beams, gates = hdw.beams, hdw.gates
    rec = []
    for gate in range(lat.shape[0]):
        for beam in range(lat.shape[1]):
            rec.append({
                "beam": beam,
                "gate": gate,
                "gdlat": lat[gate, beam],
                "glong": lon[gate, beam]
            })
    rec = pd.DataFrame.from_records(rec)
    rec.to_csv(f"figures/{rad}.csv", header=True, index=False)
    return

def gsMapSlantRange(slant_range, altitude=None, elevation=None):
    """Calculate the ground scatter mapped slant range.
    See Bristow et al. [1994] for more details. (Needs full reference)
    Parameters
    ----------
    slant_range
        normal slant range [km]
    altitude : Optional[float]
        altitude [km] (defaults to 300 km)
    elevation : Optional[float]
        elevation angle [degree]
    Returns
    -------
    gsSlantRange
        ground scatter mapped slant range [km] (typically slightly less than
        0.5 * slant_range.  Will return -1 if
        (slant_range**2 / 4. - altitude**2) >= 0. This occurs when the scatter
        is too close and this model breaks down.
    """
    Re = 6731.1

    # Make sure you have altitude, because these 2 projection models rely on it
    if not elevation and not altitude:
        # Set default altitude to 300 km
        altitude = 300.0
    elif elevation and not altitude:
        # If you have elevation but not altitude, then you calculate altitude,
        # and elevation will be adjusted anyway
        altitude = np.sqrt(Re ** 2 + slant_range ** 2 + 2. * slant_range * Re *
                           np.sin(np.radians(elevation))) - Re
    if (slant_range**2) / 4. - altitude ** 2 >= 0:
        gsSlantRange = Re * \
            np.arcsin(np.sqrt(slant_range ** 2 / 4. - altitude ** 2) / Re)
        # From Bristow et al. [1994]
    else:
        gsSlantRange = -1
    return gsSlantRange

class Analysis(object):

    def __init__(
        self,
        rad, 
        beam, 
        dates, 
        type="fitacf",
        font="sans-sarif",
        rti_panels = 2
    ):
        self.rad = rad
        self.dates = dates
        self.beam = beam
        self.type = type
        self.font = font
        os.makedirs("figures/", exist_ok=True)
        self.radar = Radar(rad, dates, type=type)
        self.rti_panels = rti_panels
        self.iniRTI()
        return

    def iniRTI(
        self,
        range = [0,4000],
        title = None
    ):
        if title is None:
            tab = f"{self.dates[0].day} {self.dates[0].strftime('%b')}, {self.dates[0].year}"
            title = f"Rad: {self.rad} / Beam: {self.beam} / Date: {tab}"
        self.rti = rti = RangeTimePlot(
            range, 
            self.dates, 
            title, 
            self.rti_panels,
            font=self.font
        )
        return

    def generateRTI(
        self, 
        params = [
            {
                "col": "jet",
                "ylim": [800, 2000],
                "xlim": None,
                "title": "", 
                "p_max":30, "p_min":3, 
                "xlabel": "", "ylabel": "Slant Range [km]",
                "zparam":"p_l", "label": "Power [dB]",
                "cmap": "jet", "cbar": True,
            }
        ]        
    ):
        for i, param in enumerate(params):
            if "_gs" in param["zparam"]:
                param["zparam"] = param["zparam"].replace("_gs", "")
                self.df["slist"] = np.array(self.df["gsMap"])
            ax = self.rti.addParamPlot(
                self.df, 
                self.beam, param["title"], 
                p_max=param["p_max"], p_min=param["p_min"],
                xlabel=param["xlabel"], ylabel=param["ylabel"], 
                zparam=param["zparam"], label=param["label"],
                cmap=param["cmap"], cbar=param["cbar"]
            )
            ax.set_ylim(param["ylim"])
            if param["xlim"]: ax.set_xlim(param["xlim"])
            if i == 0:
                ax.text(0.99, 1.05, self.filter_summ, va="bottom",
                        ha="right", transform=ax.transAxes)
        return        
    
    def saveRTI(self):
        self.rti.save(f"figures/{self.rad}-{self.beam}.png")
        self.rti.close()
        return

    def filter_dataframe(
        self,
        gflg=None,
        tfreq=None,
    ):
        self.filter_summ = ""
        self.df = self.radar.df.copy()
        self.df.slist = (self.df.slist*self.df.rsep) + self.df.frang
        self.df = self.df[
            (self.df.time>=self.dates[0]) & 
            (self.df.time<=self.dates[-1])
        ]
        self.df["unique_tfreq"] = self.df.tfreq.apply(lambda x: int(x/0.5)*0.5)
        print(f"Unique tfreq: {self.df.unique_tfreq.unique()}")
        if tfreq: 
            self.df = self.df[self.df.unique_tfreq==tfreq]
            self.filter_summ += r"$f_0$=%.1f MHz"%tfreq + "\n"
        if gflg: 
            self.df = self.df[self.df.gflg==gflg]
            self.filter_summ += r"IS/GS$\sim$%d"%gflg
        self.df["gsMap"] = self.df.apply(
            lambda row: gsMapSlantRange(row["slist"]), 
            axis = 1
        )
        return
    
def draw_IMFSW():
    omni = fetch_omni()
    wind = fetch_wind()
    overlay_sw(
        params=[
            {
                "data": wind,
                "pnames": ["Bx", "By", "Bz"],
                "colors": ["r", "b", "k"],
                "linestyles": ["-", "-", "-"],
                "linewidths": [1., 1., 1.],
                "labels": ["B_x", "B_y", "B_z"],
                "ylabel": "IMF [nT]",
                "ylim": [-15, 15],
                "xlim": [dt.datetime(2012,6,16,8), dt.datetime(2012,6,16,12)]
            },
            {
                "data": wind,
                "pnames": ["V"],
                "colors": ["r"],
                "linestyles": ["-"],
                "linewidths": [1.],
                "labels": ["V"],
                "ylabel": "Velocity [km/s]",
                "ylim": [0, 700],
                "xlim": [dt.datetime(2012,6,16,8), dt.datetime(2012,6,16,12)]
            },
            {
                "data": wind,
                "pnames": ["Np"],
                "colors": ["b"],
                "linestyles": ["-"],
                "linewidths": [1.],
                "labels": ["N_p"],
                "ylabel": "\# [/cc]",
                "ylim": [0, 20],
                "xlim": [dt.datetime(2012,6,16,8), dt.datetime(2012,6,16,12)]
            },
        ],
        fname="figures/omni.png"
    )
    return

if __name__ == "__main__":
    draw_IMFSW()
    anl = Analysis(
        rad="zho",
        beam=1, 
        dates=[dt.datetime(2012,6,16,8), dt.datetime(2012,6,16,12)],
        type="fitacf",
        font="sans-serif",
        rti_panels = 1
    )
    anl.filter_dataframe()
    anl.generateRTI(
        params = [
            {
                "ylim": [0, 3000],
                "xlim": None,
                "title": "", 
                "p_max": 500, "p_min":-500, 
                "xlabel": "Time [UT]", "ylabel": "Slant Range [km]",
                "zparam":"v", "label": "Velocity [m/s]",
                "cmap": "jet", "cbar": True,
            }
        ]
    )
    anl.saveRTI()

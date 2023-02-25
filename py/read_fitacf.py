"""
    This python module is used to read the dataset from fitacf/fitacf3 
    level dataset.
"""

import os
import pandas as pd
import pydarn 
import glob
import bz2
from loguru import logger
import datetime as dt
import numpy as np
from scipy import constants as C

class Radar(object):

    def __init__(self, rad, dates=None, clean=False, type="fitacf",):
        logger.info(f"Initialize radar: {rad}")
        self.rad = rad
        self.dates = dates
        self.clean = clean
        self.type = type
        self.__setup__()
        self.__fetch_data__()
        self.calculate_decay_rate()
        return
    
    def __setup__(self):
        logger.info(f"Setup radar: {self.rad}")
        tag = f"/sd-data/{self.dates[0].year}/{self.type}/{self.rad}/{self.dates[0].strftime('%Y%m%d')}*"
        self.files = glob.glob(tag)
        self.files.sort()
        self.hdw = pydarn.read_hdw_file(self.rad)
        self.fov = pydarn.Coords.GEOGRAPHIC(self.hdw.stid)
        logger.info(f"Files: {len(self.files)}")
        return

    def __fetch_data__(self):
        if self.clean: os.remove(f"data/{self.rad}.{self.type}.csv")
        if os.path.exists(f"data/{self.rad}.{self.type}.csv"):
            self.df = pd.read_csv(f"data/{self.rad}.{self.type}.csv", parse_dates=["time"])
        else:
            records = []
            for f in self.files:
                logger.info(f"Reading file: {f}")
                with bz2.open(f) as fp:
                    reader = pydarn.SuperDARNRead(fp.read(), True)
                    records += reader.read_fitacf()
            if len(records)>0:
                self.__tocsv__(records)
        self.df.tfreq = np.round(np.array(self.df.tfreq)/1e3, 1)
        return

    def __tocsv__(self, records):
        time, v, slist, p_l, frang, scan, beam,\
            w_l, gflg, elv, phi0, tfreq, rsep = (
            [], [], [],
            [], [], [],
            [], [], [],
            [], [], [],
            [],
        )
        for r in records:
            if "v" in r.keys():
                t = dt.datetime(
                    r["time.yr"], 
                    r["time.mo"],
                    r["time.dy"],
                    r["time.hr"],
                    r["time.mt"],
                    r["time.sc"],
                    r["time.us"],
                )
                time.extend([t]*len(r["v"]))
                tfreq.extend([r["tfreq"]]*len(r["v"]))
                rsep.extend([r["rsep"]]*len(r["v"]))
                frang.extend([r["frang"]]*len(r["v"]))
                scan.extend([r["scan"]]*len(r["v"]))
                beam.extend([r["bmnum"]]*len(r["v"]))
                v.extend(r["v"])
                gflg.extend(r["gflg"])
                slist.extend(r["slist"])
                p_l.extend(r["p_l"])
                w_l.extend(r["w_l"])
                if "elv" in r.keys(): elv.extend(r["elv"])
                if "phi0" in r.keys(): phi0.extend(r["phi0"])                
            
        self.df = pd.DataFrame()
        self.df["v"] = v
        self.df["gflg"] = gflg
        self.df["slist"] = slist
        self.df["bmnum"] = beam
        self.df["p_l"] = p_l
        self.df["w_l"] = w_l
        if len(elv) > 0: self.df["elv"] = elv
        if len(phi0) > 0: self.df["phi0"] = phi0
        self.df["time"] = time
        self.df["tfreq"] = tfreq
        self.df["scan"] = scan
        self.df["rsep"] = rsep
        self.df["frang"] = frang

        if self.dates:
            self.df = self.df[
                (self.df.time>=self.dates[0]) & 
                (self.df.time<=self.dates[1])
            ]
        self.df.to_csv(f"data/{self.rad}.{self.type}.csv", index=False, header=True)
        return

    def recalculate_elv_angle(self, XOff=0, YOff=100, ZOff=0):
        return

    def calculate_decay_rate(self):
        logger.info(f"Calculate Decay")
        f, w_l = np.array(self.df.tfreq)*1e6, np.array(self.df.w_l)
        k = 2*np.pi*f/C.c
        self.df["tau_l"] = 1.e3/(k*w_l)
        return

    
def fetch_omni():
    o = []
    with open("data/omni.csv", "r") as f: lines = f.readlines()
    for line in lines:
        line = list(filter(None, line.replace("\n", "").split(" ")))
        time = dt.datetime(int(line[0]), 1, 1) + dt.timedelta(days=int(line[1])-1) +\
            dt.timedelta(hours=int(line[2])) + dt.timedelta(minutes=int(line[3]))
        o.append({
            "time": time,
            "B": float(line[4]),
            "Bx": float(line[5]),
            "By": float(line[6]),
            "Bz": float(line[7]),
            "V": float(line[8]),
            "Np": float(line[9]),
            "Tp": float(line[10]),
            "AE": float(line[11]),
            "AL": float(line[12]),
            "AU": float(line[13]),
            "SYMD": float(line[14]),
            "SYMH": float(line[15]),
            "ASYD": float(line[16]),
            "ASYH": float(line[17]),
        })
    o = pd.DataFrame.from_records(o)
    o.replace([99999.9, 999.99, 9999999.0, 9999.99], np.nan, inplace=True)
    return o

def fetch_wind():
    o = []
    with open("data/WIND.csv", "r") as f: lines = f.readlines()
    for line in lines:
        line = list(filter(None, line.replace("\n", "").split(" ")))
        time = dt.datetime(int(line[0]), 1, 1) + dt.timedelta(days=int(line[1])-1) +\
            dt.timedelta(seconds=int(line[2])/1e3)
        o.append({
            "time": time,
            "Bx": float(line[3]),
            "By": float(line[4]),
            "Bz": float(line[5]),
            "V": float(line[6]),
            "Np": float(line[7]),
        })
    o = pd.DataFrame.from_records(o)
    dsec = 1.5e6/450
    o["dtime"] = o.time.apply(lambda x: x + dt.timedelta(seconds=dsec))
    return o
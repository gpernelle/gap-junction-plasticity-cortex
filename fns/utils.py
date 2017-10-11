__author__ = 'G. Pernelle'

import matplotlib as mpl
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import re, csv, os, datetime, random, os, sys, sh, math, socket, time
from scipy.fftpack import fft
from matplotlib.pyplot import cm

from scipy import sparse, signal
import seaborn as sns
from cycler import cycler
import matplotlib as mpl
import gc
from tqdm import tnrange, trange
from attrdict import AttrDict

def update_mpl_settings():
    #Direct input
    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath, lmodern}"]
    #Options
    fontsize = 22
    params = {'text.usetex' : True,
              'font.family' : 'lmodern',
              'text.latex.unicode': True,
              'text.color':'black',
              'xtick.labelsize': fontsize-2,
              'ytick.labelsize': fontsize-2,
              'axes.labelsize': fontsize,
              'axes.labelweight': 'bold',
              'axes.edgecolor': 'white',
              'axes.titlesize': fontsize,
              'axes.titleweight': 'bold',
              'pdf.fonttype' : 42,
              'ps.fonttype' : 42,
              'axes.grid':False,
              'axes.facecolor':'white',
              'lines.linewidth': 1,
              "figure.figsize": '5,4',
              }
    plt.rcParams.update(params)

update_mpl_settings()

def load_config(filename='params'):
    import yaml
    # Read YAML file
    with open("params/%s.yaml"%filename, 'r') as stream:
        config = AttrDict(yaml.load(stream))
    return config


"""Utils for plotting."""
import os
import sys
from colorama import Fore
from colorama import Style
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def eprint(*args, **kwargs):
    print(Fore.RED, *args, Style.RESET_ALL, file=sys.stderr, **kwargs)


def debug(*args, **kwargs):
    print(Fore.YELLOW, *args, Style.RESET_ALL, file=sys.stderr, **kwargs)


# https://stackoverflow.com/questions/36190495/drawing-lines-on-scatter-with-seaborn
def plot_hline(y,**kwargs):
    data = kwargs.pop("data")
    plt.axhline(y=y, zorder=-1, **kwargs)


def parse_filename(filename, key):
    parts = os.path.split(filename)[0].split("/")[-1].split(",")
    for part in parts:
        k,v = part.split("=")
        if k == key:
            return v

    raise KeyError("legend key not found in filename")

import matplotlib.pyplot as plt
import os
from constants import *
import mne
from pathlib import Path


def psd_plots(rec_folder_name):
    rec_folder_path = os.path.join(RECORDINGS_DIR, rec_folder_name)
    folder_path = os.path.join(rec_folder_path, 'raw.fif')
    raw = mne.io.read_raw_fif(folder_path)
    folder_path_fig = os.path.join(rec_folder_path, "figures")
    Path(folder_path_fig).mkdir(exist_ok=True)

    fig_psd = mne.viz.plot_raw_psd(raw, fmax=HIGH_PASS, show=False)
    fig_psd.savefig(os.path.join(folder_path_fig, "psd.png"))
    fig_raw = mne.viz.plot_raw(raw)
    fig_raw.savefig(os.path.join(folder_path_fig, "raw_data.png"))
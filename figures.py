import matplotlib.pyplot as plt
import os
from constants import *
import mne
from pathlib import Path:wq





def psd_plots(rec_folder_name):
    folder_path = os.path.join(RECORDINGS_DIR, rec_folder_name, 'raw.fif')
    raw = mne.io.read_raw_fif(folder_path)
    folder_path_fig = os.path.join(RECORDINGS_DIR, rec_folder_name, "figures")
    raw.plot_psd(fmax=50)
    Path(folder_path_fig).mkdir(exist_ok=True)
    plt.savefig(folder_path_fig)  #dir + r'\psd_plot.png')
    plt.close()


psd_plots("2021-12-19--21-18-28_David")
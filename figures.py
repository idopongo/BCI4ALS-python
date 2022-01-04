import os
from constants import *
import mne
from pathlib import Path


def save_plots(rec_folder_name):
    rec_folder_path = os.path.join(RECORDINGS_DIR, rec_folder_name)
    raw_path = os.path.join(rec_folder_path, 'raw.fif')
    raw = mne.io.read_raw_fif(raw_path)
    fig_path = os.path.join(rec_folder_path, "figures")
    Path(fig_path).mkdir(exist_ok=True)

    fig_psd = mne.viz.plot_raw_psd(raw, fmax=HIGH_PASS, show=False)
    fig_psd.savefig(os.path.join(fig_path, "psd.png"))
    fig_raw = mne.viz.plot_raw(raw, show=False)
    fig_raw.savefig(os.path.join(fig_path, "raw_data.png"))


if __name__ == "__main__":
    save_plots("2021-12-29--17-52-11_Ido")

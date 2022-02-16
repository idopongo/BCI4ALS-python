import os
from constants import *
import mne
from pathlib import Path
from Marker import Marker
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import json
from preprocessing import preprocess
from pipeline import get_subject_rec_folders


def save_plots(rec_folder_name, bad_electrodes=[]):
    raw, rec_params = load_raw(rec_folder_name)
    raw = preprocess(raw)
    raw.info['bads'] = bad_electrodes

    fig_path = create_figures_folder(rec_folder_name)

    fig_raw = create_raw_fig(raw)
    fig_raw.savefig(os.path.join(fig_path, "raw.png"))

    fig_psd = create_psd_fig(raw)
    fig_psd.savefig(os.path.join(fig_path, "psd.png"))

    electrodes = ["C3", "C4", "Cz"]
    class_spectrogram_fig = create_class_spectrogram_fig(raw, rec_params, electrodes)
    class_spectrogram_fig.savefig(os.path.join(fig_path, f'class_spectrogram_{"_".join(electrodes)}.png'))


def create_psd_fig(raw):
    fig = mne.viz.plot_raw_psd(raw, fmin=7, fmax=30, show=False)
    return fig


def create_raw_fig(raw):
    events = mne.find_events(raw)
    event_dict = {marker.name: marker.value for marker in Marker}
    fig = mne.viz.plot_raw(raw, events=events, clipping=None, show=False, event_id=event_dict)
    return fig


def create_figures_folder(rec_folder_name):
    rec_folder_path = os.path.join(RECORDINGS_DIR, rec_folder_name)
    fig_path = os.path.join(rec_folder_path, "figures")
    Path(fig_path).mkdir(exist_ok=True)
    return fig_path


def load_raw(rec_folder_name):
    rec_folder_path = os.path.join(RECORDINGS_DIR, rec_folder_name)
    raw_path = os.path.join(rec_folder_path, 'raw.fif')
    raw = mne.io.read_raw_fif(raw_path)
    with open(os.path.join(rec_folder_path, 'params.json')) as file:
        rec_params = json.load(file)
    return raw, rec_params


def calc_class_spectrogram(raw, rec_params, cls_marker, chan, time_before_stim):
    events = mne.find_events(raw)
    epochs = mne.Epochs(raw, events, Marker.all(), tmin=-time_before_stim, tmax=rec_params["trial_duration"],
                        picks="data")
    cls_epochs = epochs[str(cls_marker.value)].load_data().pick([chan])

    segments_per_second = 2
    nperseg = int(FS / segments_per_second)
    noverlap = nperseg * 0.3
    nfft = 256
    freq_range = (2, 40)

    _, _, total_pow = signal.spectrogram(cls_epochs.next().squeeze(), FS, nperseg=nperseg, scaling="density",
                                         nfft=nfft, noverlap=noverlap)
    for epoch in cls_epochs[1:]:
        data = epoch.squeeze()
        freq, time, power = signal.spectrogram(data, FS, nperseg=nperseg, nfft=nfft, scaling="density",
                                               noverlap=noverlap)
        total_pow = total_pow + power

    avg_power = total_pow / len(cls_epochs)
    freq_idxs = (freq >= freq_range[0]) & (freq <= freq_range[1])
    freq = freq[freq_idxs]
    avg_power = avg_power[freq_idxs]
    return 10 * np.log10(avg_power), freq, time


def create_class_spectrogram_fig(raw, rec_params, electrodes):
    chans = [EEG_CHAN_NAMES.index(elec) for elec in electrodes]
    time_before_stim = 1
    fig, axs = plt.subplots(len(chans), len(Marker.all()), figsize=(22, 11))
    for i, chan in enumerate(chans):
        for j, cls in enumerate(Marker):
            power, freq, time = calc_class_spectrogram(raw, rec_params, cls, chan, time_before_stim)
            ax = axs[i, j]
            mesh = ax.pcolormesh(time, freq, power, shading='auto', cmap="jet", )
            plt.colorbar(mesh, ax=ax)
            ax.set_xlabel('Time [sec]')
            ax.set_ylabel('Frequency [Hz]')
            ax.axvline(time_before_stim, color='r', linestyle="dashed", label="stimulus")
            ax.legend()
            ax.set_title(f'{cls.name} {EEG_CHAN_NAMES[chan]}')
    fig.tight_layout()
    return fig


def save_plots_for_subject(subject_name):
    rec_folders = get_subject_rec_folders(subject_name)
    [save_plots(folder) for folder in rec_folders]


if __name__ == "__main__":
    save_plots_for_subject("Haggai")

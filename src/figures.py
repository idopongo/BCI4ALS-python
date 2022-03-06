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
from data_utils import get_subject_rec_folders
from spectral import Preprocessor
from pipeline import get_epochs


def create_and_save_plots(rec_folder_name, bad_electrodes=[]):
    raw, rec_params = load_raw(rec_folder_name)
    raw = preprocess(raw)
    raw.info['bads'] = bad_electrodes

    fig_path = create_figures_folder(rec_folder_name)

    fig_raw = create_raw_fig(raw)
    fig_raw.savefig(os.path.join(fig_path, "raw.png"))

    fig_psd = create_psd_fig(raw)
    fig_psd.savefig(os.path.join(fig_path, "psd.png"))

    epochs, _ = get_epochs(rec_params["trial_duration"], rec_params["calibration_duration"])

    electrodes = ["C3", "C4", "Cz"]
    class_spectrogram_fig = create_class_spectrogram_fig(epochs, rec_params["trial_duration"], electrodes)
    class_spectrogram_fig.savefig(os.path.join(fig_path, f'class_spectrogram_{"_".join(electrodes)}.png'))

    class_psd_fig = create_class_psd_fig(epochs, rec_params["trial_duration"], electrodes)
    class_psd_fig.savefig(os.path.join(fig_path, f'class_psd_{"_".join(electrodes)}.png'))


def create_psd_fig(raw):
    fig = mne.viz.plot_raw_psd(raw, fmin=7, fmax=30, show=False)
    return fig


def create_raw_fig(raw):
    events = mne.find_events(raw)
    event_dict = {marker.name: marker.value for marker in Marker}
    fig = mne.viz.plot_raw(raw, events=events, clipping=None, show=False, event_id=event_dict, show_scrollbars=False,
                           start=10)
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


def calc_class_spectrogram(epochs, cls_marker, chan):
    cls_epochs = mne.Epochs(raw, mne.find_events(raw), cls_marker, tmin=-time_before_stim, tmax=trial_duration,
                            picks=[chan])
    cls_epochs = cls_epochs.get_data().squeeze(axis=1)

    sfreq = raw.info['sfreq']
    segments_per_second = 4
    fft_params = {
        "nperseg": int(sfreq / segments_per_second),
        "noverlap": int(sfreq / segments_per_second) * 0.5,
        "nfft": 512,
        "scaling": "density"
    }

    # we calculate the power for the first epoch separately so that we have a variable of the right dimensions to sum
    # onto
    freq, time, avg_power = signal.spectrogram(cls_epochs[0], sfreq, **fft_params)
    for epoch in cls_epochs[1:]:
        _, _, power = signal.spectrogram(epoch, sfreq, **fft_params)
        avg_power += power / len(cls_epochs)

    freq_range = (2, 40)
    freq_idxs = (freq >= freq_range[0]) & (freq <= freq_range[1])
    freq = freq[freq_idxs]
    avg_power = avg_power[freq_idxs]
    return avg_power, freq, time


def calc_class_psd(raw, trial_duration, cls_marker, chan):
    cls_epochs = mne.Epochs(raw, mne.find_events(raw), cls_marker, tmax=trial_duration, picks=[chan])
    cls_epochs = cls_epochs.get_data().squeeze(axis=1)
    sfreq = raw.info['sfreq']

    fft_params = {
        'scaling': "density",
        'nfft': 512,
        'nperseg': 125,
        'noverlap': 60
    }

    # calculate the first fft
    freq, avg_power = signal.welch(cls_epochs[0], sfreq, **fft_params)

    for epoch in cls_epochs[1:]:
        _, power = signal.welch(epoch, sfreq, **fft_params)
        avg_power += power / len(cls_epochs)

    freq_range = (7, 30)
    freq_idxs = (freq >= freq_range[0]) & (freq <= freq_range[1])
    freq = freq[freq_idxs]
    avg_pxx = avg_power[freq_idxs]
    return avg_pxx, freq


def create_class_psd_fig(raw, trial_duration, electrodes):
    chans = [raw.info.ch_names.index(elec) for elec in electrodes]
    fig, axs = plt.subplots(len(chans), len(Marker.all()), figsize=(22, 11))
    for i, chan in enumerate(chans):
        for j, cls in enumerate(Marker):
            power, freq = calc_class_psd(raw, trial_duration, cls, chan)
            ax = axs[i, j]
            ax.semilogy(freq, 10 * power)
            ax.set_ylabel('Power')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_title(f'{cls.name} {raw.info.ch_names[chan]}')
    fig.tight_layout()
    return fig


def create_class_spectrogram_fig(epochs, trial_duration, electrodes):
    chans = [raw.info.ch_names.index(elec) for elec in electrodes]
    time_before_stim = 1
    fig, axs = plt.subplots(len(chans), len(Marker.all()), figsize=(22, 11))
    for i, elec in enumerate(electrodes):
        for j, cls in enumerate(Marker):
            power, freq, time = calc_class_spectrogram(epochs, trial_duration, cls, chan, time_before_stim)
            ax = axs[i, j]
            mesh = ax.pcolormesh(time, freq, 10 * np.log10(power), shading='auto', cmap="jet", )
            plt.colorbar(mesh, ax=ax)
            ax.set_xlabel('Time [sec]')
            ax.set_ylabel('Frequency [Hz]')
            ax.axvline(time_before_stim, color='r', linestyle="dashed", label="stimulus")
            ax.legend()
            ax.set_title(f'{cls.name} {raw.info.ch_names[chan]}')
    fig.tight_layout()
    return fig


def save_plots_for_subject(subject_name):
    rec_folders = get_subject_rec_folders(subject_name)
    [create_and_save_plots(folder) for folder in rec_folders]


if __name__ == "__main__":
    save_plots_for_subject("David7")

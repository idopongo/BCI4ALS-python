import os
from constants import *
import mne
from pathlib import Path
from Marker import Marker
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import json

def save_plots(rec_folder_name):
    raw, rec_params = load_raw(rec_folder_name)
    fig_path = create_figures_folder(rec_folder_name)

    fig_psd = create_psd_fig(raw)
    fig_psd.savefig(os.path.join(fig_path, "psd.png"))

    fig_raw = create_raw_fig(raw)
    fig_raw.savefig(os.path.join(fig_path, "raw.png"))

    chan = 3
    for cls_marker in Marker:
        fig_left_spectogram = create_class_spectrogram_fig(raw, rec_params, cls_marker, chan)
        fig_left_spectogram.savefig(os.path.join(fig_path, f'{cls_marker.name}_{EEG_CHAN_NAMES[chan]}_spectrogram.png'))


def create_psd_fig(raw):
    return mne.viz.plot_raw_psd(raw, fmax=HIGH_PASS, show=False)

def create_raw_fig(raw):
    return mne.viz.plot_raw(raw, show=False)


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

def create_class_spectrogram_fig(raw, rec_params, cls_marker, chan):
    events = mne.find_events(raw)
    time_before_stim = 1
    epochs = mne.Epochs(raw, events, Marker.all(), tmin=-time_before_stim, tmax=rec_params["trial_duration"], picks="data")
    cls_epochs = epochs[str(cls_marker.value)].load_data().pick([chan])

    nperseg = int(125/5)
    range = (2, 40)

    _, _, total_pow = signal.spectrogram(cls_epochs.next().squeeze(), FS, nperseg=nperseg, scaling="spectrum")
    for epoch in cls_epochs[1:]:
        data = epoch.squeeze()
        freq, time, power = signal.spectrogram(data, FS, nperseg=nperseg, scaling="spectrum")
        total_pow = total_pow + power

    avg_power = total_pow/len(cls_epochs)
    freq_idxs = (freq >= range[0]) & (freq <= range[1])
    freq = freq[freq_idxs]
    avg_power = avg_power[freq_idxs]

    fig = plt.figure()
    mesh = plt.pcolormesh(time, freq, avg_power, shading='auto', cmap="jet")
    plt.colorbar(mesh)
    plt.axvline(x=time_before_stim, color='r', linestyle='--', label="stimulus")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.legend()
    plt.title(f'Average spectrogram\n class: {cls_marker.name}, electrode: {EEG_CHAN_NAMES[chan]}')
    return fig

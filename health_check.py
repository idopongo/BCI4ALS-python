from board import Board
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import scipy.signal
from scipy import signal
import scipy.stats
import mne


def health_check():
    window_size = 2
    with Board(use_synthetic=True) as board:
        plt.ion()
        ax = create_figure(len(board.eeg_channels))
        chan_plots = plot_chans(board.channel_names, window_size, ax)
        montage_plot, chan_error_texts = plot_montage(board.channel_names, ax["upright"])
        psd_plot = plot_psd(ax["downright"])
        while True:
            data = get_next_data(board, window_size)
            fs = board.sfreq
            errors_by_chan = check_chan_health(data)
            update_chan_plots(chan_plots, data, window_size)
            update_montage_plot(ax, montage_plot, errors_by_chan, chan_error_texts)
            update_psd_plot(ax["downright"], psd_plot, data, fs)
            plt.draw()
            plt.pause(1e-3)


def plot_psd(ax):
    ax.set_ylabel('Power')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_title(f'power spectrum')
    ax.grid(True)
    return ax


def update_psd_plot(ax, psd_plot, data, sfreq):
    # psd_plot.figure(clear=True)
    for i in range(len(data)):
        freq, power = signal.welch(data[i], sfreq, scaling="density")
        ax.plot(freq, power)
        ax.psd(data[i], Fs=sfreq, scale_by_freq=True)

    # ax.psd(data[0], Fs=sfreq, scale_by_freq=True)



def on_press(event):
    if event.key == 'escape':
        raise SystemExit


def create_figure(num_chans):
    scale_num = 2/3
    fig, ax = plt.subplot_mosaic([[i, "upright"] if 0 <= i < scale_num * num_chans else [i, "downright"] for i in range(num_chans)])
    fig.subplots_adjust(left=0.1)
    fig.canvas.mpl_connect('key_press_event', on_press)
    return ax


def plot_chans(chan_names, window_size, ax):
    chan_plots = []
    t = np.zeros(0)
    V = np.zeros(0)
    for i, chan_name in enumerate(chan_names):
        chan_plots.append(ax[i].plot(t, V)[0])
        ax[i].set_ylim(-50, 50)
        ax[i].set_xlim(0, window_size)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].text(-0.1, 0.5, f'{str(chan_name)} ({str(i + 1)})', horizontalalignment='center',
                   verticalalignment='center', transform=ax[i].transAxes)
    return chan_plots


def update_chan_plots(chan_plots, data, window_size):
    for plot, chan in zip(chan_plots, data):
        t = np.linspace(0, window_size, len(chan))
        plot.set_data(t, chan)


def plot_montage(ch_names, ax):
    scale_num = 2/3
    # Get channel positions from standard_1020 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_pos = {key: value*scale_num for key, value in montage.get_positions()["ch_pos"].items() if key in ch_names}
    x = [scale_num*pos[0] for pos in ch_pos.values()]
    y = [scale_num*pos[1] for pos in ch_pos.values()]
    ax.axis('off')
    montage_plot = ax.scatter(x, y, 600, "white", edgecolor="black")
    chan_error_texts = []
    for ch_name, pos in ch_pos.items():
        ax.text(scale_num*pos[0], scale_num*pos[1], ch_name, ha="center", va="center")
        chan_error_texts.append(ax.text(scale_num*pos[0], scale_num*pos[1], "errors: ", ha="center", va="center"))
        ax.set_xticks([])
        ax.set_yticks([])
    return montage_plot, chan_error_texts


def update_montage_plot(ax, montage_plot, errors_by_chan, chan_error_texts):
    colors = ["red" if len(errors) else "black" for errors in errors_by_chan.values()]
    montage_plot.set_edgecolor(colors)

    for chan, errors in errors_by_chan.items():
        chan_error_text = chan_error_texts[chan]
        chan_error_text.set_text("\n".join(errors))


def get_next_data(board, window_size):
    channels = board.eeg_channels
    n_samples = board.sfreq * window_size
    data = board.brainflow_board.get_current_board_data(n_samples)
    data = data[channels]
    for i, chan in enumerate(data):
        data[i] = chan - np.mean(chan)  # DC Offset
    return data


def check_chan_health(data):
    errors_by_chan = {}
    for i, chan in enumerate(data):
        if len(chan):
            avg_corr = get_average_corr(chan, data)
            max = np.max(np.abs(chan))
            errors = []
            if np.abs(avg_corr) < 0.05:
                errors.append("avg corr too low")
            if np.abs(avg_corr) > 0.9:
                errors.append("avg corr too high")
            if max > 300:
                errors.append("amplitude too high")
            if max < 10:
                errors.append("amplitude too low")
            errors_by_chan[i] = errors
    return errors_by_chan


def get_average_corr(chan, all_chans):
    total_corr = 0
    for other_chan in all_chans:
        if len(chan) != len(other_chan) or len(chan) < 2 or len(other_chan) < 2:
            continue
        corr, _ = scipy.stats.pearsonr(chan, other_chan)
        if not np.isnan(corr):
            total_corr += corr
    avg_corr = total_corr / len(all_chans)
    return avg_corr


health_check()

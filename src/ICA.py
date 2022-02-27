from data_utils import load_recordings
import mne
from pipeline import get_epochs
from mne.preprocessing import ICA
from Marker import Marker
import matplotlib.pyplot as plt

raw, rec_params = load_recordings("David7")
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, Marker.all(), tmin=0, tmax=rec_params["trial_duration"], picks="data",
                    baseline=(0, 0))

epochs.load_data()
epochs.filter(7, 30)
ica = ICA(n_components=10, max_iter='auto', random_state=97)
ica.fit(epochs)
ica.plot_sources(epochs, show_scrollbars=False)
plt.show()
print()

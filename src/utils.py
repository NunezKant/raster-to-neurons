"""
utility functions for the analysis of the data
"""
import os
import sys
from scipy import io

sys.path.insert(0, r"C:\Users\labadmin\Documents\suite2p")
sys.path.insert(0, r"C:\Users\labadmin\Documents\rastermap")
from sklearn.decomposition import PCA
import numpy as np
from suite2p.extraction import dcnv
from rastermap import mapping
from scipy import ndimage
from tqdm import tqdm
from dataclasses import dataclass
import os
from pathlib import Path
import h5py
from scipy.interpolate import interp1d
import pandas as pd
from datetime import datetime, timedelta

##### SUITE2P FUNCTIONS #####


def deconvolve(root, ops):
    """
    Correct the lags of the dcnv data.

    Parameters
    ----------
    root : str
        Path to the experiment.
    ops : dict
        suite2p pipeline options
    """

    # we initialize empty variables
    spks = np.zeros(
        (0, ops["nframes"]), np.float32
    )  # the neural data will be Nneurons by Nframes.
    stat = np.zeros((0,))  # these are the per-neuron stats returned by suite2p
    xpos, ypos = np.zeros((0,)), np.zeros((0,))  # these are the neurons' 2D coordinates

    # this is for channels / 2-plane mesoscope
    tlags = 0.25 + np.linspace(0.2, -0.8, ops["nplanes"] // 2 + 1)[:-1]
    tlags = np.hstack((tlags, tlags))

    # loop over planes and concatenate
    iplane = np.zeros((0,))

    th_low, th_high = 0.5, 1.1
    for n in range(ops["nplanes"]):
        ops = np.load(
            os.path.join(root, "suite2p", "plane%d" % n, "ops.npy"), allow_pickle=True
        ).item()

        # load and deconvolve
        iscell = np.load(os.path.join(root, "suite2p", "plane%d" % n, "iscell.npy"))[
            :, 1
        ]
        iscell = (iscell > th_low) * (iscell < th_high)

        stat0 = np.load(
            os.path.join(root, "suite2p", "plane%d" % n, "stat.npy"), allow_pickle=True
        )[iscell]
        ypos0 = np.array(
            [stat0[n]["med"][0] for n in range(len(stat0))]
        )  # notice the python list comprehension [X(n) for n in range(N)]
        xpos0 = np.array([stat0[n]["med"][1] for n in range(len(stat0))])

        ypos0 += ops["dy"]  # add the per plane offsets (dy,dx)
        xpos0 += ops["dx"]  # add the per plane offsets (dy,dx)

        f_0 = np.load(os.path.join(root, "suite2p", "plane%d" % n, "F.npy"))[iscell]
        f_neu0 = np.load(os.path.join(root, "suite2p", "plane%d" % n, "Fneu.npy"))[
            iscell
        ]
        f_0 = f_0 - 0.7 * f_neu0

        # compute spks0 with deconvolution
        if tlags[n] < 0:
            f_0[:, 1:] = (1 + tlags[n]) * f_0[:, 1:] + (-tlags[n]) * f_0[:, :-1]
        else:
            f_0[:, :-1] = (1 - tlags[n]) * f_0[:, :-1] + tlags[n] * f_0[:, 1:]

        f_0 = dcnv.preprocess(
            f_0.copy(),
            ops["baseline"],
            ops["win_baseline"],
            ops["sig_baseline"],
            ops["fs"],
            ops["prctile_baseline"],
        )
        spks0 = dcnv.oasis(f_0, ops["batch_size"], ops["tau"], ops["fs"])

        spks0 = spks0.astype("float32")
        iplane = np.concatenate(
            (
                iplane,
                n
                * np.ones(
                    len(stat0),
                ),
            )
        )
        stat = np.concatenate((stat, stat0), axis=0)
        if spks.shape[1] > spks0.shape[0]:
            spks0 = np.concatenate(
                (
                    spks0,
                    np.zeros(
                        (spks0.shape[0], spks.shape[1] - spks0.shape[1]), "float32"
                    ),
                ),
                axis=1,
            )
        spks = np.concatenate((spks, spks0), axis=0)
        ypos = np.concatenate((ypos, ypos0), axis=0)
        xpos = np.concatenate((xpos, xpos0), axis=0)

        print(f"plane {n}, neurons: {len(xpos0)}")

    print(f"total neurons: {len(spks)}")

    xpos = xpos / 0.75
    ypos = ypos / 0.5

    return spks, stat, xpos, ypos, iplane


def baselining(ops, tlag, F, Fneu):
    """
    Baseline the neural data before deconvolution

    Parameters:
    ----------
    ops : dict
        Dictionary with the experiment info
    tlag : int
        Time lag for the deconvolution
    F : array
        Deconvolved fluorescence
    Fneu : array
        Neurophil fluorescence
    Returns:
    ----------
    F : array
        Baselined deconvolved fluorescence
    """
    F = preprocess(F, Fneu, ops["win_baseline"], ops["sig_baseline"], ops["fs"])
    # F = dcnv.preprocess(F, ops['baseline'], ops['win_baseline'], ops['sig_baseline'],
    #                   ops['fs'], ops['prctile_baseline'])
    if tlag < 0:
        F[:, 1:] = (1 + tlag) * F[:, 1:] + (-tlag) * F[:, :-1]
    else:
        F[:, :-1] = (1 - tlag) * F[:, :-1] + tlag * F[:, 1:]
    return F


def preprocess(F, Fneu, win_baseline, sig_baseline, fs):
    """
    Preprocess the fluorescence data

    Parameters:
    ----------
    F : array
        Deconvolved fluorescence
    Fneu : array
        Neurophil fluorescence
    baseline : int
        Baseline for the fluorescence
    win_baseline : int
        Window for the baseline
    sig_baseline : int
        Sigma for the baseline
    fs : int
        Sampling rate
    Returns:
    ----------
    F : array
        Preprocessed deconvolved fluorescence
    """
    win = int(win_baseline * fs)

    Flow = ndimage.gaussian_filter(F, [0.0, sig_baseline])
    Flow = ndimage.minimum_filter1d(Flow, win)
    Flow = ndimage.maximum_filter1d(Flow, win)
    F = F - 0.7 * Fneu
    Flow2 = ndimage.gaussian_filter(F, [0.0, sig_baseline])
    Flow2 = ndimage.minimum_filter1d(Flow2, win)
    Flow2 = ndimage.maximum_filter1d(Flow2, win)

    Fdiv = np.maximum(10, Flow.mean(1))
    F = (F - Flow2) / Fdiv[:, np.newaxis]

    return F


### BEHAVIOR FUNCTIONS ###


def add_exp(database, mname, expdate, blk):
    """
    Add an experiment to the database.

    Parameters
    ----------
    db : list
        List of experiments.
    mname : str
        Mouse name.
    expdate : str
        Experiment date.
    blk : str
        Block number.

    Returns
    -------
    db : list
        Updated List of experiments.
    """
    database.append({"mname": mname, "datexp": expdate, "blk": blk})
    return database  # Return the updated list


def get_trial_categories(rewarded_trial_structure, new_trial_structure):
    """
    Compute the trial categories for the new trial structure

    Parameters
    ----------
    rewarded_trial_structure : array
        vector of the rewarded trials.
    new_trial_structure : array
        vector with new exemplar trials.

    Returns
    -------
    trial_categories : list
        List of the trial categories.
    trial_counts : dict
        Dictionary with the trial categories counts.

    """
    rewarded_trial_structure = np.array(rewarded_trial_structure)
    new_trial_structure = np.array(new_trial_structure)
    trial_categories = [None] * len(rewarded_trial_structure)
    rewarded_new_counter = 0
    rewarded_counter = 0
    non_rewarded_counter = 0
    non_rewarded_new_counter = 0

    for idx in range(new_trial_structure.shape[0]):
        if np.logical_and(rewarded_trial_structure[idx], new_trial_structure[idx]):
            trial_categories[idx] = "rewarded test"
            rewarded_new_counter += 1
        elif np.logical_and(
            rewarded_trial_structure[idx], np.logical_not(new_trial_structure[idx])
        ):
            trial_categories[idx] = "rewarded"
            rewarded_counter += 1
        elif np.logical_and(
            np.logical_not(rewarded_trial_structure[idx]), new_trial_structure[idx]
        ):
            trial_categories[idx] = "non rewarded test"
            non_rewarded_new_counter += 1
        elif np.logical_and(
            np.logical_not(rewarded_trial_structure[idx]),
            np.logical_not(new_trial_structure[idx]),
        ):
            trial_categories[idx] = "non rewarded"
            non_rewarded_counter += 1

    trial_counts = {
        "rewarded test": rewarded_new_counter,
        "rewarded": rewarded_counter,
        "non rewarded test": non_rewarded_new_counter,
        "non rewarded": non_rewarded_counter,
    }

    return np.array(trial_categories), trial_counts


##### MOUSE DATACLASS #####


def methods(MouseObject):
    """
    List all the methods of a MouseObject
    ----------
    MouseObject : object
    Returns
    -------
    methods : list
    """
    import inspect

    methods = inspect.getmembers(MouseObject, predicate=inspect.ismethod)
    m = [method[0] for method in methods if not method[0].startswith("__")]
    return m


def properties(MouseObject):
    """
    List all the properties of a MouseObject
    ----------
    MouseObject : object
    Returns
    -------
    properties : list
    """
    return MouseObject.__dict__.keys()


def get_trial_per_frame(MouseObject):
    init_frames, _, last_frame = get_init_frames_per_category(MouseObject)
    n_frames = MouseObject._spks.shape[1]
    trial_per_frame = np.empty(n_frames, dtype=float)
    trial_per_frame[:] = np.nan
    frames = np.sort(np.concatenate(init_frames))
    for i, init_frame in enumerate(frames):
        if init_frame == last_frame:
            trial_per_frame[init_frame:] = int(i + 1)
        else:
            trial_per_frame[init_frame : frames[i + 1]] = int(i + 1)
    return trial_per_frame


def get_movement_df(MouseObject):
    property_set = set(properties(MouseObject))
    assert {"_settings", "_timeline"}.issubset(
        property_set
    ), "self._settings and self._timeline not defined, make sure to use self.load_behav() first"
    data = MouseObject._timeline["Movement"]
    Movementdf = pd.DataFrame(
        {
            "pitch": data[0],
            "roll": data[1],
            "yaw": data[2],
            "distance": data[3] * 10,
            "trial": data[4],
            "time": data[5],
        }
    )
    Movementdf["time"] = pd.to_datetime(
        Movementdf["time"].apply(
            lambda x: datetime.fromordinal(int(x))
            + timedelta(days=x % 1)
            - timedelta(days=366)
        )
    )
    Movementdf["time"] = (Movementdf["time"] - Movementdf["time"][0]).dt.total_seconds()
    Movementdf = Movementdf.assign(
        dd=np.diff(Movementdf["distance"].shift(1), append=np.nan)
    )
    Movementdf = Movementdf.assign(dd=Movementdf["dd"].shift(-1))
    Movementdf = Movementdf.assign(
        dt=np.diff(Movementdf["time"].shift(1), append=np.nan)
    )
    Movementdf = Movementdf.assign(dt=Movementdf["dt"].shift(-1))
    # Movementdf.dropna(inplace = True)
    Movementdf = Movementdf.assign(speed=Movementdf["dd"] / Movementdf["dt"])
    alpha_dx = MouseObject._settings["ConstrastSteps"].item()
    alpha_dx = 1 / (alpha_dx / 2)
    Movementdf.loc[Movementdf["distance"] < 75, "alpha"] = np.minimum(
        1, Movementdf.loc[Movementdf["distance"] < 75, "distance"] / 10 * alpha_dx
    )
    Movementdf.loc[Movementdf["distance"] > 75, "alpha"] = np.maximum(
        0,
        1
        - np.abs(
            1 - Movementdf.loc[Movementdf["distance"] > 75, "distance"] / 10 * alpha_dx
        ),
    )
    Movementdf['distance_interp'] = np.nan
    Movementdf['vel_interp'] = np.nan
    for trial in Movementdf.trial.unique():
        distance = Movementdf.query(f'trial == {trial}')['distance']
        pitch = Movementdf.query(f'trial == {trial}')['pitch']
        roll = Movementdf.query(f'trial == {trial}')['roll']
        vel = np.sqrt(pitch**2 + roll**2)
        Movementdf.loc[Movementdf["trial"]==trial,'vel_interp'] = vel.cumsum()
        Movementdf.loc[Movementdf["trial"]==trial,'distance_interp'] = distance + (trial-1)*150
    return Movementdf


def get_rastermap(MouseObject, n_comp=200):
    S = MouseObject._spks.copy()
    mu = S.mean(1)
    sd = S.std(1)
    S = (S - mu[:, np.newaxis]) / (1e-10 + sd[:, np.newaxis])
    S -= S.mean(axis=0)
    # PCA for rastermap
    U = PCA(n_components=n_comp).fit_transform(S)
    model = mapping.Rastermap(
        n_clusters=100,
        n_PCs=n_comp,
        grid_upsample=10,
        n_splits=0,
        time_lag_window=10,
        ts=0.9,
    ).fit(S, normalize=False, u=U)
    return model


def get_init_frames_per_category(MouseObject):
    """
    This function return the frame index of the first frame of each trial

    Parameters
    ----------
    MouseObject : object
        Object containing the data of a mouse

    Returns
    -------
    init_frames_per_category : np.array
        Array containing the frame index of the first frame of each trial
    first_trialframes : int
        index of the firt trial start
    last_trialframes : int
        index of the last trial start
    """
    init_frames_per_category = np.empty((4), dtype=object)
    opt_dict = {
        "rewarded": "tab:green",
        "non rewarded": "tab:red",
        "rewarded test": "tab:cyan",
        "non rewarded test": "tab:orange",
    }
    first_trialframes = []
    last_trialframes = []
    categories, trial_counts = get_trial_categories(
        MouseObject._trial_info["isrewarded"], MouseObject._trial_info["istest"]
    )
    for i, cat_color in enumerate(opt_dict.items()):
        ix = np.round(MouseObject._timestamps["trial_frames"]) * (
            categories == cat_color[0]
        )
        ix = ix[ix != 0].astype(int)
        init_frames_per_category[i] = ix
        if trial_counts[cat_color[0]] != 0:
            first_trialframes.append(np.min(init_frames_per_category[i]))
            last_trialframes.append(np.max(init_frames_per_category[i]))
    first_trialframe = np.min(np.array(first_trialframes))
    last_trialframe = np.max(np.array(last_trialframes))
    return init_frames_per_category, first_trialframe, last_trialframe


def get_frametypes(MouseObject, color=True):
    """
    Return a numpy array containing the type of frame for each frame in the recording.
    The type of frame is determined by the trial type (rewarded, non rewarded, rewarded test, non rewarded test)
    Parameters
    ----------
    MouseObject : Mouse object
        Mouse object containing the data of interest
    color : bool, optional
        True, the output array will contain the color of the trial type, by default True
        False, the output array will contain the name of the trial type.
    Returns
    -------
    trial_type_byframe : numpy array
        Numpy array containing the type of frame for each frame in the recording
    """
    ttypebyframes = ["NaN"] * MouseObject._spks.shape[1]
    ttypebyframes = np.array(ttypebyframes, dtype=object)
    opt_dict = {
        "rewarded": "tab:green",
        "non rewarded": "tab:red",
        "rewarded test": "tab:cyan",
        "non rewarded test": "tab:orange",
    }
    categories, _ = get_trial_categories(
        MouseObject._trial_info["isrewarded"], MouseObject._trial_info["istest"]
    )
    for cat_color in opt_dict.items():
        ix = np.round(MouseObject._timestamps["trial_frames"]) * (
            categories == cat_color[0]
        )
        ix = ix[ix != 0].astype(int)
        if color == True:
            ttypebyframes[ix] = cat_color[1]
        else:
            ttypebyframes[ix] = cat_color[0]
    df = pd.DataFrame(ttypebyframes, columns=["trial_type"])
    df.replace("NaN", np.nan, inplace=True)
    filled = df.fillna(method="ffill")
    trial_type_byframe = filled.values.flatten()
    return trial_type_byframe

#### FRAME SELECTOR #####
def get_fremeselector(MouseObject , effective_frames = True):
    reward_delivery_frame = np.round(
        MouseObject._timestamps["reward_frames"][
            np.isnan(MouseObject._timestamps["reward_frames"]) == False
        ]
    ).astype(int)
    FrameSelector = pd.DataFrame(
        {
            "trial_no": get_trial_per_frame(MouseObject),
            "trial_type": get_frametypes(MouseObject, color=False),
            "contrast": MouseObject._timestamps["alpha"][: MouseObject._spks.shape[1]],
            "velocity": MouseObject._timestamps["run"][: MouseObject._spks.shape[1]],
            "distance": MouseObject._timestamps["distance"][
                : MouseObject._spks.shape[1]
            ],
            "reward_delivery": np.nan,
            "ordinal_time": MouseObject._timestamps["frame_times"][: MouseObject._spks.shape[1]],
        }
    )
    FrameSelector.loc[reward_delivery_frame, "reward_delivery"] = "delivery"
    rewarded_trials = FrameSelector.loc[FrameSelector["trial_type"] == "rewarded"][
        "trial_no"
    ].unique()
    for trial in rewarded_trials:
        selected_trial = FrameSelector.query(f"trial_no == {trial}")
        selected_frames = selected_trial.index.values
        delivery_frame = selected_trial.query(
            "reward_delivery == 'delivery'"
        ).index.values
        if np.any(delivery_frame):
            before_delivery = selected_frames[selected_frames < delivery_frame]
            after_delivery = selected_frames[selected_frames > delivery_frame]
            FrameSelector.loc[after_delivery, "reward_delivery"] = "after"
            FrameSelector.loc[before_delivery, "reward_delivery"] = "pre"

    FrameSelector["ordinal_time"] = pd.to_datetime(
    FrameSelector["ordinal_time"].apply(
        lambda x: datetime.fromordinal(int(x))
        + timedelta(days=x % 1)
        - timedelta(days=366)
        )
    )
    FrameSelector["time_fromstart"] = (FrameSelector["ordinal_time"] - FrameSelector["ordinal_time"][0]).dt.total_seconds()
    all_trials = FrameSelector["trial_no"].unique()
    trials = all_trials[~np.isnan(all_trials)].astype(int)
    FrameSelector["time_within_trial"] = np.nan
    for trial in trials:
        FrameSelector.loc[FrameSelector["trial_no"] == trial, "time_within_trial"] = FrameSelector.loc[FrameSelector["trial_no"] == trial, "time_fromstart"] - FrameSelector.loc[FrameSelector["trial_no"] == trial, "time_fromstart"].iloc[0]
        FrameSelector.loc[FrameSelector["trial_no"] == trial, "distance"] = np.abs(FrameSelector.loc[FrameSelector["trial_no"] == trial, "distance"] - (150 * (FrameSelector.loc[FrameSelector["trial_no"] == trial, "trial_no"]-1)))
    FrameSelector = FrameSelector.drop(columns=["ordinal_time"]) 
    if effective_frames == True:
        FrameSelector = FrameSelector.loc[~pd.isnull(FrameSelector)['trial_no']]
    return FrameSelector


def get_neurons_bytrial(
    MouseObject,
    FrameSelector,
    rwd_condition="reward_delivery == 'pre'",
    nonrwd_condition="distance <= 57",
):
    """
    This function returns a numpy array containing the mean firing rate for the n frames meeting the conditions specified in each trial type, for each neuron.
    Parameters
    ----------
    MouseObject : Mouse object
        Mouse object containing the mouse data
    FrameSelector : pandas DataFrame
        DataFrame containing the the correspondences between frames and many behavorial variables
    rwd_condition : str, optional
        Condition to select the frames of rewarded trials, by default "reward_delivery == 'pre'"
    nonrwd_condition : str, optional
        Condition to select the frames of non rewarded trials, by default "distance <= 57"
    Returns
    -------
    spks_bytrial : numpy array
        Numpy array containing the mean firing rate for the n frames meeting the conditions specified in each trial type, for each neuron.
    trials : numpy array
        Numpy array containing the original (matlab) trial numbers, the spks_bytrial array is indexed from 0, but the trials in FrameSelector are indexed from 1 (from matlab)
    bad_trials : list
        List containing the trials with detected issues
    """
    all_trials = FrameSelector["trial_no"].unique()
    trials = all_trials[~np.isnan(all_trials)].astype(int)
    n_neurons = MouseObject._spks.shape[0]
    n_trials = len(trials)
    spks_bytrial = np.empty([n_neurons, n_trials])
    bad_trials = []
    conditions = {"rewarded": f"{rwd_condition}", "non rewarded": f"{nonrwd_condition}",  "non rewarded test": f"{nonrwd_condition}" , "rewarded test": f"{nonrwd_condition}"}
    for trial in trials:
        trial_type = (
            FrameSelector.query(f"trial_no == {trial}")["trial_type"].unique().item()
        )
        full_query = conditions[trial_type] + f" & trial_no == {trial}"
        frame_num = FrameSelector.query(full_query).index.values
        if len(frame_num) == 0:
            print(
                f"{trial_type} trial: {trial}, has no frames meeting the condition: {conditions[trial_type]}"
            )
            print("check that trial!!, filling with nan")
            spks_bytrial[:, trial - 1] = np.nan
            bad_trials.append(trial-1)
        else:
            selected_spks = MouseObject._spks[:, frame_num]
            spks = selected_spks.mean(axis=1)
            spks_bytrial[:, trial - 1] = spks
    return spks_bytrial, np.array(bad_trials)

def get_trialno_bytype(FrameSelector):
    seq = ("rewarded","non rewarded","rewarded_test","non rewarded_test")
    trial_type_dict = dict.fromkeys(seq, np.nan)
    ttypes = FrameSelector["trial_type"].unique()
    nan_mask = pd.isnull(FrameSelector["trial_type"].unique())
    ttypes = ttypes[~nan_mask]
    for trial_type in ttypes:
        trialno = FrameSelector.query(f"trial_type == '{trial_type}'")["trial_no"].unique().astype(int)
        trialno = trialno - 1
        trial_type_dict[trial_type] = trialno
    return trial_type_dict

def superneuron_toneurons(isort_vect,clust_idxs,spn_binsize):
    """
    convert superneuron index to individual neuron index

    Parameters:
    isort_vect: isort vector from rastermap
    clust_idxs: tuple of (start, end) cluster index
    spn_binsize: number of neurons per superneuron

    Returns: 
    selected_neurons: list of neuron indices
    """

    nsuper = len(isort_vect)//spn_binsize
    assert clust_idxs[1] <= nsuper, "clustidx[1] should be smaller than number of superneurons"
    assert clust_idxs[0] >= 0, "clustidx[0] should be larger than 0"
    selected_neurons = isort_vect[clust_idxs[0]*spn_binsize:(clust_idxs[1]+1)*spn_binsize]
    return selected_neurons

@dataclass
class Mouse:

    name: str
    datexp: str
    blk: str

    def load_behav(self, timeline_block=None, verbose=False):

        """
        Loads the experiment info from the database
        Parameters:
        ----------
        timeline_block : int
            Specifies the timeline block to choose
        verbose : bool
            If True, prints the timeline info
        Returns:
        ----------
        Timeline : dict
            Dictionary with the experiment info
        """

        ### This function can be ad hoc for different experiments, this is just an example for mine ###
        ## at the end, this functions should always return a timeline dict, and the data_var to sync the behav and imaging data

        if timeline_block is not None:
            blk = str(timeline_block)
            root = os.path.join("Z:/data/PROC", self.name, self.datexp, blk)
            fname = "Timeline_%s_%s_%s.mat" % (self.name, self.datexp, blk)
        else:
            root = os.path.join("Z:/data/PROC", self.name, self.datexp, self.blk)
            fname = "Timeline_%s_%s_%s.mat" % (self.name, self.datexp, self.blk)

        fnamepath = Path(os.path.join(root, fname))

        # old matlab file format
        try:
            matfile = io.loadmat(fnamepath, squeeze_me=True)["Timeline"]
            self.timeline = matfile["Timeline"]
            self.data_var = matfile["data"]
        except NotImplementedError:
            print("Timeline file is in v7.3 format, loading with h5py")
            ## Syntax to load a .mat v7.3 file
            with h5py.File(fnamepath, "r") as f:
                timeline_group = f.get("Timeline").get("Results")
                settings_group = f.get("Timeline").get("Settings")
                data_var = np.array(
                    f.get("Timeline").get("data")
                )  # variable that syncs the timeline with the imaging data
                timeline = {k: np.array(v) for k, v in timeline_group.items()}
                settings = {k: np.array(v) for k, v in settings_group.items()}

        self._timeline = timeline
        self._data_var = data_var
        self._settings = settings
        if verbose:
            print("###### Behavior loaded ######")
            print("Loaded timeline from: %s" % fnamepath)
            print("----------------------------------")
            print("Timeline dict created: into self._timeline")
            print(f"Timeline keys: {self._timeline.keys()}")
            print("----------------------------------")
            print("Settings dict created: into self._settings")
            print(f"Timeline keys: {self._settings.keys()}")
            print("----------------------------------")
            print("Data_var loaded:  into self._data_var")
            print("This variable is used to sync the timeline with the imaging data")

    def load_neurons(self, dual_plane=True, baseline=True, verbose=False):
        """
        Loads the neural data from the database

        Parameters:
        ----------

        dual_plane : Boolean
            Dual plane flag indicates whether the data is from the dual plane or not
        Baseline : Boolean
            Baseline flag indicates whether the data is preproceded or not
        Returns:
        ----------
        spks : array
            Spike matrix

        """

        root = os.path.join("Z:/data/PROC", self.name, self.datexp, self.blk)
        ops = np.load(
            os.path.join(root, "suite2p", "plane0", "ops.npy"), allow_pickle=True
        ).item()

        if dual_plane:
            tlags = np.linspace(0.2, -0.8, ops["nplanes"] // 2 + 1)[:-1]
            tlags = np.hstack((tlags, tlags))
            tlags = tlags.flatten()
        else:
            tlags = np.linspace(0.2, -0.8, ops["nplanes"] + 1)[:-1]
        print(f"planes: {tlags.shape[0]}")

        spks = np.zeros((0, ops["nframes"]), np.float32)
        stat = np.zeros((0,))
        iplane = np.zeros((0,))
        xpos, ypos = np.zeros((0,)), np.zeros((0,))

        for n in tqdm(range(ops["nplanes"])):
            ops = np.load(
                os.path.join(root, "suite2p", "plane%d" % n, "ops.npy"),
                allow_pickle=True,
            ).item()

            stat0 = np.load(
                os.path.join(root, "suite2p", "plane%d" % n, "stat.npy"),
                allow_pickle=True,
            )
            ypos0 = np.array([stat0[n]["med"][0] for n in range(len(stat0))])
            xpos0 = np.array([stat0[n]["med"][1] for n in range(len(stat0))])

            ypos0 += ops["dy"]
            xpos0 += ops["dx"]

            if baseline:
                F = np.load(os.path.join(root, "suite2p", "plane%d" % n, "F.npy"))
                Fneu = np.load(os.path.join(root, "suite2p", "plane%d" % n, "Fneu.npy"))
                F = baselining(ops, tlags[n], F, Fneu)
                spks0 = dcnv.oasis(F, ops["batch_size"], ops["tau"], ops["fs"])
            else:
                spks0 = np.load(
                    os.path.join(root, "suite2p", "plane%d" % n, "spks.npy"),
                    allow_pickle=True,
                )

            spks0 = spks0.astype("float32")
            if spks.shape[1] > spks0.shape[0]:
                spks0 = np.concatenate(
                    (spks0, np.zeros((spks0.shape[0], spks.shape[1] - spks0.shape[1]))),
                    axis=1,
                )
            spks = np.concatenate((spks, spks0.astype("float32")), axis=0)
            ypos = np.concatenate((ypos, ypos0), axis=0)
            xpos = np.concatenate((xpos, xpos0), axis=0)
            iplane = np.concatenate(
                (
                    iplane,
                    n
                    * np.ones(
                        len(stat0),
                    ),
                )
            )
            stat = np.concatenate((stat, stat0), axis=0)
        self._spks = spks
        self._ypos = ypos
        self._xpos = xpos
        self._iplane = iplane
        self._stat = stat
        self._ops = ops
        if verbose:
            print("###### Neurons loaded ######")
            print(f"Total neurons loaded: {len(spks)}")
            print("---------------------------")
            print(
                f"Spikes created at: self.spks, with shape: {spks.shape} : (neurons, frames)"
            )
            print("Neurons plane information created at: self.iplane")
            print("Neurons positions created at: self.xpos, self.ypos")
            print("Suite2p stats created at: self.stat")
            print("Suite2p options created at: self.ops")
            print("---------------------------")

    def get_timestamps(self, verbose=False):
        """
        Creates the timestamps of behavior in terms of frames

        ******
        the syntax of this function will change for diferent experiments, since the timeline variable might have different named variables,
        but in escence it is the same process.
        ******
        Parameters:
        verbose : Boolean
            Verbose flag indicates whether to print timestamps info or not
        -------
        Returns:
        -------
        timestamps : dict
            Dictionary with the timestamps of the behavior in terms of frames
        """
        try:
            # Gets the time for each neural frame
            frames_time = self._data_var[0]
            ixx = (frames_time[:-1] > 2.5) * (frames_time[1:] < 2.5)
            iframes = np.argwhere(ixx).flatten()
            isamp = np.argwhere(self._data_var[1] > 1).flatten()
            ts = self._data_var[1][isamp]
            tframes = interp1d(isamp, ts)(iframes)
            nframes = len(tframes)
        except NameError:
            print(
                "data_var or _timeline not loaded, make sure to load them first by self.load_behav()"
            )

        # what frame number does each trial start on? 
        ttrial = self._timeline["Movement"][4]
        istart = np.argwhere(np.diff(ttrial) > 0.5).flatten() + 1
        frameidx_first_trial = np.where(self._timeline["Movement"][5] > tframes.min())[0][0]
        istart = np.insert(istart,0,frameidx_first_trial) # insert frameidx_first_trial in first position
        tstart = self._timeline["Movement"][5][istart]

        # get trial start frames
        trial_frames = interp1d(tframes, np.arange(1, nframes + 1))(
            tstart
        )  
        df = get_movement_df(self)
        # interpolate running speed for each neural frame
        runsp = (
            self._timeline["Movement"][0] ** 2 + self._timeline["Movement"][1] ** 2
        ) ** 0.5
        trun = self._timeline["Movement"][5]
        run = interp1d(trun, runsp, fill_value="extrapolate")(tframes)

        # interpolate running speed for each neural frame
        distance = df["distance_interp"].values
        trun = self._timeline["Movement"][5]
        dist = interp1d(trun, distance, fill_value="extrapolate")(tframes)

        # interpolate alpha values for each neural frame
        alpha = df.alpha.values
        trun = self._timeline["Movement"][5]
        alpha_interp = interp1d(trun, alpha, fill_value="extrapolate")(tframes)

        # get lick times as frame numbers
        tlick = self._timeline["Licks"][4]
        ix = self._timeline["Licks"][5] < 0.5
        tlick = tlick[ix]
        frame_lick = interp1d(tframes, np.arange(1, nframes + 1))(tlick)
        # get the reward delivery as frame numbers
        treward = self._timeline["RewardInfo"][1]
        frame_reward = interp1d(tframes, np.arange(1, nframes + 1))(treward)

        timestamps = {
            "trial_frames": trial_frames,
            "run": run,
            "distance": dist,
            "alpha": alpha_interp,
            "lick_frames": frame_lick,
            "reward_frames": frame_reward,
            "frame_times": tframes,
        }
        self._timestamps = timestamps
        if verbose:
            print("###### Timestamps created ######")
            print("---------------------------")
            print('Trial frames created at: self._timestamps["trial_frames"]')
            print('Interpolated run variable created at: self._timestamps["run"]')
            print(
                'Interpolated distance variable created at: self._timestamps["distance"]'
            )
            print('Interpolated alpha variable created at: self._timestamps["alpha"]')
            print(
                'Licks in terms of frames created at: self._timestamps["lick_frames"]'
            )
            print(
                'Rewards in terms of frames created at: self._timestamps["reward_frames"]'
            )
            print('Frame times created at: self._timestamps["frame_times"]')
            print("---------------------------")

    def get_trial_info(self, verbose=False):
        """
        Creates the trial info information vectors, indicating the trial type
        Parameters:
        -------
        verbose : Boolean
            Verbose flag indicates whether to print trial type info or not
        Returns:
        -------
        trial_info : dict
            Dictionary with the trial type information
        """
        try:
            ttrial = self._timeline["Movement"][4]
            istart = np.argwhere(np.diff(ttrial) > 0.5).flatten() + 1
            frameidx_first_trial = np.where(self._timeline["Movement"][5] > self._timestamps["frame_times"].min())[0][0]
            istart = np.insert(istart,0,frameidx_first_trial) # insert frameidx_first_trial in first position
            tstart = self._timeline["Movement"][5][istart]
        except NameError:
            print(
                "data_var, _timeline or _timestamps not loaded, make sure to load them first by self.load_behav() and self.get_timestamps()"
            )

        # get the trial type for each trial
        ntrials = len(tstart)
        trial_type = self._timeline["TrialRewardStrct"].flatten()[:ntrials]
        trial_new = self._timeline["TrialNewTextureStrct"].flatten()[:ntrials]
        #trial_type = np.roll(trial_type, -1)
        #trial_new = np.roll(trial_new, -1)
        trial_info = {
            "isrewarded": trial_type,
            "istest": trial_new,
        }

        self._trial_info = trial_info
        if verbose:
            print("###### Trial informartion dict created ######")
            print("---------------------------")
            print(
                'Boolean array describing if the trial is rewarded created at: self._trial_info["isrewarded"]'
            )
            print(
                'Boolean array describing if the trial was a test trial created at: self._trial_info["istest"]'
            )
            print("---------------------------")

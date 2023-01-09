import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from src import utils
import os

sns.set_context("notebook")


def rastermap_plot(
    MouseObject,
    neuron_embedding,
    frame_selection=0,
    frame_num=500,
    svefig=False,
    savepath=None,
    format="png",
    clustidx = None,
):
    """
    plot the rastermap embedding with behavioral annotations for a given mouse object and embedding

    Parameters
    ----------
    MouseObject : Mouse object
        Mouse object containing the data
    neuron_embedding : np.array
        rastermap embedding of the spks
    frame_selection : int
        which frame_num frames to plot, i.e. 3 means the frames from 1500 to 2000 if frame_num = 500
    frame_num : int
        number of frames to plot
    svefig : bool
        whether to save the figure or not
    format : str
        format of the saved figure
    savepath : str
        path to save the figure
    clustidx : tuple, optional
        tuple of two idx containing the start and end of a desired cluster, by default None
    """
    ## data unpacking ##
    xmin = frame_selection * frame_num
    xmax = (frame_selection + 1) * frame_num

    run = MouseObject._timestamps["run"]
    tframe = MouseObject._timestamps["trial_frames"]
    isrewarded = MouseObject._trial_info["isrewarded"]
    istest = MouseObject._trial_info["istest"]
    alpha = MouseObject._timestamps["alpha"]
    nsuper = neuron_embedding.shape[0]

    ## figure creation ##
    fig = plt.figure(figsize=(16, 9), dpi=300)
    grid = plt.GridSpec(11, 5, hspace=0.2, wspace=0.2)
    raster_ax = fig.add_subplot(grid[2:, :5], facecolor="w")
    alpha_ax = fig.add_subplot(grid[:1, :5], sharex=raster_ax)
    vel_ax = fig.add_subplot(grid[1:2, :5], sharex=raster_ax)
    # the embedding is zscored for this particular time range so the contrast is better.
    raster_ax.imshow(
        zscore(neuron_embedding[:, xmin:xmax], 1),
        cmap="gray_r",
        aspect="auto",
        vmin=0,
        vmax=2,
    )
    if clustidx is not None:
        raster_ax.fill_between(np.arange(0,frame_num),clustidx[1],clustidx[0], color='tab:purple', alpha=0.2)
    vel_ax.plot(run[xmin:xmax], linewidth=0.5, color="k")
    alpha_ax.plot(alpha[xmin:xmax], linewidth=0.5, color="k")
    for label_vel, label_alpha in zip(
        vel_ax.get_xticklabels(), alpha_ax.get_xticklabels()
    ):
        label_vel.set_visible(False)
        label_alpha.set_visible(False)

    ## behavioral annotations ##
    for i, annot in enumerate(["lick_frames", "reward_frames"]):
        ranges = (MouseObject._timestamps[annot] > xmin) * (
            MouseObject._timestamps[annot] < xmax
        )
        pos = MouseObject._timestamps[annot][ranges] - xmin
        if i == 0:
            vel_ax.plot(
                pos,
                -np.min(run[xmin:xmax]) * np.ones(len(pos)),
                "|b",
                markersize=5,
                label="licks",
                alpha=0.2,
            )
        else:
            for p in pos:
                vel_ax.axvline(
                    p,
                    ymin=0,
                    ymax=nsuper,
                    label="reward delivery",
                    linestyle="dashed",
                    color="m",
                    alpha=0.5,
                    lw=0.5,
                )
                alpha_ax.axvline(
                    p,
                    ymin=0,
                    ymax=nsuper,
                    label="reward delivery",
                    linestyle="dashed",
                    color="m",
                    alpha=0.5,
                    lw=0.5,
                )
                raster_ax.axvline(
                    p,
                    ymin=0,
                    ymax=nsuper,
                    label="reward delivery",
                    linestyle="dashed",
                    color="m",
                    alpha=0.5,
                    lw=0.5,
                )
    vel_ax.text(
        1.01,
        0.4,
        "reward delivery",
        c="m",
        va="center",
        transform=vel_ax.transAxes,
        fontsize=12,
        alpha=0.8,
    )
    vel_ax.text(
        1.01,
        0.1,
        "licks",
        c="b",
        va="center",
        transform=vel_ax.transAxes,
        fontsize=12,
        alpha=0.8,
    )

    # trial type annotations #
    frame_ranges = (tframe > xmin) * (tframe <= xmax)
    opt_dict = {
        "rewarded": "tab:green",
        "non rewarded": "tab:red",
        "rewarded test": "tab:cyan",
        "non rewarded test": "tab:orange",
    }
    categories, _ = utils.get_trial_categories(isrewarded, istest)
    for cat_color in opt_dict.items():
        ix = frame_ranges * (categories == cat_color[0])
        ix = ix.nonzero()[0]
        for i in ix:
            raster_ax.axvline(
                x=tframe[i] - xmin,
                ymin=0,
                ymax=nsuper,
                color=cat_color[1],
                lw=0.8,
                alpha=0.5,
            )
            vel_ax.axvline(
                x=tframe[i] - xmin,
                ymin=0,
                ymax=nsuper,
                color=cat_color[1],
                lw=0.8,
                alpha=0.5,
            )
            alpha_ax.axvline(
                x=tframe[i] - xmin,
                ymin=0,
                ymax=nsuper,
                color=cat_color[1],
                lw=0.8,
                alpha=0.5,
            )

    text_offset = 0
    for cat_color in opt_dict.items():
        text_offset -= 0.05
        if np.sum((categories == cat_color[0])) > 0:
            raster_ax.text(
                1.01,
                0.9 + text_offset,
                cat_color[0],
                c=cat_color[1],
                va="center",
                transform=raster_ax.transAxes,
                fontsize=12,
                alpha=0.8,
            )

    raster_ax.set_ylabel("Superneuron #")
    if frame_num < 100:
        raster_ax.set_xticks([0, frame_num], [str(xmin), str(xmax)])
    else:
        raster_ax.set_xticks(
            np.arange(0, frame_num, 100), np.arange(xmin, xmax, 100).astype(str)
        )
    alpha_ax.set_ylabel("Contrast")
    vel_ax.set_ylabel("Velocity")
    raster_ax.set_xlabel("Frame #")
    sns.despine()
    if svefig:
        if savepath is None:
            plt.savefig(
                f"rastermap_embedding_{str(frame_selection)}.{format}",
                bbox_inches="tight",
            )
            plt.close("all")
        else:
            plt.savefig(
                os.path.join(
                    savepath, f"rastermap_embedding_{str(frame_selection)}.{format}"
                ),
                bbox_inches="tight",
            )
            plt.close("all")

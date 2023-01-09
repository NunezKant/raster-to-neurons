import sys 
import mkl
import os
mkl.set_num_threads(10)
sys.path.insert(0,r"C:\Users\labadmin\Documents\suite2p")
sys.path.insert(0,r"C:\Users\labadmin\Documents\rastermap")
import numpy as np 
from src import utils, plots # this is our own library of functions
from rastermap import mapping
import streamlit as st
from suite2p.extraction import dcnv
from sklearn.decomposition import PCA

st.title("Rastermap cluster explorer")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.header("Load experiment data")
name = st.sidebar.text_input("Enter mouse name", key = "mouse_name")
date = st.sidebar.text_input("Enter experiment date in format 'YYYY_MM_DD'", key="date")
block = st.sidebar.text_input("Enter block number", key = "block")
if 'n_frames' not in st.session_state:
    st.session_state['n_frames'] = 10
if 'n_superneurons' not in st.session_state:
    st.session_state['n_superneurons'] = np.arange(100)

    
st.sidebar.header("Rastermap parameters")
n_comp = st.sidebar.number_input("ncomp", value = 200, key = "ncomp")
n_clusters = st.sidebar.number_input("ncomp", value = 100, key = "n_clusters")
time_lag_window = st.sidebar.number_input("time_lag_window", value = 10, key="time_lag_window")
grid_upsample = st.sidebar.number_input("grid_upsample", value = 10, key="grid_upsample")
n_splits = st.sidebar.number_input("n_splits", value = 0, key = "n_splits")
ts = st.sidebar.number_input("ts", value = 0.9, key = "ts")

# Load data
if st.sidebar.button("load experiment and run rastermap"):
    dual_plane = True
    baseline = True
    verbose = False
    my_bar= st.sidebar.progress(0)
    mouse = utils.Mouse(name,date,block)
    root = os.path.join("Z:/data/PROC", name, date, block)
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

    for n in range(ops["nplanes"]):
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
            F = utils.baselining(ops, tlags[n], F, Fneu)
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
        my_bar.progress((n+1)/ops["nplanes"])
    st.sidebar.write("experiment data loaded.")
    mouse._spks = spks
    mouse._ypos = ypos
    mouse._xpos = xpos
    mouse._iplane = iplane
    mouse._stat = stat
    mouse._ops = ops
    mouse.load_behav(timeline_block=None)
    mouse.get_timestamps()
    mouse.get_trial_info()
    
    FrameSelector = utils.get_fremeselector(mouse)
    trialno_dict = utils.get_trialno_bytype(FrameSelector)
    st.session_state['FrameSelector'] = FrameSelector
    st.session_state['trialno_dict'] = trialno_dict
    S = mouse._spks.copy()
    mu = S.mean(1)
    sd = S.std(1)
    S = (S - mu[:, np.newaxis]) / (1e-10 + sd[:, np.newaxis])
    S -= S.mean(axis=0)
    # PCA for rastermap
    U = PCA(n_components=n_comp).fit_transform(S)
    model = mapping.Rastermap(
        n_clusters=n_clusters,
        n_PCs=n_comp,
        grid_upsample=grid_upsample,
        n_splits=n_splits,
        time_lag_window=time_lag_window,
        ts=ts,
    ).fit(S, normalize=False, u=U)
    st.sidebar.write("Rastermap model fitted")
    if 'neuron_embedding' not in st.session_state:
        st.session_state['neuron_embedding'] = model.X_embedding
        st.session_state['isort'] = model.isort
        st.session_state['n_superneurons'] = np.arange(model.X_embedding.shape[0])
        st.session_state['n_frames'] = mouse._spks.shape[1]//500
    if 'mouse' not in st.session_state:
        st.session_state['mouse'] = mouse
else:
    st.sidebar.write("Please enter mouse name, experiment date and block number and rastermap settings")

def update_plot():
    fig = plots.rastermap_plot(st.session_state['mouse'], st.session_state['neuron_embedding'], frame_selection=frame_selection, clustidx = cluster_selection)
    binsize = 50
    selected_neurons = st.session_state['isort'][cluster_selection[0]*binsize:(cluster_selection[1]+1)*binsize]
    selected_neurons_plane_1 = st.session_state['mouse']._iplane[selected_neurons] >= 10
    selected_neurons_plane_2 = st.session_state['mouse']._iplane[selected_neurons] < 10
    plane1_neurons = selected_neurons[selected_neurons_plane_1]
    plane2_neurons = selected_neurons[selected_neurons_plane_2]
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig2 = make_subplots(rows=1, cols=1)
    fig2.add_trace(go.Scatter(x=st.session_state['mouse']._xpos, y=-st.session_state['mouse']._ypos, mode = 'markers', marker= dict(size=3, color='gray',opacity = 0.2)))
    fig2.add_trace(go.Scatter(x=st.session_state['mouse']._xpos[plane1_neurons], y=-st.session_state['mouse']._ypos[plane1_neurons], mode = 'markers', marker= dict(size=6, color='red',opacity = 0.5)))
    fig2.add_trace(go.Scatter(x=st.session_state['mouse']._xpos[plane2_neurons], y=-st.session_state['mouse']._ypos[plane2_neurons], mode = 'markers', marker= dict(size=6, color='blue',opacity = 0.5)))
    fig2.update_layout(height=700, width=700, template="simple_white")
    fig2.update_xaxes(visible=False)
    fig2.update_yaxes(visible=False)
    for i, name in zip(range(len(fig2.data)),["population", "layer 1", "layer 2"]):
        fig2.data[i].name = name
    st.pyplot(fig)
    st.plotly_chart(fig2)

container1 = st.container()
container2 = st.container()
with container1:
    st.header("Select plot parameters once you loaded the data")
    frame_selection = st.slider("Select frame", 0, st.session_state['n_frames'], 0)
    cluster_selection = st.select_slider("Select cluster", options = st.session_state['n_superneurons'], value=(0,10))
if st.button("plot!"):
    with container2:
        update_plot()
        

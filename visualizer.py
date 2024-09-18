"""Visualizer for SLEAP and NWB data. To run the web app,
run `streamlit run visualizer.py` in the terminal."""

import numpy as np
import streamlit as st
import pygfx as gfx
from PyQt5.QtWidgets import QApplication
from wgpu.gui.auto import WgpuCanvas
import sleap_io as sio
from sleap_io import Labels

colors = [
    (255, 0, 0, 1.0),
    (0, 255, 0, 1.0),
    (0, 0, 255, 1.0),
    (255, 255, 0, 1.0),
    (0, 255, 255, 1.0),
    (255, 0, 255, 1.0),
]

def draw_slp(labels: Labels, frame_index: int):
    instances = labels.labeled_frames[frame_index].instances
    video = labels.videos[0]
    image_height, image_width = video.shape[1], video.shape[2]
    canvas = WgpuCanvas()
    renderer = gfx.WgpuRenderer(canvas)
    scene = gfx.Scene()
    scene.add(gfx.Background(material=gfx.BackgroundMaterial("#000000")))

    for i, instance in enumerate(instances):
        instance_np = instance.numpy() - 100
        positions = np.hstack((instance_np, np.ones((instance_np.shape[0], 1)))).astype(
            np.float32
        )
        points = gfx.Points(
            gfx.Geometry(positions=positions),
            gfx.PointsMaterial(size=4, color=colors[i % len(colors)]),
        )
        scene.add(points)
        for index1, index2 in instance.skeleton.edge_inds:
            edge = instance_np[[index1, index2]]
            edge = np.hstack((edge, np.ones((edge.shape[0], 1)))).astype(np.float32)
            scene.add(
                gfx.Line(
                    gfx.Geometry(positions=edge),
                    gfx.LineMaterial(color=colors[i % len(colors)]),
                )
            )
    camera = gfx.OrthographicCamera(image_width, image_height)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    canvas


st.title("SLEAP and NWB data visualizer")
st.write("See https://sleap.ai/ for more information about SLEAP.")
st.write("See https://pynwb.readthedocs.io/ for more information about NWB.")
filename = st.text_input("Enter the path to a .slp or .nwb file:")
if not filename.endswith(".slp") and not filename.endswith(".nwb"):
    st.write("Please enter a valid .slp or .nwb file path.")

else:
    try:
        labels = sio.load_file(filename)
        st.write(labels)
        frame_index = st.slider(
            "Choose the frame index to visualize:", 0, len(labels) - 1, 0
        )
        st.write(f"Frame {frame_index} instance points:")
        st.write(labels[frame_index].instances[0].numpy())
        # st.write(labels[frame_index].instances[0].skeleton)
        # draw_slp(labels, frame_index)
    except FileNotFoundError:
        st.write("File not found. Please enter a valid .slp or .nwb file path.")
    except Exception as e:
        st.write(f"Error loading file: {e}")
        st.write("Please enter a valid .slp or .nwb file path.")

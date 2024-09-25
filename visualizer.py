"""Visualizer for SLEAP and NWB data. To run the web app,
run `streamlit run visualizer.py` in the terminal."""

import sys
import numpy as np
import streamlit as st
import pygfx as gfx
from PyQt5.QtWidgets import QApplication
from wgpu.gui.auto import WgpuCanvas
import sleap_io as sio
from sleap_io import Labels
from streamlit.errors import StreamlitAPIException
import tempfile
import imageio.v3 as iio

colors = [
    (255, 0, 0, 1.0),
    (0, 255, 0, 1.0),
    (0, 0, 255, 1.0),
    (255, 255, 0, 1.0),
    (0, 255, 255, 1.0),
    (255, 0, 255, 1.0),
]


def draw_sleap(labels: Labels, frame_index: int) -> str:
    """Renders an image using pygfx, saves it to a temporary file,
    and returns the filename."""
    canvas = WgpuCanvas()
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
    color_buffer = np.zeros((image_height, image_width, 4), dtype=np.uint8)
    renderer.render(scene, camera, target=color_buffer)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as file:
        filename = file.name
        iio.imwrite(filename, color_buffer)
    return filename


def make_app():
    st.title("SLEAP and NWB data visualizer")
    st.write("See [sleap-io](https://io.sleap.ai/) for more information about SLEAP.")
    st.write("See [pynwb](https://pynwb.readthedocs.io/) for more information about NWB.")
    st.write(
        """**Note**: If you want to visualize an .nwb file, we recommend converting it 
            to a .slp file using the following method before using this visualizer
            to significantly decrease the load time.

        pip install sleap-io
        import sleap_io as sio
        filename = "/.../example.nwb"
        labels = sio.load_file(filename)
        sio.save_slp(labels, "/.../example.slp")"""
    )

    local_file = st.text_input("Enter the path to a .slp or .nwb file stored locally here:")
    st.write("OR")
    url_file = st.text_input(
        "Enter the URL to a .slp or .nwb file stored online here: **Note: not yet supported**"
    )
    filename = local_file or url_file

    if not filename.endswith(".slp") and not filename.endswith(".nwb"):
        st.write("Please enter a valid .slp or .nwb file path.")
    else:
        try:
            labels = sio.load_file(filename)
            st.write(labels)
            frame_index = st.slider(
                "Choose the frame index to visualize:", 0, len(labels) - 1, 0
            )
            for i, instance in enumerate(labels[frame_index].instances):
                st.write(f"Instance {i} points:")
                st.write(instance.numpy())
            
            img_file = draw_sleap(labels, frame_index)
            # st.image(filename)
        except FileNotFoundError:
            st.write("File not found. Please enter a valid .slp or .nwb file path.")
        except StreamlitAPIException:
            # This error is raised if there is only one frame because
            # the slider widget does not work with a single value.
            for i, instance in enumerate(labels[0].instances):
                st.write(f"Instance {i} points:")
                st.write(instance.numpy())


def main():
    app = QApplication([])
    make_app()

if __name__ == "__main__":
    main()

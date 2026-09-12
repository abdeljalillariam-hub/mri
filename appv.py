import math
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="AutoMRI – Localizer Simulator",
    layout="wide"
)

st.title("AutoMRI – Localizer & MRI Parameter Simulator")
st.caption("Educational simulator: pick an imaging plane and adjust the prescription angle.")

# ------------------------------------------------------------
# Image
# ------------------------------------------------------------
image_path = Path(__file__).parent / "localizer.png"

if not image_path.exists():
    st.error("localizer.png is missing. Put it in the same folder as appii.py.")
    st.stop()

base_image = Image.open(image_path).convert("RGB")

# The supplied screenshot contains 3 panels of approximately equal width,
# ordered Sagittal | Axial | Coronal.
W, H = base_image.size
panel_width = W / 3

PLANES = ["Sagittal", "Axial", "Coronal"]


def panel_x_range(plane):
    i = PLANES.index(plane)
    return i * panel_width, (i + 1) * panel_width


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("Acquisition parameters")

fov = st.sidebar.slider("FOV (mm)", 100, 400, 220, 5)
slice_thickness = st.sidebar.slider("Slice thickness (mm)", 0.5, 10.0, 5.0, 0.5)
base_resolution = st.sidebar.select_slider(
    "Base resolution",
    options=[64, 80, 96, 128, 160, 192, 224, 256, 320, 384, 512],
    value=256
)
phase_oversampling = st.sidebar.slider("Phase oversampling (%)", 0, 100, 20, 5)
nex = st.sidebar.select_slider("NEX / Averages", options=[1, 2, 3, 4, 5, 6, 8], value=2)
grappa = st.sidebar.select_slider("GRAPPA", options=[1, 2, 3, 4], value=2)
bandwidth = st.sidebar.select_slider(
    "Receiver bandwidth (Hz/px)",
    options=[130, 200, 260, 330, 400, 500, 650, 780],
    value=260
)

st.sidebar.divider()
st.sidebar.header("Slice prescription")

selected_plane = st.sidebar.selectbox("Plane", PLANES, index=1)
angle = st.sidebar.slider("Prescription angle (°)", -45, 45, 0, 1)

# ------------------------------------------------------------
# Calculations
# ------------------------------------------------------------
read_resolution = fov / base_resolution
phase_fov = fov  # square FOV assumption
phase_matrix = round(base_resolution * (1 + phase_oversampling / 100))
phase_resolution = phase_fov / (phase_matrix / (1 + phase_oversampling / 100))

voxel_volume = read_resolution * phase_resolution * slice_thickness

reference_voxel = (220 / 256) * (220 / 256) * 5
reference_bw = 260

relative_snr = (
    math.sqrt(nex)
    * (voxel_volume / reference_voxel)
    / math.sqrt(grappa)
    * math.sqrt(reference_bw / bandwidth)
)

# ------------------------------------------------------------
# Draw prescription overlay (fixed at the center of the selected panel)
# ------------------------------------------------------------
def draw_prescription(img, plane, angle_deg):
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)

    left, right = panel_x_range(plane)
    cx = (left + right) / 2
    cy = H / 2

    theta = math.radians(angle_deg)
    length = max(W, H) * 1.5
    dx = math.cos(theta) * length
    dy = math.sin(theta) * length

    pts = []
    for t in np.linspace(-1, 1, 1000):
        px = cx + dx * t
        py = cy + dy * t
        if left <= px <= right and 0 <= py <= H:
            pts.append((px, py))

    if len(pts) > 1:
        draw.line(pts, fill=(255, 230, 0), width=4)

    r = 7
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 230, 0), width=3)

    return canvas


# ------------------------------------------------------------
# Localizer display
# ------------------------------------------------------------
st.subheader("1. Prescription preview")

display_image = draw_prescription(base_image, selected_plane, angle)
st.image(display_image, width="stretch")

st.success(f"Selected plane: {selected_plane} | Angle: {angle}°")

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
st.divider()
st.subheader("2. Current acquisition")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Read resolution", f"{read_resolution:.2f} mm")
c2.metric("Phase resolution", f"{phase_resolution:.2f} mm")
c3.metric("Voxel volume", f"{voxel_volume:.2f} mm³")
c4.metric("NEX", f"{nex}")

c1, c2, c3 = st.columns(3)
c1.metric("GRAPPA", f"R = {grappa}")
c2.metric("Bandwidth", f"{bandwidth} Hz/px")
c3.metric("Relative SNR index", f"{relative_snr:.2f}")

# ------------------------------------------------------------
# Educational explanation
# ------------------------------------------------------------
st.divider()
st.subheader("3. Educational interpretation")

messages = []

if slice_thickness < 5:
    messages.append("Slice thickness ↓ → voxel volume ↓ → SNR tends to ↓.")
elif slice_thickness > 5:
    messages.append(
        "Slice thickness ↑ → voxel volume ↑ → SNR tends to ↑, but through-plane resolution decreases."
    )

if base_resolution > 256:
    messages.append(
        "Base resolution ↑ at constant FOV → smaller pixels → higher spatial resolution, lower SNR."
    )
elif base_resolution < 256:
    messages.append(
        "Base resolution ↓ at constant FOV → larger pixels → more SNR, lower spatial resolution."
    )

if phase_oversampling > 0:
    messages.append("Phase oversampling ↑ → more phase-encoding steps → longer acquisition, no SNR loss.")

if nex > 1:
    messages.append(f"NEX ↑ → SNR ∝ √NEX. For NEX={nex}, √NEX ≈ {math.sqrt(nex):.2f}.")

if grappa > 1:
    messages.append("GRAPPA ↑ → shorter acquisition, SNR penalty ∝ 1/√R (plus g-factor, not modeled here).")

if bandwidth != reference_bw:
    direction = "↓" if bandwidth > reference_bw else "↑"
    messages.append(f"Bandwidth {'↑' if bandwidth > reference_bw else '↓'} → SNR {direction} (SNR ∝ 1/√BW).")

messages.append(
    f"You selected the {selected_plane.lower()} plane. Changing the angle changes the "
    "orientation of the planned slice in this educational preview."
)

for m in messages:
    st.info(m)

st.caption(
    "Educational model only. The SNR index is simplified (no g-factor, no coil geometry) "
    "and is not a prediction of measured scanner SNR."
)

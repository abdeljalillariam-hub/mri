import math
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit-image-coordinates import streamlit-image-coordinates
st.set_page_config(page_title="AutoMRI V3", layout="wide")
st.title("AutoMRI V3 – MRI Parameter Simulator")
st.caption("Interactive 3-plane localizer + educational MRI parameter simulator.")

image_path = Path(__file__).parent / "localizer.png"
if not image_path.exists():
    st.error("localizer.png is missing. Put it next to app.py.")
    st.stop()

base_image = Image.open(image_path).convert("RGB")
W, H = base_image.size
panel_width = W / 3

PLANES = ["Sagittal", "Axial", "Coronal"]
PANEL_AXES = {
    "Sagittal": ("y", "z"),
    "Axial": ("x", "y"),
    "Coronal": ("x", "z"),
}
PLANE_COLOR = {
    "Sagittal": (255, 230, 0),
    "Axial": (255, 60, 60),
    "Coronal": (60, 170, 255),
}

st.sidebar.header("Acquisition parameters")
fov = st.sidebar.slider("FOV (mm)", 100, 400, 220, 5)
slice_thickness = st.sidebar.slider("Slice thickness (mm)", 0.5, 10.0, 5.0, 0.5)
base_resolution = st.sidebar.select_slider(
    "Base resolution",
    options=[64,80,96,128,160,192,224,256,320,384,512],
    value=256,
)
phase_oversampling = st.sidebar.slider("Phase oversampling (%)", 0, 100, 20, 5)
nex = st.sidebar.select_slider("NEX / Averages", options=[1,2,3,4,5,6,8], value=2)
grappa = st.sidebar.select_slider("GRAPPA", options=[1,2,3,4], value=2)
bandwidth = st.sidebar.select_slider(
    "Receiver bandwidth (Hz/px)",
    options=[130,200,260,330,400,500,650,780],
    value=260,
)
phase_direction = st.sidebar.selectbox(
    "Phase encoding direction",
    ["A → P","P → A","R → L","L → R","H → F","F → H"],
)

st.sidebar.divider()
st.sidebar.header("Slice prescription")
if "angles" not in st.session_state:
    st.session_state.angles = {"Sagittal":0, "Axial":0, "Coronal":0}
for plane in PLANES:
    st.session_state.angles[plane] = st.sidebar.slider(
        f"{plane} angle (°)", -45, 45,
        st.session_state.angles[plane], 1,
        key=f"angle_{plane}",
    )

# Educational calculations
read_resolution = fov / base_resolution
phase_matrix = round(base_resolution * (1 + phase_oversampling / 100))
phase_resolution = fov / base_resolution
voxel_volume = read_resolution * phase_resolution * slice_thickness

reference_voxel = (220 / 256) ** 2 * 5
relative_snr = (
    math.sqrt(nex)
    * voxel_volume / reference_voxel
    / math.sqrt(grappa)
    * math.sqrt(260 / bandwidth)
)

reference_burden = 307 * 2 / 2
relative_burden = (phase_matrix * nex / grappa) / reference_burden

# Shared 3D crosshair
if "crosshair" not in st.session_state:
    st.session_state.crosshair = {"x":0.5, "y":0.5, "z":0.5}
if "selected_plane" not in st.session_state:
    st.session_state.selected_plane = "Axial"

def panel_offset(plane):
    return PLANES.index(plane) * panel_width

def crosshair_pixel(plane):
    h, v = PANEL_AXES[plane]
    return (
        panel_offset(plane) + st.session_state.crosshair[h] * panel_width,
        st.session_state.crosshair[v] * H,
    )

def update_crosshair(plane, local_x, y):
    h, v = PANEL_AXES[plane]
    st.session_state.crosshair[h] = min(max(local_x / panel_width, 0), 1)
    st.session_state.crosshair[v] = min(max(y / H, 0), 1)

def dashed_line(draw, p1, p2, color, width=2, dash=10, gap=6):
    x1,y1 = p1
    x2,y2 = p2
    length = math.hypot(x2-x1, y2-y1)
    if length == 0:
        return
    ux, uy = (x2-x1)/length, (y2-y1)/length
    pos = 0
    while pos < length:
        end = min(pos+dash, length)
        draw.line(
            [(x1+ux*pos, y1+uy*pos), (x1+ux*end, y1+uy*end)],
            fill=color, width=width
        )
        pos += dash + gap

def draw_localizer(img):
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    for plane in PLANES:
        left = panel_offset(plane)
        right = left + panel_width
        cx, cy = crosshair_pixel(plane)

        # Own prescription line
        theta = math.radians(st.session_state.angles[plane])
        length = max(panel_width, H) * 1.5
        dx, dy = math.cos(theta)*length, math.sin(theta)*length
        pts = []
        for t in np.linspace(-1, 1, 500):
            px, py = cx+dx*t, cy+dy*t
            if left <= px <= right and 0 <= py <= H:
                pts.append((px,py))
        if len(pts) > 1:
            draw.line(pts, fill=PLANE_COLOR[plane], width=3)

        # Other planes' linked position
        for other in PLANES:
            if other == plane:
                continue
            h, v = PANEL_AXES[plane]
            oh, ov = PANEL_AXES[other]
            normal = ({"x","y","z"} - {oh,ov}).pop()
            if normal == h:
                xp = left + st.session_state.crosshair[normal] * panel_width
                dashed_line(draw, (xp,0), (xp,H), PLANE_COLOR[other])
            elif normal == v:
                yp = st.session_state.crosshair[normal] * H
                dashed_line(draw, (left,yp), (right,yp), PLANE_COLOR[other])

        r = 7
        draw.ellipse((cx-r,cy-r,cx+r,cy+r), outline=(255,255,255), width=2)
    return canvas

st.subheader("1. Click the localizer")
display_image = draw_localizer(base_image)
clicked = streamlit_image_coordinates(display_image, key="localizer_click")
if clicked is not None:
    x, y = float(clicked["x"]), float(clicked["y"])
    plane = PLANES[min(int(x // panel_width), 2)]
    update_crosshair(plane, x-panel_offset(plane), y)
    st.session_state.selected_plane = plane
    st.rerun()

st.image(display_image, width="stretch")
st.success(
    f"Selected plane: {st.session_state.selected_plane} | "
    f"Crosshair X={st.session_state.crosshair['x']:.2f}, "
    f"Y={st.session_state.crosshair['y']:.2f}, "
    f"Z={st.session_state.crosshair['z']:.2f}"
)
st.caption("Solid = own prescription. Dashed = linked positions of the other planes.")

st.divider()
st.subheader("2. Current acquisition")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Read resolution", f"{read_resolution:.2f} mm")
c2.metric("Phase resolution", f"{phase_resolution:.2f} mm")
c3.metric("Voxel volume", f"{voxel_volume:.2f} mm³")
c4.metric("Phase matrix", f"{phase_matrix}")
c1,c2,c3,c4 = st.columns(4)
c1.metric("NEX", str(nex))
c2.metric("GRAPPA", f"R = {grappa}")
c3.metric("Bandwidth", f"{bandwidth} Hz/px")
c4.metric("Relative SNR", f"{relative_snr:.2f}")

st.divider()
st.subheader("3. What happens when you change a parameter?")
messages = []
if slice_thickness < 5:
    messages.append("Slice thickness ↓ → voxel volume ↓ → signal per voxel ↓ → SNR tends to ↓.")
elif slice_thickness > 5:
    messages.append("Slice thickness ↑ → voxel volume ↑ → SNR tends to ↑, but through-plane resolution decreases.")
if base_resolution > 256:
    messages.append("Base resolution ↑ at constant FOV → smaller pixels → higher spatial resolution, with a tendency toward lower SNR.")
elif base_resolution < 256:
    messages.append("Base resolution ↓ at constant FOV → larger pixels → generally more SNR, but lower spatial resolution.")
if phase_oversampling > 0:
    messages.append("Phase oversampling ↑ → more phase-encoding steps → acquisition burden tends to increase while prescribed resolution is approximately maintained.")
if nex > 1:
    messages.append(f"NEX ↑ → SNR ∝ √NEX. For NEX={nex}, √NEX ≈ {math.sqrt(nex):.2f}.")
if grappa > 1:
    messages.append("GRAPPA ↑ → fewer phase-encoding acquisitions and usually shorter acquisition, but with an SNR penalty.")
if bandwidth > 260:
    messages.append("Bandwidth ↑ → noise tends to ↑ → SNR tends to ↓ (simplified: SNR ∝ 1/√BW).")
elif bandwidth < 260:
    messages.append("Bandwidth ↓ → noise tends to ↓ → SNR tends to ↑, with other bandwidth-related effects not modeled here.")
messages.append(
    f"Phase encoding direction = {phase_direction}. Changing it mainly changes the direction/distribution of phase-related artifacts."
)
for m in messages:
    st.info(m)

st.divider()
st.subheader("4. Try an MRI optimization experiment")
st.write(
    "Start with 5 mm thickness and NEX 2. Change only the thickness to 3 mm. "
    "Then increase NEX and observe how SNR is recovered at the cost of acquisition burden."
)
st.caption(
    "Educational simulator only. SNR is a simplified relative index; exact scanner SNR and scan time require "
    "sequence-specific parameters and reconstruction details."
)

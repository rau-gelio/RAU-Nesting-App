import streamlit as st
import ezdxf
import tempfile
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import math

from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from io import StringIO


st.set_page_config(layout="wide")
st.title("DXF Circle Nesting App - Count Based Blue Noise")

uploaded_file = st.file_uploader("Upload DXF", type=["dxf"])

st.sidebar.header("Interior Fill Circles")

fill_mode = st.sidebar.selectbox(
    "Fill placement mode",
    ["By Gap Grid", "By Circle Count - Blue Noise"]
)

fill_circle_dia = st.sidebar.number_input("Fill circle diameter", value=0.110, step=0.001)

if fill_mode == "By Gap Grid":
    fill_gap = st.sidebar.number_input("Fill circle gap", value=0.015, step=0.001)
    desired_fill_count = 0
else:
    desired_fill_count = st.sidebar.number_input("Desired fill circle count", value=500, step=1, min_value=1)
    fill_gap = st.sidebar.number_input("Minimum fill circle gap", value=0.015, step=0.001)

fill_edge_gap = st.sidebar.number_input("Fill gap from boundary", value=0.015, step=0.001)

fill_radius = fill_circle_dia / 2
fill_pitch = fill_circle_dia + fill_gap


def load_dxf(uploaded_file):
    data = uploaded_file.read()
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(data)
            temp_path = tmp.name

        return ezdxf.readfile(temp_path)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def ellipse_to_polygon(e, segments=360):
    center = e.dxf.center
    major = e.dxf.major_axis
    ratio = e.dxf.ratio

    major_len = (major.x ** 2 + major.y ** 2) ** 0.5
    minor_len = major_len * ratio
    angle = np.arctan2(major.y, major.x)

    pts = []

    for t in np.linspace(0, 2 * np.pi, segments):
        x0 = major_len * np.cos(t)
        y0 = minor_len * np.sin(t)

        x = center.x + x0 * np.cos(angle) - y0 * np.sin(angle)
        y = center.y + x0 * np.sin(angle) + y0 * np.cos(angle)

        pts.append((x, y))

    return Polygon(pts)


def extract_boundaries(doc):
    shapes = []
    counts = {}

    for e in doc.modelspace():
        etype = e.dxftype()
        counts[etype] = counts.get(etype, 0) + 1

        try:
            if etype == "ELLIPSE":
                poly = ellipse_to_polygon(e)
                shapes.append(("Ellipse", poly))

            elif etype == "CIRCLE":
                c = e.dxf.center
                r = e.dxf.radius
                poly = Point(c.x, c.y).buffer(r, resolution=180)
                shapes.append(("Circle", poly))

            elif etype == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in e.get_points()]
                if len(pts) >= 3 and e.closed:
                    poly = Polygon(pts)
                    if poly.is_valid and not poly.is_empty:
                        shapes.append(("Closed Polyline", poly))

            elif etype == "POLYLINE":
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                if len(pts) >= 3 and e.is_closed:
                    poly = Polygon(pts)
                    if poly.is_valid and not poly.is_empty:
                        shapes.append(("Closed Polyline", poly))

        except Exception:
            pass

    return shapes, counts


def build_fill_region(shapes, area_roles):
    fill_polys = []
    cutout_polys = []

    for i, role in area_roles.items():
        poly = shapes[i][1]

        if role == "Fill Inside":
            fill_polys.append(poly)

        elif role == "Subtract / Keep Empty":
            cutout_polys.append(poly)

    if not fill_polys:
        return None

    region = unary_union(fill_polys)

    if cutout_polys:
        region = region.difference(unary_union(cutout_polys))

    return region


def get_offset_polygon(poly, position, radius):
    if position == "Center On Boundary":
        return poly

    if position == "Inside Boundary":
        offset_poly = poly.buffer(-radius)
        if offset_poly.is_empty:
            return poly
        if offset_poly.geom_type == "MultiPolygon":
            offset_poly = max(offset_poly.geoms, key=lambda p: p.area)
        return offset_poly

    if position == "Outside Boundary":
        offset_poly = poly.buffer(radius)
        if offset_poly.is_empty:
            return poly
        if offset_poly.geom_type == "MultiPolygon":
            offset_poly = max(offset_poly.geoms, key=lambda p: p.area)
        return offset_poly

    return poly


def place_boundary_circles_by_count(poly, dia, count, position):
    radius = dia / 2
    offset_poly = get_offset_polygon(poly, position, radius)
    line = LineString(offset_poly.exterior.coords)
    length = line.length

    if length <= 0 or count <= 0:
        return [], 0

    actual_spacing = length / count
    actual_gap = actual_spacing - dia

    circles = []

    for i in range(count):
        distance = i * actual_spacing
        p = line.interpolate(distance)
        circles.append((p.x, p.y, radius))

    return circles, actual_gap


def optimize_boundary_count(length, dia, min_gap, max_gap):
    min_pitch = dia + min_gap
    max_pitch = dia + max_gap

    if length <= 0:
        return 0, 0

    max_count = int(length // min_pitch)
    min_count = int(length // max_pitch)

    if max_count < 1:
        return 1, length

    if min_count < 1:
        min_count = 1

    best_count = max_count
    best_spacing = length / best_count

    for count in range(max_count, min_count - 1, -1):
        spacing = length / count
        gap = spacing - dia

        if min_gap <= gap <= max_gap:
            best_count = count
            best_spacing = spacing
            break

    return best_count, best_spacing


def place_boundary_circles_by_gap(poly, dia, gap_mode, fixed_gap, min_gap, max_gap, position):
    radius = dia / 2
    offset_poly = get_offset_polygon(poly, position, radius)
    line = LineString(offset_poly.exterior.coords)
    length = line.length

    if length <= 0:
        return [], 0, 0

    if gap_mode == "Fixed Gap":
        spacing = dia + fixed_gap
        count = max(1, int(length // spacing))
        actual_spacing = length / count
        actual_gap = actual_spacing - dia

    else:
        count, actual_spacing = optimize_boundary_count(length, dia, min_gap, max_gap)
        actual_gap = actual_spacing - dia

    circles = []

    for i in range(count):
        distance = i * actual_spacing
        p = line.interpolate(distance)
        circles.append((p.x, p.y, radius))

    return circles, count, actual_gap


def circle_overlaps_existing(x, y, r, existing_circles, min_gap):
    for ex, ey, er in existing_circles:
        dist = ((x - ex) ** 2 + (y - ey) ** 2) ** 0.5
        required = r + er + min_gap

        if dist < required:
            return True

    return False


def nest_fill_grid(region, existing_circles):
    if region is None or region.is_empty:
        return []

    usable_region = region.buffer(-fill_edge_gap)

    if usable_region.is_empty:
        return []

    minx, miny, maxx, maxy = usable_region.bounds
    circles = []

    y = miny + fill_radius
    row = 0

    while y <= maxy - fill_radius:
        offset = 0 if row % 2 == 0 else fill_pitch / 2
        x = minx + fill_radius + offset

        while x <= maxx - fill_radius:
            test_circle = Point(x, y).buffer(fill_radius, resolution=32)

            if usable_region.contains(test_circle):
                if not circle_overlaps_existing(x, y, fill_radius, existing_circles, fill_gap):
                    circles.append((x, y, fill_radius))

            x += fill_pitch

        y += fill_pitch * 0.866
        row += 1

    return circles


def estimate_max_capacity(region, radius, min_gap, existing_circles):
    if region is None or region.is_empty:
        return 0

    usable_region = region.buffer(-fill_edge_gap)

    if usable_region.is_empty:
        return 0

    circle_area = math.pi * radius * radius
    packing_efficiency = 0.70

    blocked_area = 0

    for x, y, r in existing_circles:
        blocked_area += math.pi * (r + min_gap + radius) ** 2

    available_area = max(0, usable_region.area - blocked_area)

    return int((available_area * packing_efficiency) // circle_area)


def blue_noise_fill(region, existing_circles, desired_count):
    if region is None or region.is_empty:
        return [], 0

    usable_region = region.buffer(-fill_edge_gap)

    if usable_region.is_empty:
        return [], 0

    max_capacity = estimate_max_capacity(region, fill_radius, fill_gap, existing_circles)

    if desired_count > max_capacity:
        target_count = max_capacity
    else:
        target_count = desired_count

    if target_count <= 0:
        return [], max_capacity

    minx, miny, maxx, maxy = usable_region.bounds

    circles = []
    all_existing = existing_circles.copy()

    attempts = 0
    max_attempts = target_count * 500

    while len(circles) < target_count and attempts < max_attempts:
        attempts += 1

        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)

        test_circle = Point(x, y).buffer(fill_radius, resolution=24)

        if not usable_region.contains(test_circle):
            continue

        if circle_overlaps_existing(x, y, fill_radius, all_existing, fill_gap):
            continue

        circles.append((x, y, fill_radius))
        all_existing.append((x, y, fill_radius))

    return circles, max_capacity


def plot_preview(shapes, boundary_circles, fill_region=None, fill_circles=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, (name, poly) in enumerate(shapes):
        x, y = poly.exterior.xy
        ax.plot(x, y, linewidth=1)
        ax.text(poly.centroid.x, poly.centroid.y, str(i), fontsize=14)

    if fill_region is not None and not fill_region.is_empty:
        if fill_region.geom_type == "Polygon":
            x, y = fill_region.exterior.xy
            ax.fill(x, y, alpha=0.20)

            for interior in fill_region.interiors:
                ix, iy = interior.xy
                ax.plot(ix, iy, linewidth=1)

        elif fill_region.geom_type == "MultiPolygon":
            for poly in fill_region.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.20)

    for x, y, r in boundary_circles:
        ax.add_patch(plt.Circle((x, y), r, fill=False, linewidth=0.8))

    if fill_circles:
        for x, y, r in fill_circles:
            ax.add_patch(plt.Circle((x, y), r, fill=False, linewidth=0.5))

    ax.set_aspect("equal")
    ax.grid(True)
    st.pyplot(fig)


def export_dxf(shapes, boundary_circles, fill_circles):
    doc = ezdxf.new()
    msp = doc.modelspace()

    for name, poly in shapes:
        coords = list(poly.exterior.coords)
        msp.add_lwpolyline(coords, close=True)

    for x, y, r in boundary_circles:
        msp.add_circle((x, y), r)

    for x, y, r in fill_circles:
        msp.add_circle((x, y), r)

    output = StringIO()
    doc.write(output)
    return output.getvalue().encode("utf-8")


if uploaded_file:
    doc = load_dxf(uploaded_file)
    shapes, counts = extract_boundaries(doc)

    st.success("DXF loaded successfully")

    st.subheader("Entity Counts")
    st.json(counts)

    st.subheader("Detected Boundaries")
    st.write(f"Usable boundaries: {len(shapes)}")

    if not shapes:
        st.warning("No usable closed boundaries detected.")
        st.stop()

    area_roles = {}
    boundary_circles = []
    boundary_stats = []

    for i, (name, poly) in enumerate(shapes):
        st.markdown("---")
        st.subheader(f"Boundary {i}: {name}")

        boundary_mode = st.selectbox(
            f"Boundary {i} circle placement",
            ["Off", "Place Circles On Boundary"],
            key=f"boundary_mode_{i}"
        )

        if boundary_mode == "Place Circles On Boundary":
            b_position = st.selectbox(
                f"Boundary {i} circle position",
                ["Center On Boundary", "Inside Boundary", "Outside Boundary"],
                key=f"boundary_position_{i}"
            )

            b_dia = st.number_input(
                f"Boundary {i} circle diameter",
                value=0.110,
                step=0.001,
                key=f"boundary_dia_{i}"
            )

            b_place_mode = st.selectbox(
                f"Boundary {i} placement method",
                ["By Gap", "By Circle Count"],
                key=f"boundary_place_method_{i}"
            )

            if b_place_mode == "By Circle Count":
                b_count = st.number_input(
                    f"Boundary {i} desired circle count",
                    value=50,
                    step=1,
                    min_value=1,
                    key=f"boundary_count_{i}"
                )

                new_circles, actual_gap = place_boundary_circles_by_count(
                    poly,
                    b_dia,
                    int(b_count),
                    b_position
                )

                boundary_circles.extend(new_circles)

                boundary_stats.append(
                    {
                        "Boundary": i,
                        "Method": "By Count",
                        "Position": b_position,
                        "Requested": int(b_count),
                        "Placed": len(new_circles),
                        "Actual Gap": round(actual_gap, 4)
                    }
                )

            else:
                gap_mode = st.selectbox(
                    f"Boundary {i} gap mode",
                    ["Fixed Gap", "Auto Optimize Min/Max Gap"],
                    key=f"gap_mode_{i}"
                )

                if gap_mode == "Fixed Gap":
                    fixed_gap = st.number_input(
                        f"Boundary {i} fixed gap",
                        value=0.015,
                        step=0.001,
                        key=f"fixed_gap_{i}"
                    )

                    min_b_gap = fixed_gap
                    max_b_gap = fixed_gap

                else:
                    fixed_gap = 0.015

                    min_b_gap = st.number_input(
                        f"Boundary {i} minimum gap",
                        value=0.015,
                        step=0.001,
                        key=f"min_b_gap_{i}"
                    )

                    max_b_gap = st.number_input(
                        f"Boundary {i} maximum gap",
                        value=0.025,
                        step=0.001,
                        key=f"max_b_gap_{i}"
                    )

                    if max_b_gap < min_b_gap:
                        st.warning(f"Boundary {i}: maximum gap must be larger than minimum gap.")
                        max_b_gap = min_b_gap

                new_circles, count, actual_gap = place_boundary_circles_by_gap(
                    poly,
                    b_dia,
                    gap_mode,
                    fixed_gap,
                    min_b_gap,
                    max_b_gap,
                    b_position
                )

                boundary_circles.extend(new_circles)

                boundary_stats.append(
                    {
                        "Boundary": i,
                        "Method": gap_mode,
                        "Position": b_position,
                        "Requested": "-",
                        "Placed": count,
                        "Actual Gap": round(actual_gap, 4)
                    }
                )

        area_roles[i] = st.selectbox(
            f"Boundary {i} area role",
            ["Ignore Area", "Fill Inside", "Subtract / Keep Empty"],
            key=f"area_role_{i}"
        )

    fill_region = build_fill_region(shapes, area_roles)

    if fill_mode == "By Gap Grid":
        fill_circles = nest_fill_grid(fill_region, boundary_circles)
        max_capacity = len(fill_circles)

    else:
        fill_circles, max_capacity = blue_noise_fill(
            fill_region,
            boundary_circles,
            int(desired_fill_count)
        )

        if int(desired_fill_count) > max_capacity:
            st.warning(
                f"Requested {int(desired_fill_count)} fill circles, but estimated max capacity is about {max_capacity}. "
                f"The app placed {len(fill_circles)} circles."
            )

        elif len(fill_circles) < int(desired_fill_count):
            st.warning(
                f"Requested {int(desired_fill_count)} fill circles, but only {len(fill_circles)} could be placed without overlap."
            )

    st.markdown("---")
    st.subheader("Results")

    if boundary_stats:
        st.write("Boundary circle stats:")
        st.table(boundary_stats)

    st.write(f"Boundary circles: {len(boundary_circles)}")
    st.write(f"Fill circles: {len(fill_circles)}")
    st.write(f"Total circles: {len(boundary_circles) + len(fill_circles)}")

    plot_preview(shapes, boundary_circles, fill_region, fill_circles)

    dxf_data = export_dxf(shapes, boundary_circles, fill_circles)

    st.download_button(
        "Download DXF",
        data=dxf_data,
        file_name="blue_noise_count_based_circles.dxf",
        mime="application/dxf"
    )

else:
    st.info("Upload a DXF file to begin.")
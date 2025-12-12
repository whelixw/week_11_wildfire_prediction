"""Lightweight GeoPandas smoke tests for the wildfire project.

The script demonstrates five common spatial workflows so you can confirm the
GeoPandas install works end-to-end without touching the heavier notebook:

1. Basic GeoSeries creation (sanity-checks the GeoPandas/Shapely stack)
2. Loading wildfire station metadata and plotting their footprints
3. Spatially joining those stations to state polygons to compute counts
4. Overlaying station points on the per-state choropleth for quick visual QA
5. Plotting historical wildfire ignition points for context
6. Drawing 100 km station buffers and highlighting which wildfires fall inside them
7. Buffering a sample ignition point and finding nearby stations
8. Building a simple choropleth (station counts per state)

Running the script will drop PNGs under ``artifacts/geopandas_demo`` and print a
few textual summaries to the console so you can inspect the outputs quickly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point

ARTIFACT_DIR = Path("artifacts/geopandas_demo")
CITIES_PATH = Path("shared_data/cities.csv")
FIRE_PARQUET_PATH = Path("shared_data/wildfires_fires.parquet")
USA_STATES_URL = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
)
WGS84 = "EPSG:4326"
USA_EQUAL_AREA = "EPSG:5070"  # NAD83 / Conus Albers – decent for distance buffers


def ensure_artifact_dir() -> Path:
    """Create the output directory tree if it does not already exist."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR


def build_quick_geoseries() -> gpd.GeoSeries:
    """Return a toy GeoSeries to verify GeoPandas+Shapely plumbing."""
    demo_points = [
        Point(-121.4944, 38.5816),  # Sacramento, CA
        Point(-118.2437, 34.0522),  # Los Angeles, CA
        Point(-122.3321, 47.6062),  # Seattle, WA
    ]
    series = gpd.GeoSeries(demo_points, crs=WGS84)
    print(f"GeoPandas version: {gpd.__version__}")
    print("Sample GeoSeries bounds:", series.total_bounds)
    return series


def load_us_station_geometries(max_rows: int = 5000) -> gpd.GeoDataFrame:
    """Load US station metadata and convert to a GeoDataFrame."""
    if not CITIES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CITIES_PATH}. Pull shared_data/ from the repo before running."
        )
    cities = pd.read_csv(CITIES_PATH)
    us_cities = cities[cities["iso2"] == "US"].copy()
    if us_cities.empty:
        raise ValueError("No US stations found in shared_data/cities.csv")
    if max_rows:
        us_cities = us_cities.head(max_rows)
    gdf = gpd.GeoDataFrame(
        us_cities,
        geometry=gpd.points_from_xy(us_cities["longitude"], us_cities["latitude"]),
        crs=WGS84,
    )
    print(f"Loaded {len(gdf):,} US stations")
    return gdf


def plot_station_scatter(stations: gpd.GeoDataFrame, out_dir: Path) -> Path:
    """Scatter the stations and save the PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))
    stations.plot(ax=ax, markersize=5, alpha=0.6, color="tab:orange")
    ax.set_title("US Weather Stations (sample)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    path = out_dir / "stations_scatter.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def load_us_states() -> gpd.GeoDataFrame:
    """Fetch US state boundaries (GeoJSON over HTTP, fall    to country outline)."""
    try:
        states = gpd.read_file(USA_STATES_URL)
        # Normalize column names from the GeoJSON (varies on source revs)
        if "name" not in states.columns:
            raise ValueError("State GeoJSON missing 'name' column")
    except Exception as err:  # pragma: no cover - network fallback
        print(
            "State-level GeoJSON fetch failed; falling back to Natural Earth country outline."
        )
        print(f"Reason: {err}")
        natural_earth = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        states = natural_earth[natural_earth["iso_a3"] == "USA"].copy()
        states["name"] = "United States"
    return states.to_crs(WGS84)


def compute_state_station_counts(
    stations: gpd.GeoDataFrame, states: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Spatially join stations to states and return polygons with counts."""
    joined = gpd.sjoin(stations, states[["name", "geometry"]], how="left", predicate="within")
    counts = joined.groupby("name").size().rename("station_count")
    states_with_counts = states.merge(counts, left_on="name", right_index=True, how="left")
    states_with_counts["station_count"] = states_with_counts["station_count"].fillna(0).astype(int)
    print("Top 5 states by station count:")
    print(states_with_counts.nlargest(5, "station_count")[ ["name", "station_count"] ])
    return states_with_counts


def plot_state_counts(states_with_counts: gpd.GeoDataFrame, out_dir: Path) -> Path:
    """Save a quick choropleth showing station density by state."""
    fig, ax = plt.subplots(figsize=(9, 6))
    states_with_counts.plot(
        ax=ax,
        column="station_count",
        cmap="OrRd",
        legend=True,
        linewidth=0.5,
        edgecolor="gray",
        missing_kwds={"color": "lightgray", "label": "No data"},
    )
    ax.set_axis_off()
    ax.set_title("Station Counts by State")
    path = out_dir / "station_counts.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_overlay_counts_and_scatter(
    states_with_counts: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, out_dir: Path
) -> Path:
    """Overlay station points on the per-state choropleth for quick visual QA."""
    fig, ax = plt.subplots(figsize=(9, 6))
    states_with_counts.plot(
        ax=ax,
        column="station_count",
        cmap="OrRd",
        legend=True,
        linewidth=0.5,
        edgecolor="gray",
        alpha=0.9,
        missing_kwds={"color": "lightgray", "label": "No data"},
    )
    station_ax = stations.plot(
        ax=ax,
        color="navy",
        markersize=5,
        alpha=0.5,
        label="Stations",
    )
    ax.set_axis_off()
    ax.set_title("Station Counts with Station Locations")
    # Extract the PathCollection produced by GeoPandas for the legend.
    station_handle = station_ax.collections[0] if station_ax.collections else None
    if station_handle is not None:
        station_handle.set_label("Stations")
        ax.legend(handles=[station_handle], loc="lower left", frameon=True)
    path = out_dir / "station_counts_overlay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def load_wildfire_points(max_points: int = 50000) -> gpd.GeoDataFrame:
    """Load wildfire ignition records (parquet) and return a GeoDataFrame sample."""
    if not FIRE_PARQUET_PATH.exists():
        raise FileNotFoundError(
            "Wildfire parquet missing; run scripts/filter_data.py to materialize data/wildfires_fires.parquet."
        )
    fires = pd.read_parquet(FIRE_PARQUET_PATH)
    required_cols = {"LATITUDE", "LONGITUDE"}
    missing_cols = required_cols - set(fires.columns)
    if missing_cols:
        raise ValueError(
            f"Parquet file is missing required coordinate columns: {', '.join(sorted(missing_cols))}"
        )
    fires = fires.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    optional_cols = ["FOD_ID", "STATE", "DISCOVERY_DATE"]
    keep_cols = ["LATITUDE", "LONGITUDE"] + [c for c in optional_cols if c in fires.columns]
    fires = fires[keep_cols]
    if fires.empty:
        raise ValueError("No wildfire coordinates found in shared_data/wildfires_fires.parquet")
    if max_points and len(fires) > max_points:
        fires = fires.sample(n=max_points, random_state=42)
    gdf = gpd.GeoDataFrame(
        fires,
        geometry=gpd.points_from_xy(fires["LONGITUDE"], fires["LATITUDE"]),
        crs=WGS84,
    )
    print(f"Loaded {len(gdf):,} wildfire ignition points")
    return gdf


def plot_wildfire_points(
    fires: gpd.GeoDataFrame, states: gpd.GeoDataFrame, out_dir: Path
) -> Path:
    """Plot wildfire ignition points atop state boundaries."""
    fig, ax = plt.subplots(figsize=(9, 6))
    states.boundary.plot(ax=ax, color="lightgray", linewidth=0.5)
    fires.plot(ax=ax, markersize=2, color="firebrick", alpha=0.5)
    ax.set_axis_off()
    ax.set_title("Wildfire Ignitions (sample)")
    path = out_dir / "wildfire_points.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def build_station_buffers(
    stations: gpd.GeoDataFrame, radius_km: float = 100.0
) -> Tuple[gpd.GeoDataFrame, object]:
    """Return buffers in the equal-area CRS plus their unary union."""
    if stations.empty:
        raise ValueError("Station GeoDataFrame is empty; cannot build buffers")
    stations_proj = stations.to_crs(USA_EQUAL_AREA)
    buffers_proj = gpd.GeoDataFrame(
        geometry=stations_proj.geometry.buffer(radius_km * 1000.0), crs=USA_EQUAL_AREA
    )
    buffer_union = buffers_proj.geometry.unary_union
    return buffers_proj, buffer_union


def plot_station_buffers_with_wildfires(
    stations: gpd.GeoDataFrame,
    fires: gpd.GeoDataFrame,
    states: gpd.GeoDataFrame,
    out_dir: Path,
    radius_km: float = 100.0,
) -> Path:
    """Overlay station buffers and color wildfires by whether they fall inside."""
    buffers_proj, buffer_union = build_station_buffers(stations, radius_km)
    buffers = buffers_proj.to_crs(WGS84)

    fires_proj = fires.to_crs(USA_EQUAL_AREA)
    fires = fires.copy()
    fires["within_buffer"] = fires_proj.geometry.within(buffer_union)

    fig, ax = plt.subplots(figsize=(9, 6))
    states.boundary.plot(ax=ax, color="lightgray", linewidth=0.5)
    buffers.plot(ax=ax, color="royalblue", alpha=0.15, edgecolor="royalblue", linewidth=0.3)
    station_ax = stations.plot(ax=ax, color="navy", markersize=4, alpha=0.6, label="Stations")

    inside = fires[fires["within_buffer"]]
    outside = fires[~fires["within_buffer"]]
    outside_ax = outside.plot(
        ax=ax,
        markersize=2,
        color="#bdbdbd",
        alpha=0.4,
        label="Wildfires outside buffer",
    )
    inside_ax = inside.plot(
        ax=ax,
        markersize=3,
        color="#d7301f",
        alpha=0.6,
        label="Wildfires inside buffer",
    )

    ax.set_axis_off()
    ax.set_title(f"Station Buffers (~{int(radius_km)} km) and Wildfires")

    handles = []
    labels = []
    if hasattr(station_ax, "collections") and station_ax.collections:
        handles.append(station_ax.collections[0])
        labels.append("Stations")
    if hasattr(outside_ax, "collections") and outside_ax.collections:
        handles.append(outside_ax.collections[0])
        labels.append("Wildfires outside buffer")
    if hasattr(inside_ax, "collections") and inside_ax.collections:
        handles.append(inside_ax.collections[0])
        labels.append("Wildfires inside buffer")
    if handles:
        ax.legend(handles, labels, loc="lower left", frameon=True)

    path = out_dir / "station_buffers_wildfires_overlay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def sample_fire_point(stations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Use a known ignition point if parquet exists, else fall back to a station."""
    if FIRE_PARQUET_PATH.exists():
        fires = pd.read_parquet(FIRE_PARQUET_PATH)
        subset = fires[["LONGITUDE", "LATITUDE", "STATE"]].dropna().head(1)
        if len(subset):
            fire_gdf = gpd.GeoDataFrame(
                subset,
                geometry=gpd.points_from_xy(subset["LONGITUDE"], subset["LATITUDE"]),
                crs=WGS84,
            )
            print("Using ignition point extracted from data/wildfires_fires.parquet")
            return fire_gdf
    print("Falling back to the first station as a synthetic ignition point")
    return stations.iloc[[0]][["geometry", "state"]].copy()


def buffer_fire_point(
    fire_point: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, radius_km: float = 50.0
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create a buffer around the fire point and filter stations inside it."""
    fire_proj = fire_point.to_crs(USA_EQUAL_AREA)
    stations_proj = stations.to_crs(USA_EQUAL_AREA)
    buffer_geom = fire_proj.buffer(radius_km * 1000.0)
    buffer_gdf = gpd.GeoDataFrame(geometry=buffer_geom, crs=USA_EQUAL_AREA).to_crs(WGS84)
    stations_in_buffer = stations_proj[stations_proj.within(buffer_geom.iloc[0])].to_crs(WGS84)
    print(
        f"Found {len(stations_in_buffer):,} stations within {radius_km:.0f} km of the sample point"
    )
    return buffer_gdf, stations_in_buffer


def plot_buffer_region(
    fire_point: gpd.GeoDataFrame,
    buffer_gdf: gpd.GeoDataFrame,
    nearby_stations: gpd.GeoDataFrame,
    out_dir: Path,
) -> Path:
    """Render the buffer region and nearby stations."""
    fig, ax = plt.subplots(figsize=(6, 6))
    buffer_gdf.plot(ax=ax, color="lightblue", alpha=0.3, edgecolor="steelblue")
    fire_point.plot(ax=ax, color="red", markersize=50, label="Ignition")
    if not nearby_stations.empty:
        nearby_stations.plot(ax=ax, color="black", markersize=20, label="Stations")
    else:
        ax.text(
            0.5,
            0.05,
            "No stations within buffer",
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    ax.set_title("Stations within 50 km")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    path = out_dir / "buffer_demo.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    out_dir = ensure_artifact_dir()
    build_quick_geoseries()
    stations = load_us_station_geometries()
    scatter_path = plot_station_scatter(stations, out_dir)
    states = load_us_states()
    states_with_counts = compute_state_station_counts(stations, states)
    choropleth_path = plot_state_counts(states_with_counts, out_dir)
    overlay_path = plot_overlay_counts_and_scatter(states_with_counts, stations, out_dir)
    try:
        fires = load_wildfire_points()
    except FileNotFoundError as missing_data_err:
        wildfire_points_path = None
        buffer_overlay_path = None
        print(missing_data_err)
    else:
        wildfire_points_path = plot_wildfire_points(fires, states, out_dir)
        buffer_overlay_path = plot_station_buffers_with_wildfires(stations, fires, states, out_dir)
    fire_point = sample_fire_point(stations)
    buffer_gdf, stations_in_buffer = buffer_fire_point(fire_point, stations)
    buffer_path = plot_buffer_region(fire_point, buffer_gdf, stations_in_buffer, out_dir)

    print("Artifacts written to:")
    for path in (
        scatter_path,
        choropleth_path,
        overlay_path,
        wildfire_points_path,
        buffer_overlay_path,
        buffer_path,
    ):
        if path is not None:
            print(f" - {path}")


if __name__ == "__main__":
    main()

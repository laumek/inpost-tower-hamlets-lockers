# %% [markdown]
# # InPost locker coverage in Tower Hamlets
# **Question:** What share of Tower Hamlets residents live within a ~7-minute walk of an
# InPost locker, where are the gaps, and where would the next 5 lockers add most coverage?
#
# Runs as a plain script (`python tower_hamlets_lockers.py`) or cell-by-cell in
# VS Code / Jupyter (the `# %%` markers split it into cells).
#
# Manual downloads needed first (see README.md):
#   data/census2021-ts001-oa.csv   -> Census 2021 TS001 population by Output Area (Nomis)
#   data/oa_boundaries.*           -> Output Areas Dec 2021 boundaries (ONS Open Geography)

# %% Setup
import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

DATA = Path("data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

BOROUGH = "London Borough of Tower Hamlets, London, UK"
CRS = "EPSG:27700"          # British National Grid, units in metres
WALK_SPEED_KMH = 4.8        # typical urban walking speed
WALK_MINUTES = 7            # InPost's own headline accessibility measure
NETWORK_RADIUS_M = WALK_SPEED_KMH * 1000 / 60 * WALK_MINUTES   # ~560 m along streets
STRAIGHT_RADIUS_M = 500     # straight-line approximation (streets are never straight)
N_NEW_SITES = 5
USE_NETWORK = True          # set False for a fast straight-line-only run
SHOW_WATER = True           # set False to skip the water layer on the map (one extra OSM download)

ox.settings.use_cache = True
ox.settings.log_console = True   # shows OSM download progress, so a slow download doesn't look like a freeze

def flat_index(gdf):
    """OSMnx 1.x indexes features by (element_type, osmid); 2.x by (element, id).
    Flatten either into a plain 'osmid' column."""
    gdf = gdf.reset_index()
    return gdf.rename(columns={"id": "osmid", "element": "element_type"})

# %% 1. Borough boundary
borough = ox.geocode_to_gdf(BOROUGH).to_crs(CRS)
borough_poly = borough.geometry.iloc[0]
# Lockers just outside the boundary still serve residents near the edge
search_area = gpd.GeoSeries([borough_poly.buffer(NETWORK_RADIUS_M + 100)], crs=CRS)
search_poly_wgs = search_area.to_crs(4326).iloc[0]
print(f"Borough area: {borough_poly.area / 1e6:.1f} km²")

# %% 2. InPost lockers from OpenStreetMap
raw = ox.features_from_polygon(search_poly_wgs, tags={"amenity": "parcel_locker"})
raw = raw.to_crs(CRS)
raw["geometry"] = raw.geometry.representative_point()   # some lockers are mapped as shapes

def is_inpost(row):
    for col in ("brand", "operator", "name", "network"):
        val = row.get(col)
        if isinstance(val, str) and "inpost" in val.lower():
            return True
    return row.get("brand:wikidata") == "Q3182097"   # InPost's Wikidata ID

lockers = raw[raw.apply(is_inpost, axis=1)].copy()
lockers = flat_index(lockers)[["osmid", "geometry"] +
                                [c for c in ("name", "ref", "addr:street", "addr:postcode") if c in lockers.columns]]
lockers["in_borough"] = lockers.within(borough_poly)
print(f"Parcel lockers in OSM (any brand): {len(raw)}")
print(f"InPost lockers: {len(lockers)} ({lockers['in_borough'].sum()} inside the borough)")
if len(lockers) < 30:
    print("WARNING: very few InPost lockers found. OSM is likely incomplete here; "
          "treat coverage as a lower bound and check the gap areas on InPost's map.")

# Validation sheet: tick off a sample against the locker finder on inpost.co.uk
val = lockers.to_crs(4326)
val_sheet = pd.DataFrame({
    "osmid": val["osmid"],
    "name": val.get("name"),
    "lat": val.geometry.y.round(6),
    "lon": val.geometry.x.round(6),
    "confirmed_on_inpost_map": "",   # fill with Y / N
})
val_sheet.sample(min(20, len(val_sheet)), random_state=1).to_csv(OUT / "validation_sample.csv", index=False)

# %% 2b. Hand-collected lockers from InPost's own locker finder (replaces OSM if present)
# data/inpost_lockers_manual.csv columns: name, address, postcode, type (locker/shop), lat, lon
# lat/lon are optional; rows without them are placed at their postcode via postcodes.io
# (free, open ONS-based geocoder; a postcode is ~15 addresses, so typically within ~100 m).
MANUAL = DATA / "inpost_lockers_manual.csv"
osm_lockers = lockers.copy()
data_source = "OpenStreetMap"
osm_match_rate = None
all_points = None
if MANUAL.exists():
    import requests
    man = pd.read_csv(MANUAL, dtype=str).fillna("")
    man["postcode"] = man["postcode"].str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    need = man.loc[(man["lat"] == "") | (man["lon"] == ""), "postcode"].unique().tolist()
    lookup = {}
    for i in range(0, len(need), 100):   # bulk endpoint takes up to 100 postcodes
        r = requests.post("https://api.postcodes.io/postcodes", json={"postcodes": need[i:i + 100]}, timeout=30)
        r.raise_for_status()
        for item in r.json()["result"]:
            if item["result"]:
                lookup[item["query"]] = (item["result"]["latitude"], item["result"]["longitude"])
    bad = [pc for pc in need if pc not in lookup]
    if bad:
        print(f"WARNING: {len(bad)} postcodes not found, rows skipped: {bad}")
    for idx, row in man.iterrows():
        if (row["lat"] == "" or row["lon"] == "") and row["postcode"] in lookup:
            man.at[idx, "lat"], man.at[idx, "lon"] = lookup[row["postcode"]]
    man = man[(man["lat"] != "") & (man["lon"] != "")]
    if "type" not in man.columns:
        man["type"] = "locker"
    man["type"] = man["type"].str.lower().str.strip().replace("", "locker")
    lockers = gpd.GeoDataFrame(
        man[["name", "address", "postcode", "type"]].reset_index(drop=True),
        geometry=gpd.points_from_xy(man["lon"].astype(float), man["lat"].astype(float)),
        crs=4326).to_crs(CRS)
    lockers["osmid"] = range(len(lockers))   # placeholder id so later code works unchanged
    lockers["in_borough"] = lockers.within(borough_poly)
    # Points beyond the walking network's edge would snap to its boundary and look closer than they are
    reach_poly = search_area.iloc[0]
    far = ~lockers.within(reach_poly)
    if far.any():
        print(f"Dropped {far.sum()} points too far outside the borough to affect coverage: "
              f"{lockers.loc[far, 'name'].tolist()}")
        lockers = lockers[~far].reset_index(drop=True)
    all_points = lockers.copy()                      # lockers + shops, for InPost's own headline measure
    lockers = lockers[lockers["type"] == "locker"].reset_index(drop=True)
    shops = all_points[all_points["type"] != "locker"]
    print(f"Hand-collected: {len(lockers)} lockers, {len(shops)} shop points")
    data_source = "InPost locker finder (hand-collected)"
    # How complete is OSM? Share of real lockers with an OSM InPost locker within 100 m
    if len(osm_lockers):
        d = lockers.geometry.apply(lambda p: osm_lockers.distance(p).min())
        osm_match_rate = float((d <= 100).mean())
    else:
        osm_match_rate = 0.0
    print(f"Using {len(lockers)} hand-collected lockers ({lockers['in_borough'].sum()} inside the borough)")
    print(f"OSM completeness: {osm_match_rate:.0%} of these have an OSM InPost locker within 100 m")

# %% 3. Population by Output Area (Census 2021)
def load_population(path):
    """Works with both Nomis bulk files and Nomis query downloads, which add
    title rows and footnotes around the table. Keeps only rows with an OA code."""
    import csv
    oa_code = re.compile(r"E00\d{6}")
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f) if any(oa_code.fullmatch(x.strip()) for x in r)]
    df = pd.DataFrame(rows)
    code_col = max(df.columns, key=lambda c: df[c].str.strip().str.fullmatch(r"E00\d{6}").sum())
    num = df.drop(columns=[code_col]).apply(
        lambda s: pd.to_numeric(s.str.replace(",", "").str.strip(), errors="coerce"))
    num = num.loc[:, num.notna().all() & (num.nunique() > 1)]   # drops text and constant 'date' columns
    pop_col = num.sum().idxmax()   # 'Total' is always the largest of total / household / communal
    out = pd.DataFrame({"OA21CD": df[code_col].str.strip(), "population": num[pop_col]})
    print(f"Population rows loaded: {len(out)}, residents: {out['population'].sum():,.0f}")
    return out

pop = load_population(DATA / "census2021-ts001-oa.csv")

oa_file = next(p for ext in (".geojson", ".gpkg", ".shp", ".zip")
               for p in DATA.glob(f"oa_boundaries*{ext}"))
print(f"Boundary file: {oa_file.name}")
oas = gpd.read_file(oa_file, mask=borough.to_crs(gpd.read_file(oa_file, rows=1).crs))
oas = oas.to_crs(CRS)
code_col = next(c for c in oas.columns if c.upper().startswith("OA21CD"))
oas = oas.rename(columns={code_col: "OA21CD"})[["OA21CD", "geometry"]]
# Keep OAs whose centre is inside the borough (mask also returns neighbours that touch it)
oas = oas[oas.geometry.representative_point().within(borough_poly)]
oas = oas.merge(pop, on="OA21CD", how="left")
missing = oas["population"].isna().sum()
assert missing == 0, f"{missing} Output Areas have no population match; check the census file"
oas["centre"] = oas.geometry.representative_point()
total_pop = oas["population"].sum()
print(f"Output Areas: {len(oas)}, residents: {total_pop:,.0f}")
# Sanity check: Tower Hamlets had ~310k residents in Census 2021

# %% 4a. Coverage, straight-line
centres = gpd.GeoDataFrame(oas[["OA21CD", "population"]], geometry=oas["centre"], crs=CRS)
dist_to_nearest = centres.geometry.apply(lambda p: lockers.distance(p).min())
oas["dist_straight_m"] = dist_to_nearest.values
oas["covered_straight"] = oas["dist_straight_m"] <= STRAIGHT_RADIUS_M
cov_straight = oas.loc[oas["covered_straight"], "population"].sum() / total_pop
print(f"Straight-line coverage (≤{STRAIGHT_RADIUS_M} m): {cov_straight:.1%}")

# %% 4b. Coverage, walking network (streets, paths, footbridges)
if USE_NETWORK:
    G = ox.graph_from_polygon(search_poly_wgs, network_type="walk")
    G = ox.project_graph(G, to_crs=CRS)
    oas["node"] = ox.distance.nearest_nodes(G, oas["centre"].x, oas["centre"].y)
    lockers["node"] = ox.distance.nearest_nodes(G, lockers.geometry.x, lockers.geometry.y)
    # One multi-source search: walking distance from every OA to its nearest locker
    walk_dist = nx.multi_source_dijkstra_path_length(G, set(lockers["node"]), weight="length")
    oas["dist_walk_m"] = oas["node"].map(walk_dist).fillna(np.inf)
    oas["covered"] = oas["dist_walk_m"] <= NETWORK_RADIUS_M
    cov_walk = oas.loc[oas["covered"], "population"].sum() / total_pop
    print(f"Walking coverage (≤{WALK_MINUTES} min ≈ {NETWORK_RADIUS_M:.0f} m): {cov_walk:.1%}")
    cov_walk_all = None
    if all_points is not None and len(all_points) > len(lockers):
        nodes_all = ox.distance.nearest_nodes(G, all_points.geometry.x, all_points.geometry.y)
        dist_all = nx.multi_source_dijkstra_path_length(G, set(nodes_all), weight="length")
        covered_all = oas["node"].map(dist_all).fillna(np.inf) <= NETWORK_RADIUS_M
        cov_walk_all = oas.loc[covered_all, "population"].sum() / total_pop
        print(f"Walking coverage incl. shop points (comparable to InPost's 7-minute measure): {cov_walk_all:.1%}")
else:
    cov_walk_all = None
    oas["covered"] = oas["covered_straight"]
    cov_walk = None

# %% 5. Candidate sites: places InPost already partners with or could plausibly use
cand_tags = {
    "shop": ["supermarket", "convenience"],
    "amenity": ["fuel"],
    "railway": ["station"],
}
cands = ox.features_from_polygon(borough.to_crs(4326).iloc[0].geometry, tags=cand_tags).to_crs(CRS)
cands["geometry"] = cands.geometry.representative_point()
cands = flat_index(cands)
def site_type(r):
    if r.get("railway") == "station":
        return "station"
    if r.get("amenity") == "fuel":
        return "fuel station"
    return r.get("shop")
cands["type"] = cands.apply(site_type, axis=1)
cands["label"] = cands.get("name", pd.Series(index=cands.index)).fillna(cands["type"])
cands = cands[["osmid", "label", "type", "geometry"]].drop_duplicates("osmid")
print(f"Candidate sites: {len(cands)} ({cands['type'].value_counts().to_dict()})")

if USE_NETWORK:
    cands["node"] = ox.distance.nearest_nodes(G, cands.geometry.x, cands.geometry.y)
    node_to_oas = oas.groupby("node").groups

def reachable_oas(cand):
    """Output Areas that would be within a 7-minute walk of this candidate."""
    if USE_NETWORK:
        reach = nx.single_source_dijkstra_path_length(G, cand["node"], cutoff=NETWORK_RADIUS_M, weight="length")
        idx = [i for n in reach for i in node_to_oas.get(n, [])]
        return set(idx)
    return set(oas.index[oas["centre"].distance(cand.geometry) <= STRAIGHT_RADIUS_M])

cands["reach"] = cands.apply(reachable_oas, axis=1)

# %% 6. Greedy maximal-coverage: add the site that covers the most uncovered residents, repeat
uncovered = set(oas.index[~oas["covered"]])
chosen = []
for step in range(1, N_NEW_SITES + 1):
    gains = cands["reach"].apply(lambda r: oas.loc[list(r & uncovered), "population"].sum())
    best = gains.idxmax()
    if gains[best] == 0:
        break
    newly = cands.at[best, "reach"] & uncovered
    uncovered -= newly
    chosen.append({
        "rank": step,
        "site": cands.at[best, "label"],
        "type": cands.at[best, "type"],
        "residents_added": int(gains[best]),
        "coverage_after": 1 - oas.loc[list(uncovered), "population"].sum() / total_pop,
        "geometry": cands.at[best, "geometry"],
    })
    cands = cands.drop(best)

new_sites = gpd.GeoDataFrame(chosen, crs=CRS)
oas["newly_covered"] = oas.index.isin(set(oas.index[~oas["covered"]]) - uncovered)
print(new_sites.drop(columns="geometry").to_string(index=False))

# %% 7. Gap list: the biggest uncovered areas, to check against InPost's own locker map
gaps = oas[~oas["covered"]].sort_values("population", ascending=False).head(15)
gap_pts = gpd.GeoSeries(gaps["centre"], crs=CRS).to_crs(4326)
gap_check = pd.DataFrame({
    "OA21CD": gaps["OA21CD"], "population": gaps["population"],
    "walk_to_nearest_m": gaps.get("dist_walk_m", gaps["dist_straight_m"]).round(),
    "lat": gap_pts.y.round(6), "lon": gap_pts.x.round(6),
    "locker_actually_nearby": "",   # check on inpost.co.uk; fill Y / N
})
gap_check.to_csv(OUT / "gap_check.csv", index=False)

# %% 8. Map
# Context layers are drawn from OpenStreetMap data already downloaded, rather than map tiles:
# no API keys, no tile-server usage limits, and a cleaner look.
water_tags = {"natural": "water", "waterway": ["dock", "riverbank", "canal"], "water": True}
water = None
if SHOW_WATER:
    try:
        print("Downloading water layer for the map (can take a few minutes the first time)...")
        water = ox.features_from_polygon(search_poly_wgs, tags=water_tags).to_crs(CRS)
        water = water[water.geom_type.isin(["Polygon", "MultiPolygon"])]
    except Exception as e:
        print(f"Water layer skipped: {e}")
        water = None

fig, ax = plt.subplots(figsize=(9, 9))
oas[oas["covered"]].plot(ax=ax, color="#c9d7ea", edgecolor="white", linewidth=0.2)
oas[~oas["covered"] & ~oas["newly_covered"]].plot(ax=ax, color="#e8a39c", edgecolor="white", linewidth=0.2)
oas[oas["newly_covered"]].plot(ax=ax, color="#f3d27a", edgecolor="white", linewidth=0.2)
if USE_NETWORK:
    streets = ox.graph_to_gdfs(G, nodes=False)
    streets.clip(borough_poly.buffer(50)).plot(ax=ax, color="#555555", linewidth=0.25, alpha=0.35)
if water is not None and len(water):
    water.clip(borough_poly.buffer(300)).plot(ax=ax, color="#9ec5e0", edgecolor="none", zorder=2)
borough.boundary.plot(ax=ax, color="#1f497d", linewidth=1.2, zorder=3)
if all_points is not None and len(all_points) > len(lockers):
    shop_pts = all_points[(all_points["type"] != "locker") & all_points["in_borough"]]
    shop_pts.plot(ax=ax, facecolor="none", edgecolor="#1f497d", marker="o", markersize=12,
                  linewidth=0.7, zorder=4, label="InPost shop")
locker_label = "InPost locker" if MANUAL.exists() else "InPost locker (OSM)"
lockers[lockers["in_borough"]].plot(ax=ax, color="#1f497d", markersize=14, zorder=4, label=locker_label)
if len(new_sites):
    new_sites.plot(ax=ax, color="#d35400", marker="*", markersize=260, edgecolor="black",
                   linewidth=0.6, zorder=5, label="Proposed site")
    for _, r in new_sites.iterrows():
        ax.annotate(str(r["rank"]), (r.geometry.x, r.geometry.y), xytext=(6, 6),
                    textcoords="offset points", fontsize=10, fontweight="bold", zorder=6)
from matplotlib.patches import Patch
handles, _ = ax.get_legend_handles_labels()
handles += [Patch(color="#c9d7ea", label="Within 7-min walk of a locker"),
            Patch(color="#e8a39c", label="Gap"),
            Patch(color="#f3d27a", label="Covered by proposed sites")]
ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True)
ax.set_axis_off()
ax.set_title("InPost locker coverage, Tower Hamlets", fontsize=13, color="#1f497d", loc="left")
ax.annotate("Streets and water © OpenStreetMap contributors. Population: ONS Census 2021.",
            xy=(1, 0), xycoords="axes fraction", ha="right", va="bottom", fontsize=6, color="#666666")
plt.tight_layout()
plt.savefig(OUT / "coverage_map.png", dpi=250)
plt.close(fig)   # saved to file; plt.show() would pause the script until the window is closed

# %% 9. Results summary 
results = {
    "borough": "Tower Hamlets",
    "locker_data_source": data_source,
    "osm_inpost_lockers_found": int(len(osm_lockers)),
    "osm_completeness_vs_inpost": round(osm_match_rate, 4) if osm_match_rate is not None else None,
    "residents": int(total_pop),
    "output_areas": int(len(oas)),
    "lockers_used_total": int(len(lockers)),
    "lockers_used_in_borough": int(lockers["in_borough"].sum()),
    "coverage_straight_line": round(cov_straight, 4),
    "coverage_walking": round(cov_walk, 4) if cov_walk is not None else None,
    "coverage_walking_incl_shops": round(cov_walk_all, 4) if cov_walk_all is not None else None,
    "shop_points": int(len(all_points) - len(lockers)) if all_points is not None else None,
    "walk_radius_m": round(NETWORK_RADIUS_M),
    "uncovered_residents": int(oas.loc[~oas["covered"], "population"].sum()),
    "proposed_sites": new_sites.drop(columns="geometry").to_dict(orient="records"),
    "coverage_after_new_sites": round(new_sites["coverage_after"].iloc[-1], 4) if len(new_sites) else None,
    "candidate_counts": cands["type"].value_counts().to_dict(),
}
(OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
print(json.dumps(results, indent=2, default=float))
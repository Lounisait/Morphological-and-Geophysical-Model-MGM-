#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main entry point for the coupled Morphological and Geophysical Model (MGM)."""

import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["text.usetex"] = False
from matplotlib import cm
from matplotlib.colors import Normalize, LightSource
from matplotlib.lines import Line2D
try:
    from matplotlib.colors import TwoSlopeNorm
except ImportError:
    class TwoSlopeNorm(Normalize):
        """MGM internal routine."""

        def __init__(self, vmin=None, vcenter=0.0, vmax=None):
            super().__init__(vmin=vmin, vmax=vmax)
import matplotlib.gridspec as gridspec
from tqdm import tqdm


from landlab import RasterModelGrid
from landlab.components import (
    FlowAccumulator,
    SpaceLargeScaleEroder,
    ExponentialWeatherer,
    DepthDependentTaylorDiffuser
)


from function_flexure import calc_load, calc_flex_extended
from Gravi import compute_gravity_anomalies
from mag_harmonized import (
    compute_mag_evolution_column_prism,
    compute_mag_evolution_column_prism_2layers,
    compute_mag_evolution_column_prism_case1,
    compute_mag_simple_flat_layer,
    compute_mag_evolution_column_prism_case2_lenses_lobe
)

from Topo import (
    generate_topography,
    detect_crater_diameter_hybrid,
    creer_masque_circulaire,
)






def initialize_simulation():
    """MGM internal routine."""
    params = {}
    
    
    
    
    
    params['file_path'] = "data/surf_profile_2000.csv"
    params['file_path_grav'] = "data/bouguer_2000.csv"
    params['xy_space'] = 500  
    params['limit'] = 60000    
    params['output_root'] = "outputs"
    params['initial_seed_path'] = None
    params['boundary_mode'] = "all_open"
    
    
    
    
    
    params['t'] = 1200100   
    params['dt'] = 100      
    params['nt'] = int((params['t'] - params['dt']) // params['dt']) + 1
    params['nb_step'] = 4   
    params['show_progress'] = True
    params['log_progress_interval_yr'] = 25_000
    params['save_topography_maps'] = False
    params['topography_snapshot_times_yr'] = None
    params['random_seed'] = 0
    params['stop_on_crater_breach'] = False
    params['crater_radius_m'] = 15000.0
    params['crater_mask_radius_m'] = 15000.0
    params['event_check_interval_yr'] = 5000
    params['crater_event_min_area_m2'] = 1.0e6

    
    
    

    params['save_figures'] = True
    params['show_figures'] = False
    params['save_dpi'] = 300
    params['use_pyvista_3d'] = False   
    params['run_gravity'] = False
    params['gravity_method'] = "all"   
    params['gravity_kernel'] = "numba"       
    params['gravity_spc'] = 1
    params['gravity_terrain_spc'] = 40
    params['gravity_terrain_density'] = 2670.0
    params['gravity_density_mode'] = "decreasing"  
    params['gravity_constant_delta_rho'] = -170.0
    params['gravity_compaction_lithology'] = "shaly_sand"  
    params['gravity_host_density'] = 2670.0
    params['gravity_grain_density'] = 2650.0
    params['gravity_fluid_density'] = 1000.0
    params['gravity_compaction_layer_thickness'] = 50.0  
    params['gravity_add_initial_background'] = True
    
    
    
    
    
    params['Sc'] = 0.8        
    params['n_terms'] = 2     
    
    
    
    
    
    params['K_sed'] = 0.0001   
    params['K_bed'] = 0.00001  
    
    
    
    
    
    params['soil_prod_max'] = 0.3 / 10000  
    params['soil_prod_decay'] = 1.0         
    
    
    
    
    
    params['soil_transport_vel'] = 0.02    
    params['soil_transport_decay'] = 1.0   

    
    
    

    params['uplift_rate_m_per_Ma'] = 0    
    params['uplift_rate'] = params['uplift_rate_m_per_Ma'] * 1e-6  

    
    
    
    
    params['g'] = 9.81           
    params['E'] = 25e9           
    params['Te'] = 30e3          
    params['nu'] = 0.25          
    params['rhom'] = 3300        
    params['rhoc'] = 2800        
    params['rho_s'] = 2480       
    params['rho_b'] = 2680       
    params['flexure_include_direct_uplift_load'] = True
    
    params['dt_flex'] = 10000    
    params['nb_step_flex'] = max(1, int(round(params['t'] / params['dt_flex'])))
    params['mod_flex'] = max(1, int(round(params['dt_flex'] / params['dt'])))

    
    
    

    params['figures_root'] = build_run_output_dir(params)

    return params


def format_value_for_path(value, decimals=3):
    """MGM internal routine."""
    rounded = round(float(value), decimals)
    if abs(rounded - round(rounded)) < 10 ** (-decimals):
        return str(int(round(rounded)))
    return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "neg")


def build_run_output_dir(params):
    """MGM internal routine."""
    te_label = format_value_for_path(params["Te"] / 1e3)
    uplift_label = format_value_for_path(params["uplift_rate_m_per_Ma"])
    xy_label = format_value_for_path(params["xy_space"])
    return (
        f"{params['output_root']}/Te{te_label}km"
        f"_uplift{uplift_label}mMa"
        f"_xy{xy_label}m"
        f"_t{int(params['t'])}"
    )


def classify_crater_sector(rel_x_m, rel_y_m):
    """MGM internal routine."""
    if abs(rel_y_m) >= abs(rel_x_m):
        return "north" if rel_y_m >= 0.0 else "south"
    return "east" if rel_x_m >= 0.0 else "west"


def extract_crater_crossings(barringer, crater_mask, min_area_m2=0.0):
    """MGM internal routine."""
    mask_flat = crater_mask.reshape(-1)
    receiver = barringer.at_node["flow__receiver_node"].astype(int)
    drainage_area = barringer.at_node["drainage_area"]
    node_ids = np.arange(barringer.number_of_nodes, dtype=int)
    has_receiver = receiver != node_ids

    outgoing = mask_flat & (~mask_flat[receiver]) & has_receiver & (drainage_area >= min_area_m2)
    incoming = (~mask_flat) & mask_flat[receiver] & has_receiver & (drainage_area >= min_area_m2)

    center_x_m = 0.5 * (float(np.min(barringer.node_x)) + float(np.max(barringer.node_x)))
    center_y_m = 0.5 * (float(np.min(barringer.node_y)) + float(np.max(barringer.node_y)))

    def build_records(mask, kind):
        node_list = np.where(mask)[0]
        records = []
        for node in node_list:
            rec = int(receiver[node])
            crossing_x_m = 0.5 * (float(barringer.node_x[node]) + float(barringer.node_x[rec]))
            crossing_y_m = 0.5 * (float(barringer.node_y[node]) + float(barringer.node_y[rec]))
            rel_x_m = crossing_x_m - center_x_m
            rel_y_m = crossing_y_m - center_y_m
            records.append(
                {
                    "kind": kind,
                    "node": int(node),
                    "receiver_node": rec,
                    "drainage_area_m2": float(drainage_area[node]),
                    "node_x_m": float(barringer.node_x[node]),
                    "node_y_m": float(barringer.node_y[node]),
                    "receiver_x_m": float(barringer.node_x[rec]),
                    "receiver_y_m": float(barringer.node_y[rec]),
                    "crossing_x_m": crossing_x_m,
                    "crossing_y_m": crossing_y_m,
                    "rel_x_m": rel_x_m,
                    "rel_y_m": rel_y_m,
                    "sector": classify_crater_sector(rel_x_m, rel_y_m),
                }
            )
        return records

    return {
        "center_x_m": center_x_m,
        "center_y_m": center_y_m,
        "outgoing": build_records(outgoing, "outgoing"),
        "incoming": build_records(incoming, "incoming"),
    }


def compute_drainage_summary(barringer, params):
    """MGM internal routine."""
    crater_mask = creer_masque_circulaire(
        nx=barringer.shape[1],
        ny=barringer.shape[0],
        dx=params["xy_space"],
        dy=params["xy_space"],
        rayon_m=params.get("crater_mask_radius_m", 15000.0),
    )
    crossing_info = extract_crater_crossings(
        barringer,
        crater_mask=crater_mask,
        min_area_m2=float(params.get("crater_event_min_area_m2", 0.0)),
    )
    outgoing = crossing_info["outgoing"]

    sector_metrics = {
        side: {
            "sum_area_m2": 0.0,
            "max_area_m2": 0.0,
            "outlet_node": -1,
            "outlet_x_m": np.nan,
            "outlet_y_m": np.nan,
        }
        for side in ("north", "south", "east", "west")
    }

    for record in outgoing:
        sector = record["sector"]
        stats = sector_metrics[sector]
        stats["sum_area_m2"] += float(record["drainage_area_m2"])
        if float(record["drainage_area_m2"]) > stats["max_area_m2"]:
            stats["max_area_m2"] = float(record["drainage_area_m2"])
            stats["outlet_node"] = int(record["node"])
            stats["outlet_x_m"] = float(record["crossing_x_m"])
            stats["outlet_y_m"] = float(record["crossing_y_m"])

    north_max = sector_metrics["north"]["max_area_m2"]
    south_max = sector_metrics["south"]["max_area_m2"]
    ns_denom = north_max + south_max
    north_south_score = 0.0 if ns_denom == 0.0 else (north_max - south_max) / ns_denom

    if outgoing:
        dominant_record = max(outgoing, key=lambda item: item["drainage_area_m2"])
        dominant_side = dominant_record["sector"]
        dominant_outlet_node = int(dominant_record["node"])
        dominant_outlet_x_m = float(dominant_record["crossing_x_m"])
        dominant_outlet_y_m = float(dominant_record["crossing_y_m"])
    else:
        dominant_side = "none"
        dominant_outlet_node = -1
        dominant_outlet_x_m = np.nan
        dominant_outlet_y_m = np.nan

    if outgoing and north_south_score > 0:
        north_south_label = "north"
    elif outgoing and north_south_score < 0:
        north_south_label = "south"
    elif outgoing:
        north_south_label = "balanced"
    else:
        north_south_label = "no_breach"

    return {
        "Te_m": float(params["Te"]),
        "Te_km": float(params["Te"] / 1e3),
        "uplift_m_per_Ma": float(params["uplift_rate_m_per_Ma"]),
        "simulation_time_yr": int(params["t"]),
        "xy_space_m": float(params["xy_space"]),
        "surface_profile_path": str(params.get("file_path") or ""),
        "initial_seed_path": str(params.get("initial_seed_path") or ""),
        "domain_boundaries": str(params.get("boundary_mode", "all_open")),
        "random_seed": None if params.get("random_seed") is None else int(params["random_seed"]),
        "crater_mask_radius_m": float(params.get("crater_mask_radius_m", 0.0)),
        "north_south_score": float(north_south_score),
        "north_south_label": north_south_label,
        "north_south_reference": "crater_rim_crossing",
        "dominant_outlet_side": dominant_side,
        "dominant_outlet_node": dominant_outlet_node,
        "dominant_outlet_x_m": dominant_outlet_x_m,
        "dominant_outlet_y_m": dominant_outlet_y_m,
        "edge_metrics": sector_metrics,
        "crater_center_x_m": float(crossing_info["center_x_m"]),
        "crater_center_y_m": float(crossing_info["center_y_m"]),
        "crater_outgoing_crossings_count": int(len(outgoing)),
    }


def detect_crater_drainage_event(barringer, crater_mask, min_area_m2):
    """MGM internal routine."""
    crossing_info = extract_crater_crossings(
        barringer,
        crater_mask=crater_mask,
        min_area_m2=min_area_m2,
    )
    candidates = crossing_info["outgoing"] + crossing_info["incoming"]
    if not candidates:
        return {"detected": False}

    record = max(candidates, key=lambda item: item["drainage_area_m2"])
    event = {
        "detected": True,
        "kind": record["kind"],
        "node": int(record["node"]),
        "receiver_node": int(record["receiver_node"]),
        "drainage_area_m2": float(record["drainage_area_m2"]),
        "node_x_m": float(record["node_x_m"]),
        "node_y_m": float(record["node_y_m"]),
        "receiver_x_m": float(record["receiver_x_m"]),
        "receiver_y_m": float(record["receiver_y_m"]),
        "crossing_x_m": float(record["crossing_x_m"]),
        "crossing_y_m": float(record["crossing_y_m"]),
        "sector": record["sector"],
    }
    return event


def resolve_topography_snapshot_times_yr(params):
    """MGM internal routine."""
    requested_times_yr = params.get("topography_snapshot_times_yr")
    total_time_yr = float(params.get("t", 0))
    dt_yr = float(params.get("dt", 1))

    if dt_yr <= 0.0:
        raise ValueError("dt doit être strictement positif pour définir les snapshots.")

    if requested_times_yr is None:
        n_snapshots = max(int(params.get("nb_step", 1)), 1)
        if n_snapshots == 1:
            target_times_yr = np.array([0.0], dtype=float)
        else:
            target_times_yr = np.linspace(0.0, total_time_yr, n_snapshots)
    else:
        target_times_yr = np.asarray(requested_times_yr, dtype=float).reshape(-1)
        if target_times_yr.size == 0:
            raise ValueError("topography_snapshot_times_yr ne peut pas être vide.")

    if np.any(~np.isfinite(target_times_yr)):
        raise ValueError("Les temps de snapshots topo doivent être des nombres finis.")
    if np.any(target_times_yr < -1.0e-9) or np.any(target_times_yr > total_time_yr + 1.0e-9):
        raise ValueError("Les temps de snapshots topo doivent rester dans l'intervalle [0, t].")

    snapped_times_yr = np.rint(target_times_yr / dt_yr) * dt_yr
    snapped_times_yr = np.clip(snapped_times_yr, 0.0, total_time_yr)
    snapped_times_yr = snapped_times_yr.astype(int)
    unique_times_yr = np.unique(snapped_times_yr)

    if requested_times_yr is not None and unique_times_yr.size != target_times_yr.size:
        raise ValueError(
            "Les temps demandés pour les snapshots topo se recouvrent après projection sur le pas de temps."
        )

    return unique_times_yr


def save_drainage_summary(summary, figures_root):
    """MGM internal routine."""
    os.makedirs(figures_root, exist_ok=True)
    summary_path = os.path.join(figures_root, "drainage_summary.json")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=True)
    return summary_path


def load_initial_seed(seed_path):
    """MGM internal routine."""
    with np.load(seed_path, allow_pickle=True) as topo_seed:
        return {
            "z_final": topo_seed["z_final"].copy(),
            "bedrock_final": topo_seed["bedrock_final"].copy() if "bedrock_final" in topo_seed.files else None,
            "soil_depth_final": topo_seed["soil_depth_final"].copy() if "soil_depth_final" in topo_seed.files else None,
            "X": topo_seed["X"].copy() if "X" in topo_seed.files else None,
            "Y": topo_seed["Y"].copy() if "Y" in topo_seed.files else None,
            "xy_space": float(topo_seed["xy_space"]) if "xy_space" in topo_seed.files else None,
            "limit": float(topo_seed["limit"]) if "limit" in topo_seed.files else None,
        }



def apply_boundary_conditions(barringer, params):
    """MGM internal routine."""
    boundary_mode = str(params.get("boundary_mode", "all_open")).lower()

    if boundary_mode == "all_open":
        barringer.set_closed_boundaries_at_grid_edges(
            right_is_closed=False,
            top_is_closed=False,
            left_is_closed=False,
            bottom_is_closed=False,
        )
    elif boundary_mode == "south_open":
        barringer.set_closed_boundaries_at_grid_edges(
            right_is_closed=True,
            top_is_closed=True,
            left_is_closed=True,
            bottom_is_closed=False,
        )
    elif boundary_mode in {"southwest_corner", "southeast_corner"}:
        barringer.set_closed_boundaries_at_grid_edges(
            right_is_closed=True,
            top_is_closed=True,
            left_is_closed=True,
            bottom_is_closed=True,
        )
        if boundary_mode == "southwest_corner":
            outlet_id = int(barringer.nodes_at_bottom_edge[0])
        else:
            outlet_id = int(barringer.nodes_at_bottom_edge[-1])
        barringer.set_watershed_boundary_condition_outlet_id(
            outlet_id,
            barringer.at_node["topographic__elevation"],
            -9999.0,
        )
    else:
        raise ValueError(
            "boundary_mode doit valoir 'all_open', 'south_open', "
            "'southwest_corner' ou 'southeast_corner'."
        )

    return boundary_mode


def initialize_grid(z, xy_space, params):
    """MGM internal routine."""
    nby, nbx = z.shape
    
    
    
    
    
    barringer = RasterModelGrid((nby, nbx), xy_spacing=(xy_space, xy_space))
    
    seed_bedrock = params.get("initial_bedrock_seed")
    seed_soil = params.get("initial_soil_seed")
    use_seed_layers = (seed_bedrock is not None) and (seed_soil is not None)

    if use_seed_layers:
        if seed_bedrock.shape != z.shape or seed_soil.shape != z.shape:
            raise ValueError("Le seed bedrock/soil est incompatible avec la grille demandée.")
    else:
        
        rng = np.random.default_rng(params.get("random_seed"))
        bruit = np.reshape(
            rng.random(barringer.number_of_nodes),
            barringer.shape
        )
        z += bruit
    
    
    barringer.add_field(
        "topographic__elevation",
        z,
        at="node",
        clobber=True
    )
    
    apply_boundary_conditions(barringer, params)
    
    
    
    
    z_bedrock = np.zeros(z.shape)
    
    
    barringer.add_zeros("node", "soil__depth")
    
    
    barringer.add_field(
        "bedrock__elevation",
        z_bedrock,
        at="node",
        clobber=True
    )
    
    if use_seed_layers:
        barringer.at_node["bedrock__elevation"][:] = seed_bedrock.reshape(-1)
        barringer.at_node["soil__depth"][:] = seed_soil.reshape(-1)
        barringer.at_node["topographic__elevation"][:] = (seed_bedrock + seed_soil).reshape(-1)
    else:
        barringer.at_node["soil__depth"][barringer.core_nodes] = 1.0  
        barringer.at_node["bedrock__elevation"][:] = \
            barringer.at_node["topographic__elevation"]
        barringer.at_node["topographic__elevation"][:] += \
            barringer.at_node["soil__depth"]
    
    
    
    
    
    components = {}
    
    
    components['frr'] = FlowAccumulator(
        barringer,
        flow_director="FlowDirectorD8"
    )
    
    
    components['space_large'] = SpaceLargeScaleEroder(
        barringer,
        K_sed=params['K_sed'],
        K_br=params['K_bed'],
        F_f=0.0,
        phi=0.0,
        H_star=1.0,
        v_s=5.0,
        m_sp=0.5,
        n_sp=1.0,
        sp_crit_sed=0,
        sp_crit_br=0,
    )
    
    
    components['expweath'] = ExponentialWeatherer(
        barringer,
        soil_production_maximum_rate=params['soil_prod_max'],
        soil_production_decay_depth=params['soil_prod_decay']
    )
    
    
    components['ddtd'] = DepthDependentTaylorDiffuser(
        barringer,
        soil_transport_velocity=params['soil_transport_vel'],
        soil_transport_decay_depth=params['soil_transport_decay'],
        slope_crit=params['Sc'],
        dynamic_dt=True,
        nterms=params['n_terms']
    )
    
    return barringer, components






def run_simulation(barringer, components, z, X, Y, params):
    """MGM internal routine."""
    nby, nbx = z.shape
    rows, colums = z.shape
    
    
    
    
    
    z_init = copy.copy(z)
    
    
    requested_snapshot_times_yr = resolve_topography_snapshot_times_yr(params)
    pending_snapshot_times_yr = requested_snapshot_times_yr.tolist()
    TOPO = []
    EPAIS = []
    FLEX = []
    topo_snapshot_times_yr = []
    cumulative_flexure = np.zeros_like(z_init)

    if pending_snapshot_times_yr and pending_snapshot_times_yr[0] == 0:
        TOPO.append(copy.copy(z_init))
        EPAIS.append(copy.copy(barringer.at_node["soil__depth"]))
        FLEX.append(np.zeros_like(z_init))
        topo_snapshot_times_yr.append(0)
        pending_snapshot_times_yr.pop(0)
    
    
    topo = []
    epais = []
    t_simu = []
    
    
    mid_y_index = nby // 2
    crater_diameters = []
    
    
    qs = np.zeros_like(z)
    rho = np.ones_like(z) * params['rho_s']
    
    
    z_bed_previous = copy.copy(
        barringer.at_node["bedrock__elevation"]
    ).reshape(z.shape)
    z_soil_previous = copy.copy(
        barringer.at_node["soil__depth"]
    ).reshape(z.shape)
    
    
    log_progress_interval_yr = max(
        params["dt"],
        int(params.get("log_progress_interval_yr", params["dt"])),
    )
    log_progress_steps = max(1, int(round(log_progress_interval_yr / params["dt"])))

    stop_on_crater_breach = params.get("stop_on_crater_breach", False)
    crater_mask = None
    event_check_steps = 1
    crater_event = {"detected": False}
    executed_time_yr = int(params["t"])
    stop_reason = "completed"
    stopped_early = False
    if stop_on_crater_breach:
        crater_mask = creer_masque_circulaire(
            nx=nbx,
            ny=nby,
            dx=params["xy_space"],
            dy=params["xy_space"],
            rayon_m=params.get("crater_mask_radius_m", 15000.0),
        )
        event_check_steps = max(1, int(round(params.get("event_check_interval_yr", params["dt"]) / params["dt"])))
    
    
    
    
    
    for i in tqdm(
        range(params['nt']),
        desc="Simulation en cours",
        bar_format="{l_bar}{bar:10}{r_bar}",
        unit=" iter",
        leave=True,
        disable=not params.get("show_progress", True),
    ):
        current_time_yr = int(min((i + 1) * params["dt"], params["t"]))

        
        
        
        
        components['frr'].run_one_step()
        components['space_large'].run_one_step(dt=params['dt'])
        components['expweath'].calc_soil_prod_rate()
        components['ddtd'].run_one_step(dt=params['dt'])

        
        
        

        uplift_increment = params['uplift_rate'] * params['dt']  
        barringer.at_node["bedrock__elevation"][barringer.core_nodes] += uplift_increment
        
        barringer.at_node["topographic__elevation"][barringer.core_nodes] += uplift_increment
        
        
        
        
        
        z_bed = copy.copy(
            barringer.at_node["bedrock__elevation"]
        ).reshape(z.shape)
        z_soil = copy.copy(
            barringer.at_node["soil__depth"]
        ).reshape(z.shape)
        
        
        
        
        bedrock_delta = z_bed - z_bed_previous
        soil_delta = z_soil - z_soil_previous
        if not params.get('flexure_include_direct_uplift_load', True):
            bedrock_delta = bedrock_delta.copy()
            bedrock_delta.reshape(-1)[barringer.core_nodes] -= uplift_increment

        diff = (
            params['rho_b'] * bedrock_delta +
            params['rho_s'] * soil_delta
        )

        qs += calc_load(diff, rho, params['g'], 1)
        
        
        
        
        
        if i % params['mod_flex'] == 0:
            
            nx, ny = z.shape
            dx = X[1, 2] - X[2, 1]
            dy = Y[2, 1] - Y[1, 2]
            
            deflec = calc_flex_extended(
                qs, nx, ny, dx, dy,
                params['E'], params['Te'], params['nu'],
                params['rhom'], params['rhoc'], params['g'],
                margin_km=0, crater_radius_km=100
            )
            
            
            applied_deflec = np.zeros_like(deflec)
            applied_deflec.reshape(-1)[barringer.core_nodes] = deflec.reshape(-1)[barringer.core_nodes]
            z.reshape(-1)[barringer.core_nodes] += applied_deflec.reshape(-1)[barringer.core_nodes]

            _zb = barringer.at_node["bedrock__elevation"]
            _zb[barringer.core_nodes] += applied_deflec.reshape(-1)[barringer.core_nodes]
            cumulative_flexure += applied_deflec
            
            
            qs = np.zeros_like(z)
            
            
            t_simu_previous = i * params['dt']
            t_simu.append(copy.copy(t_simu_previous))
            topo.append(copy.copy(z))
            epais.append(copy.copy(
                barringer.at_node["soil__depth"].reshape(z.shape)
            ))
            
            
            
            crater_diameter_value = detect_crater_diameter_hybrid(
                z[mid_y_index, :],
                0,  
                5,  
                1,  
                params['xy_space']
            )
            crater_diameters.append(crater_diameter_value)
        
        
        
        
        
        z_bed_previous = copy.copy(
            barringer.at_node["bedrock__elevation"]
        ).reshape(z.shape)
        z_soil_previous = copy.copy(
            barringer.at_node["soil__depth"]
        ).reshape(z.shape)
        
        
        
        

        if stop_on_crater_breach and i % event_check_steps == 0:
            components['frr'].run_one_step()
            crater_event = detect_crater_drainage_event(
                barringer,
                crater_mask=crater_mask,
                min_area_m2=params.get("crater_event_min_area_m2", 1.0e6),
            )
            if crater_event["detected"]:
                executed_time_yr = current_time_yr
                crater_event["time_yr"] = executed_time_yr
                stop_reason = f"crater_{crater_event['kind']}"
                stopped_early = True
                while pending_snapshot_times_yr and pending_snapshot_times_yr[0] <= current_time_yr:
                    snapshot_time_yr = int(pending_snapshot_times_yr.pop(0))
                    TOPO.append(copy.copy(z))
                    EPAIS.append(copy.copy(barringer.at_node["soil__depth"]))
                    FLEX.append(copy.copy(cumulative_flexure))
                    topo_snapshot_times_yr.append(snapshot_time_yr)
                print(
                    " Événement cratère détecté : "
                    f"{crater_event['kind']} à t={executed_time_yr/1000:.1f} kyr "
                    f"sur le secteur {crater_event.get('sector', 'unknown')} "
                    f"(A={crater_event['drainage_area_m2']/1e6:.2f} km²)"
                )
                break

        
        
        

        progress_due = (i == 0) or ((i % log_progress_steps) == 0) or (i == params["nt"] - 1)
        if progress_due:
            progress_pct = 100.0 * current_time_yr / max(params["t"], 1)
            print(f"[PROGRESS] t = {current_time_yr/1000:.1f} kyr ({progress_pct:.1f}%)")

        while pending_snapshot_times_yr and pending_snapshot_times_yr[0] <= current_time_yr:
            snapshot_time_yr = int(pending_snapshot_times_yr.pop(0))
            TOPO.append(copy.copy(z))
            EPAIS.append(copy.copy(barringer.at_node["soil__depth"]))
            FLEX.append(copy.copy(cumulative_flexure))
            topo_snapshot_times_yr.append(snapshot_time_yr)
    
    
    
    
    
    results = {
        'TOPO': TOPO,
        'EPAIS': EPAIS,
        'FLEX': FLEX,
        'topo': topo,
        'epais': epais,
        't_simu': t_simu,
        'crater_diameters': crater_diameters,
        'z_init': z_init,
        'topography_snapshot_times_yr': topo_snapshot_times_yr,
        'executed_time_yr': executed_time_yr,
        'stop_reason': stop_reason,
        'crater_event': crater_event,
    }
    
    return results






def compute_magnetic_anomalies_1layer(TOPO, X, Y, params):
    """MGM internal routine."""
    TOPO_array = -np.array(TOPO)
    
    
    
    mag_maps, mag_prof, H_map, H_rem_all, top_rem_all, z_base_layer = \
        compute_mag_evolution_column_prism(
            TOPO=TOPO_array,
            X=X,
            Y=Y,
            obs_mode="flat",
            xy_space=params['xy_space'],
            nb_step=params['nb_step'],
            z_comp_down=250.0,
            crater_radius=7500.0
        )

    mag_maps = np.abs(mag_maps)
    mag_prof = np.abs(mag_prof)

    return mag_maps, mag_prof, H_map, H_rem_all, top_rem_all, z_base_layer



def compute_magnetic_anomalies_case1(TOPO, X, Y, params):
    """MGM internal routine."""
    TOPO_array = -np.array(TOPO)  

    mag_maps, mag_prof, H0_map, H_rem_all, top_rem_all, base_rem_all = \
        compute_mag_evolution_column_prism_case1(
            TOPO=TOPO_array,
            X=X,
            Y=Y,
            xy_space=params['xy_space'],
            nb_step=params['nb_step'],
            obs_mode='flat',
            z_obs_up=1000.0,
            z_comp_down=1000.0,
            crater_radius=7500.0,
            base_mode='flat_at_center',
            erosion_mode='linear',
            return_layer=True
        )

    mag_maps = np.abs(mag_maps)
    mag_prof = np.abs(mag_prof)

    return mag_maps, mag_prof, H0_map, H_rem_all, top_rem_all, base_rem_all

def compute_magnetic_anomalies_case3(TOPO, X, Y, params):
    """MGM internal routine."""

    
    TOPO_array = -np.array(TOPO)

    mag_maps, mag_prof, H0_map, top_rem_all, base_rem_all = \
        compute_mag_simple_flat_layer(
            TOPO=TOPO_array,
            X=X,
            Y=Y,
            xy_space=params["xy_space"],
            nb_step=params["nb_step"],
            
            z_top_layer_down=params.get("z_top_layer_down", 2000.0),
            thickness_down=params.get("thickness_down", 500.0),
            crater_radius=params.get("crater_radius", None),
            
            obs_mode="flat",
            z_obs_up=1000.0,
            return_layer=True,
        )

    mag_maps = np.abs(mag_maps)
    mag_prof = np.abs(mag_prof)

    return mag_maps, mag_prof, H0_map, top_rem_all, base_rem_all

def compute_magnetic_anomalies_2layers(TOPO, X, Y, params):
    """MGM internal routine."""
    TOPO_array = -np.array(TOPO)
    
    
    mag_maps, mag_prof, base1, top1_rem_all, base2, top2_rem_all = \
        compute_mag_evolution_column_prism_2layers(
            TOPO_array, X, Y, 
            params['xy_space'], 
            params['nb_step'],
            I_deg=90.0, 
            D_deg=0.0,
            obs_mode="flat", 
            z_obs_up=1000.0,
            
            z_comp_down1=250.0,
            thickness_mode1="taper_to_zero",
            M_center1=0.4, 
            M_background1=0.02,
            
            use_layer2=True,
            z_comp_down2=500.0,
            thickness_mode2="taper_to_zero",
            M_center2=0.6, 
            M_background2=0.01,
            erosion_mode="linear",
        )

    mag_maps = np.abs(mag_maps)
    mag_prof = np.abs(mag_prof)

    return mag_maps, mag_prof, base1, top1_rem_all, base2, top2_rem_all

def compute_magnetic_anomalies_case2(TOPO, X, Y, params, return_components=False):
    """MGM internal routine."""

    TOPO_array = -np.array(TOPO)  

    
    params["crater_radius_case2"] = 15000.0
    params["lens_offset_range_km_case2"] = (2.0, 8.0)  
    params["lens_sigma_km_case2"] = 1.2
    params["lens_thickness_down_case2"] = 200.0
    params["lobe_thickness_down_case2"] = 500.0

    
    lens_offset_range_km = params.get("lens_offset_range_km_case2", None)
    lens_offset_km = params.get("lens_offset_km_case2", None)

    
    lens_sigma_km = params.get("lens_sigma_km_case2", None)

    
    result_case2 = compute_mag_evolution_column_prism_case2_lenses_lobe(
            TOPO=TOPO_array,
            X=X,
            Y=Y,
            xy_space=params["xy_space"],
            nb_step=params["nb_step"],

            
            crater_radius=params.get("crater_radius_case2", 15000.0),

            
            lens_offset_range_km=lens_offset_range_km,
            lens_offset_km=lens_offset_km,
            lens_offset_frac=params.get("lens_offset_frac_case2", params.get("lens_offset_frac", 0.45)),

            
            lens_sigma_km=lens_sigma_km,
            lens_sigma_frac=params.get("lens_sigma_frac_case2", params.get("lens_sigma_frac", 0.18)),
            lens_thickness_down=params.get("lens_thickness_down_case2", params.get("lens_thickness_down", 400.0)),

            
            lobe_thickness_down=params.get("lobe_thickness_down_case2", params.get("lobe_thickness_down", 250.0)),
            lobe_inner_hole_frac=params.get("lobe_inner_hole_frac_case2", params.get("lobe_inner_hole_frac", 0.0)),
            lobe_taper_power=params.get("lobe_taper_power_case2", params.get("lobe_taper_power", 1.0)),

            
            M_center_lens=params.get("M_center_lens_case2", params.get("M_center_lens", 0.6)),
            M_background_lens=params.get("M_background_lens_case2", params.get("M_background_lens", 0.02)),
            r_scale_frac_lens=params.get("r_scale_frac_lens_case2", params.get("r_scale_frac_lens", 0.35)),
            z_scale_lens=params.get("z_scale_lens_case2", params.get("z_scale_lens", 1500.0)),

            
            M_center_lobe=params.get("M_center_lobe_case2", params.get("M_center_lobe", 0.35)),
            M_background_lobe=params.get("M_background_lobe_case2", params.get("M_background_lobe", 0.01)),
            r_scale_frac_lobe=params.get("r_scale_frac_lobe_case2", params.get("r_scale_frac_lobe", 0.50)),
            z_scale_lobe=params.get("z_scale_lobe_case2", params.get("z_scale_lobe", 2000.0)),

            
            I_deg=params.get("I_deg", 90.0),
            D_deg=params.get("D_deg", 0.0),

            
            obs_mode=params.get("obs_mode_mag", "flat"),
            z_obs_up=params.get("z_obs_up", 1000.0),
            h_above_topo_up=params.get("h_above_topo_up", 0.0),

            
            erosion_mode=params.get("erosion_mode_mag", "linear"),

            return_layer=True,
            return_components=return_components,
            force_negative=params.get("force_negative_mag", True),
            use_numba=params.get("use_numba_mag", True),
        )

    
    if return_components:
        (
            mag_maps, mag_prof, H0_map, H_rem_all, top_rem_all, base_rem_all,
            top_lens_rem_all, base_lens_rem_all, top_lobe_rem_all, base_lobe_rem_all
        ) = result_case2
    else:
        mag_maps, mag_prof, H0_map, H_rem_all, top_rem_all, base_rem_all = result_case2

    
    mag_maps = np.abs(mag_maps)
    mag_prof = np.abs(mag_prof)

    if return_components:
        return (
            mag_maps, mag_prof, H0_map, H_rem_all, top_rem_all, base_rem_all,
            top_lens_rem_all, base_lens_rem_all, top_lobe_rem_all, base_lobe_rem_all
        )
    return mag_maps, mag_prof, H0_map, H_rem_all, top_rem_all, base_rem_all


def compute_snapshot_times_Ma(n_snapshots, params):
    """MGM internal routine."""
    snapshot_times_yr = params.get("snapshot_times_yr")
    if snapshot_times_yr is not None:
        snapshot_times_yr = np.asarray(snapshot_times_yr, dtype=float).reshape(-1)
        if snapshot_times_yr.size == n_snapshots:
            return snapshot_times_yr / 1.0e6
    executed_time_yr = float(params.get("executed_time_yr", params.get("t", 0)))
    if n_snapshots <= 1:
        return np.array([0.0], dtype=float)
    return np.linspace(0.0, executed_time_yr / 1.0e6, n_snapshots)


def _extract_center_west_distance_km(X):
    """MGM internal routine."""
    rows, colums = X.shape
    mid_y = rows // 2
    mid_x = colums // 2
    x_line = np.asarray(X[mid_y, :], dtype=float)
    return (x_line[mid_x] - x_line[:mid_x + 1][::-1]) / 1000.0


def _extract_center_west_profiles(stack_2d, X):
    """MGM internal routine."""
    stack_2d = np.asarray(stack_2d, dtype=float)
    if stack_2d.ndim != 3:
        raise ValueError("stack_2d doit être un array 3D (n_snapshots, nby, nbx).")

    rows, colums = stack_2d[0].shape
    mid_y = rows // 2
    mid_x = colums // 2
    profiles = np.asarray(
        [snapshot[mid_y, :mid_x + 1][::-1] for snapshot in stack_2d],
        dtype=float,
    )
    return _extract_center_west_distance_km(X), profiles, mid_y, mid_x


def _gravity_map_extrema(gravi_maps):
    """MGM internal routine."""
    gravi_maps = np.asarray(gravi_maps, dtype=float)
    return (
        np.nanmin(gravi_maps, axis=(1, 2)),
        np.nanmax(gravi_maps, axis=(1, 2)),
        np.nanmax(np.abs(gravi_maps), axis=(1, 2)),
    )


def export_gravity_field_data(
    TOPO,
    gravi_maps,
    X,
    Y,
    params,
    field_name,
    title,
    backend=None,
    save_path=None,
):
    """MGM internal routine."""
    if save_path is None:
        return

    topo_stack = np.asarray(TOPO, dtype=float)
    gravi_maps = np.asarray(gravi_maps, dtype=float)
    if topo_stack.shape != gravi_maps.shape:
        raise ValueError("TOPO et gravi_maps doivent partager la même forme pour l'export.")

    distance_km, topo_profiles, mid_y, mid_x = _extract_center_west_profiles(topo_stack, X)
    _, gravity_profiles, _, _ = _extract_center_west_profiles(gravi_maps, X)
    snapshot_times_Ma = compute_snapshot_times_Ma(gravi_maps.shape[0], params)
    map_min_mgal, map_max_mgal, map_absmax_mgal = _gravity_map_extrema(gravi_maps)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        field_name=np.asarray(str(field_name)),
        title=np.asarray(str(title)),
        backend=np.asarray("" if backend is None else str(backend)),
        snapshot_times_Ma=np.asarray(snapshot_times_Ma, dtype=float),
        x_coords_m=np.asarray(X[0, :], dtype=float),
        y_coords_m=np.asarray(Y[:, 0], dtype=float),
        center_west_distance_km=np.asarray(distance_km, dtype=float),
        center_west_topography_m=np.asarray(topo_profiles, dtype=float),
        gravity_profiles_center_west_mgal=np.asarray(gravity_profiles, dtype=float),
        maps_mgal=np.asarray(gravi_maps, dtype=float),
        map_min_mgal=np.asarray(map_min_mgal, dtype=float),
        map_max_mgal=np.asarray(map_max_mgal, dtype=float),
        map_absmax_mgal=np.asarray(map_absmax_mgal, dtype=float),
        center_row_index=np.asarray(int(mid_y)),
        center_col_index=np.asarray(int(mid_x)),
    )


def export_gravity_comparison_data(
    TOPO,
    first_maps,
    second_maps,
    X,
    Y,
    params,
    comparison_name,
    first_label="Legacy",
    second_label="Reference",
    save_path=None,
):
    """MGM internal routine."""
    if save_path is None:
        return

    topo_stack = np.asarray(TOPO, dtype=float)
    first_maps = np.asarray(first_maps, dtype=float)
    second_maps = np.asarray(second_maps, dtype=float)
    diff_maps = first_maps - second_maps

    if topo_stack.shape != first_maps.shape or first_maps.shape != second_maps.shape:
        raise ValueError("TOPO, first_maps et second_maps doivent partager la même forme pour l'export.")

    distance_km, topo_profiles, mid_y, mid_x = _extract_center_west_profiles(topo_stack, X)
    _, first_profiles, _, _ = _extract_center_west_profiles(first_maps, X)
    _, second_profiles, _, _ = _extract_center_west_profiles(second_maps, X)
    _, diff_profiles, _, _ = _extract_center_west_profiles(diff_maps, X)
    snapshot_times_Ma = compute_snapshot_times_Ma(first_maps.shape[0], params)
    diff_rms_mgal = np.sqrt(np.mean(diff_maps ** 2, axis=(1, 2)))
    diff_max_abs_mgal = np.max(np.abs(diff_maps), axis=(1, 2))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        comparison_name=np.asarray(str(comparison_name)),
        first_label=np.asarray(str(first_label)),
        second_label=np.asarray(str(second_label)),
        snapshot_times_Ma=np.asarray(snapshot_times_Ma, dtype=float),
        x_coords_m=np.asarray(X[0, :], dtype=float),
        y_coords_m=np.asarray(Y[:, 0], dtype=float),
        center_west_distance_km=np.asarray(distance_km, dtype=float),
        center_west_topography_m=np.asarray(topo_profiles, dtype=float),
        first_profiles_center_west_mgal=np.asarray(first_profiles, dtype=float),
        second_profiles_center_west_mgal=np.asarray(second_profiles, dtype=float),
        diff_profiles_center_west_mgal=np.asarray(diff_profiles, dtype=float),
        first_maps_mgal=np.asarray(first_maps, dtype=float),
        second_maps_mgal=np.asarray(second_maps, dtype=float),
        diff_maps_mgal=np.asarray(diff_maps, dtype=float),
        diff_rms_mgal=np.asarray(diff_rms_mgal, dtype=float),
        diff_max_abs_mgal=np.asarray(diff_max_abs_mgal, dtype=float),
        center_row_index=np.asarray(int(mid_y)),
        center_col_index=np.asarray(int(mid_x)),
    )






def plot_topography_maps(TOPO, X, Y, params, title="Topographies",
                         save_path=None, show=True, dpi=300):
    """MGM internal routine."""
    if len(TOPO) == 0:
        raise ValueError("TOPO est vide, impossible de tracer maps_topography.")

    snapshot_times_Ma = compute_snapshot_times_Ma(len(TOPO), params)
    if len(TOPO) <= 4:
        idx_times = np.arange(len(TOPO), dtype=int)
    else:
        n_panels = 4
        idx_times = np.linspace(0, len(TOPO) - 1, n_panels, dtype=int)
        idx_times = np.unique(idx_times)

    ncols = 2
    nrows = int(np.ceil(len(idx_times) / ncols))
    fig = plt.figure(figsize=(14, 6 * nrows))
    gs = gridspec.GridSpec(
        nrows, 3,
        figure=fig,
        width_ratios=[1, 1, 0.05],
        hspace=0.25,
        wspace=0.12,
    )

    axes = []
    for row in range(nrows):
        for col in range(ncols):
            axes.append(fig.add_subplot(gs[row, col]))
    cax = fig.add_subplot(gs[:, 2])

    ext = [X.min() / 1000.0, X.max() / 1000.0, Y.min() / 1000.0, Y.max() / 1000.0]
    displayed_topographies = [TOPO[idx] for idx in idx_times]
    all_vals = np.concatenate([snap.ravel() for snap in displayed_topographies])
    norm = Normalize(vmin=np.nanmin(all_vals), vmax=np.nanmax(all_vals))
    cmap = plt.get_cmap("terrain")
    ls = LightSource(azdeg=315, altdeg=45)

    im_topo = None
    for ax, idx in zip(axes, idx_times):
        topo = TOPO[idx]
        shaded = ls.shade(topo, cmap=plt.get_cmap("gray"), blend_mode="overlay")
        im_topo = ax.imshow(topo, extent=ext, cmap=cmap, norm=norm, origin="lower")
        ax.imshow(shaded, extent=ext, alpha=0.35, origin="lower")
        ax.set_title(f"t = {snapshot_times_Ma[idx]:.3f} Ma")
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.set_aspect("equal")

    for ax in axes[len(idx_times):]:
        ax.axis("off")

    cbar = fig.colorbar(im_topo, cax=cax, aspect=20)
    cbar.set_label("Altitude [m]")
    fig.suptitle(title, fontsize=18, y=0.98)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_flexure_maps(FLEX, X, Y, params, title="Déflexion flexurale cumulée",
                      save_path=None, show=True, dpi=300):
    """MGM internal routine."""
    if len(FLEX) == 0:
        raise ValueError("FLEX est vide, impossible de tracer maps_flexure.")

    n_panels = min(4, len(FLEX))
    idx_times = np.linspace(0, len(FLEX) - 1, n_panels, dtype=int)
    idx_times = np.unique(idx_times)

    ncols = 2
    nrows = int(np.ceil(len(idx_times) / ncols))
    fig = plt.figure(figsize=(14, 6 * nrows))
    gs = gridspec.GridSpec(
        nrows, 3,
        figure=fig,
        width_ratios=[1, 1, 0.05],
        hspace=0.25,
        wspace=0.12,
    )

    axes = []
    for row in range(nrows):
        for col in range(ncols):
            axes.append(fig.add_subplot(gs[row, col]))
    cax = fig.add_subplot(gs[:, 2])

    ext = [X.min() / 1000.0, X.max() / 1000.0, Y.min() / 1000.0, Y.max() / 1000.0]
    all_vals = np.concatenate([snap.ravel() for snap in FLEX])
    vmax = float(np.nanmax(np.abs(all_vals)))
    if (not np.isfinite(vmax)) or vmax == 0.0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    executed_time_yr = float(params.get("executed_time_yr", params.get("t", 0)))
    snapshot_times_Ma = np.linspace(0.0, executed_time_yr / 1.0e6, len(FLEX))

    im_flex = None
    for ax, idx in zip(axes, idx_times):
        flex = FLEX[idx]
        im_flex = ax.imshow(flex, extent=ext, cmap=cmap, norm=norm, origin="lower")
        ax.set_title(f"t = {snapshot_times_Ma[idx]:.3f} Ma")
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.set_aspect("equal")

    for ax in axes[len(idx_times):]:
        ax.axis("off")

    cbar = fig.colorbar(im_flex, cax=cax, aspect=20)
    cbar.set_label("Déflexion cumulée [m]")
    fig.suptitle(title, fontsize=18, y=0.98)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_gravity_maps(gravi_maps, X, Y, params, title="Anomalies gravimetriques",
                      colorbar_label=r"$\Delta g$ [mGal]",
                      save_path=None, show=True, dpi=300):
    """MGM internal routine."""
    gravi_maps = np.asarray(gravi_maps, dtype=float)
    if gravi_maps.ndim != 3:
        raise ValueError("gravi_maps doit être un array 3D (n_snapshots, nby, nbx).")

    n_snapshots = gravi_maps.shape[0]
    idx_times = np.linspace(0, n_snapshots - 1, min(4, n_snapshots), dtype=int)
    idx_times = np.unique(idx_times)

    ncols = 2
    nrows = int(np.ceil(len(idx_times) / ncols))
    fig = plt.figure(figsize=(14, 6 * nrows))
    gs = gridspec.GridSpec(
        nrows, 3,
        figure=fig,
        width_ratios=[1, 1, 0.05],
        hspace=0.25,
        wspace=0.12,
    )

    axes = []
    for row in range(nrows):
        for col in range(ncols):
            axes.append(fig.add_subplot(gs[row, col]))
    cax = fig.add_subplot(gs[:, 2])

    ext = [X.min() / 1000.0, X.max() / 1000.0, Y.min() / 1000.0, Y.max() / 1000.0]
    vmax = float(np.nanmax(np.abs(gravi_maps)))
    if (not np.isfinite(vmax)) or vmax == 0.0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("seismic")
    snapshot_times_Ma = compute_snapshot_times_Ma(n_snapshots, params)

    im_gravi = None
    for ax, idx in zip(axes, idx_times):
        im_gravi = ax.imshow(gravi_maps[idx], extent=ext, cmap=cmap, norm=norm, origin="lower")
        ax.set_title(f"t = {snapshot_times_Ma[idx]:.3f} Ma")
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.set_aspect("equal")

    for ax in axes[len(idx_times):]:
        ax.axis("off")

    cbar = fig.colorbar(im_gravi, cax=cax, aspect=20)
    cbar.set_label(colorbar_label)
    fig.suptitle(title, fontsize=18, y=0.98)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_gravity_profiles_center_west(TOPO, gravi_maps, X, params,
                                      title="Profils gravimetriques",
                                      save_path=None, show=True, dpi=300):
    """MGM internal routine."""
    topo_stack = np.asarray(TOPO, dtype=float)
    gravi_maps = np.asarray(gravi_maps, dtype=float)
    if topo_stack.shape != gravi_maps.shape:
        raise ValueError("TOPO et gravi_maps doivent partager la même forme.")

    rows, colums = topo_stack[0].shape
    mid_y = rows // 2
    mid_x = colums // 2
    dist_west_km = _extract_center_west_distance_km(X)
    snapshot_times_Ma = compute_snapshot_times_Ma(gravi_maps.shape[0], params)

    fig, (ax_topo, ax_gravi) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.plasma
    norm = Normalize(vmin=0, vmax=max(gravi_maps.shape[0] - 1, 1))

    for istep in range(gravi_maps.shape[0]):
        color = cmap(norm(istep))
        topo_prof = topo_stack[istep][mid_y, :mid_x + 1][::-1]
        gravi_prof = gravi_maps[istep][mid_y, :mid_x + 1][::-1]

        ax_topo.plot(dist_west_km, topo_prof, color=color, label=f"t = {snapshot_times_Ma[istep]:.2f} Ma")
        ax_gravi.plot(dist_west_km, gravi_prof, color=color, label=f"t = {snapshot_times_Ma[istep]:.2f} Ma")

    ax_topo.set_title("Profil topographique (centre -> Ouest)")
    ax_topo.set_xlabel("Distance [km]")
    ax_topo.set_ylabel("Altitude [m]")
    ax_topo.grid(True, alpha=0.25)
    ax_topo.legend(fontsize=10)

    ax_gravi.set_title(title)
    ax_gravi.set_xlabel("Distance [km]")
    ax_gravi.set_ylabel(r"$\Delta g$ [mGal]")
    ax_gravi.grid(True, alpha=0.25)
    ax_gravi.legend(fontsize=10)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_gravity_profiles_comparison_center_west(
    TOPO,
    first_maps,
    second_maps,
    X,
    params,
    first_label="Legacy",
    second_label="Reference",
    save_path=None,
    show=True,
    dpi=300,
):
    """MGM internal routine."""
    topo_stack = np.asarray(TOPO, dtype=float)
    first_maps = np.asarray(first_maps, dtype=float)
    second_maps = np.asarray(second_maps, dtype=float)
    diff_maps = first_maps - second_maps

    rows, colums = topo_stack[0].shape
    mid_y = rows // 2
    mid_x = colums // 2
    dist_west_km = _extract_center_west_distance_km(X)
    snapshot_times_Ma = compute_snapshot_times_Ma(first_maps.shape[0], params)
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, first_maps.shape[0]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    ax_topo, ax_gravi, ax_diff = axes

    for idx, color in enumerate(colors):
        topo_prof = topo_stack[idx][mid_y, :mid_x + 1][::-1]
        first_prof = first_maps[idx][mid_y, :mid_x + 1][::-1]
        second_prof = second_maps[idx][mid_y, :mid_x + 1][::-1]
        diff_prof = diff_maps[idx][mid_y, :mid_x + 1][::-1]
        time_label = f"{snapshot_times_Ma[idx]:.2f} Ma"

        ax_topo.plot(dist_west_km, topo_prof, color=color, lw=2.0, label=time_label)
        ax_gravi.plot(dist_west_km, first_prof, color=color, lw=2.0, ls="-")
        ax_gravi.plot(dist_west_km, second_prof, color=color, lw=2.0, ls="--")
        ax_diff.plot(dist_west_km, diff_prof, color=color, lw=2.0)

    ax_topo.set_title("Topographie")
    ax_topo.set_xlabel("Distance [km]")
    ax_topo.set_ylabel("Altitude [m]")
    ax_topo.grid(True, alpha=0.25)
    ax_topo.legend(title="Snapshots", fontsize=9)

    ax_gravi.set_title("Gravite")
    ax_gravi.set_xlabel("Distance [km]")
    ax_gravi.set_ylabel(r"$\Delta g$ [mGal]")
    ax_gravi.grid(True, alpha=0.25)
    ax_gravi.legend(
        handles=[
            Line2D([0], [0], color="black", lw=2.0, ls="-", label=first_label),
            Line2D([0], [0], color="black", lw=2.0, ls="--", label=second_label),
        ],
        loc="best",
    )

    ax_diff.set_title(f"{first_label} - {second_label}")
    ax_diff.set_xlabel("Distance [km]")
    ax_diff.set_ylabel("Difference [mGal]")
    ax_diff.grid(True, alpha=0.25)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_gravity_comparison_summary(
    first_maps,
    second_maps,
    params,
    save_path,
    first_label="legacy",
    second_label="reference",
):
    """MGM internal routine."""
    first_maps = np.asarray(first_maps, dtype=float)
    second_maps = np.asarray(second_maps, dtype=float)
    diff_maps = first_maps - second_maps
    snapshot_times_Ma = compute_snapshot_times_Ma(first_maps.shape[0], params)
    first_label_slug = first_label.lower().replace(" ", "_")
    second_label_slug = second_label.lower().replace(" ", "_")

    lines = [
        "Gravity comparison summary",
        f"gravity_spc = {int(params.get('gravity_spc', 1))}",
        f"gravity_kernel = {params.get('gravity_kernel', 'numba')}",
        f"gravity_density_mode = {params.get('gravity_density_mode', 'decreasing')}",
        f"gravity_constant_delta_rho = {float(params.get('gravity_constant_delta_rho', -170.0))}",
        f"gravity_compaction_lithology = {params.get('gravity_compaction_lithology', 'shaly_sand')}",
        f"gravity_compaction_layer_thickness = {float(params.get('gravity_compaction_layer_thickness', 50.0))}",
        f"gravity_host_density = {float(params.get('gravity_host_density', 2670.0))}",
        f"gravity_grain_density = {float(params.get('gravity_grain_density', 2650.0))}",
        f"gravity_fluid_density = {float(params.get('gravity_fluid_density', 1000.0))}",
        f"gravity_add_initial_background = {bool(params.get('gravity_add_initial_background', True))}",
        "",
        (
            "snapshot_Ma "
            f"{first_label_slug}_min_mgal {first_label_slug}_max_mgal "
            f"{second_label_slug}_min_mgal {second_label_slug}_max_mgal "
            "diff_rms_mgal diff_max_abs_mgal"
        ),
    ]

    for idx, time_Ma in enumerate(snapshot_times_Ma):
        diff_map = diff_maps[idx]
        lines.append(
            f"{time_Ma:.6f} "
            f"{np.nanmin(first_maps[idx]):.6f} {np.nanmax(first_maps[idx]):.6f} "
            f"{np.nanmin(second_maps[idx]):.6f} {np.nanmax(second_maps[idx]):.6f} "
            f"{np.sqrt(np.mean(diff_map ** 2)):.6f} {np.max(np.abs(diff_map)):.6f}"
        )

    with open(save_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def plot_magnetic_maps(mag_maps, X, Y, params, title="Anomalies magnétiques",
                       save_path=None, show=True, dpi=300):
    """MGM internal routine."""
    nb_step = params['nb_step']
    
    
    mod = ((params['t'] - params['dt']) / params['dt']) / (nb_step - 1)
    x_temps = np.array([i * mod * params['dt'] for i in range(nb_step)]) / 1e6  
    
    idx_times = np.linspace(0, nb_step - 1, 4, dtype=int)
    
    
    ext = [
        X.min() / 1000, X.max() / 1000,
        Y.min() / 1000, Y.max() / 1000
    ]
    
    
    axes_size = 30
    fig_mag = plt.figure(figsize=(26, 13))
    gs_mag = gridspec.GridSpec(
        2, 3,
        figure=fig_mag,
        width_ratios=[1, 1, 0.05],
        hspace=0.25,
        wspace=0.1
    )
    
    axm1 = fig_mag.add_subplot(gs_mag[0, 0])
    axm2 = fig_mag.add_subplot(gs_mag[0, 1])
    axm3 = fig_mag.add_subplot(gs_mag[1, 0])
    axm4 = fig_mag.add_subplot(gs_mag[1, 1])
    cax_mag = fig_mag.add_subplot(gs_mag[:, 2])
    
    
    cmap_mag = cm.get_cmap('viridis')
    vmax_mag = np.nanmax(mag_maps)
    normalizer_mag = Normalize(vmin=0.0, vmax=vmax_mag)
    
    
    for ax, it in zip((axm1, axm2, axm3, axm4), idx_times):
        mag = mag_maps[it, :, :]
        t_Ma = x_temps[it]
        im_mag = ax.imshow(
            mag,
            extent=ext,
            cmap=cmap_mag,
            norm=normalizer_mag
        )
        ax.set_title(
            r'$Time = %.2f\,\mathrm{Ma}$' % t_Ma,
            fontsize=axes_size
        )
    
    
    for ax in (axm1, axm3):
        ax.set_ylabel('Y [km]', fontsize=axes_size)
    for ax in (axm3, axm4):
        ax.set_xlabel('X [km]', fontsize=axes_size)
    
    
    cbar_mag = fig_mag.colorbar(im_mag, cax=cax_mag, aspect=20)
    cbar_mag.set_label(r'$|\Delta T|$ [nT]', fontsize=axes_size)
    
    fig_mag.suptitle(title, fontsize=axes_size + 5, y=0.98)
    if save_path is not None:
        fig_mag.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig_mag)


def plot_magnetic_profiles(TOPO, mag_prof, X, Y, params, title="Profils magnétiques",
                           save_path=None, show=True, dpi=300):
    """MGM internal routine."""
    nb_step = params['nb_step']
    
    
    mod = ((params['t'] - params['dt']) / params['dt']) / (nb_step - 1)
    x_temps = np.array([i * mod * params['dt'] for i in range(nb_step)]) / 1e6  
    
    rows, colums = TOPO[0].shape
    mid_y = rows // 2
    mid_x = colums // 2
    
    x_profil = X[mid_y, :]
    x_km_half = x_profil[mid_x:] / 1000.0
    
    fig, (ax_topo, ax_mag) = plt.subplots(1, 2, figsize=(13, 5))
    
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=nb_step - 1)
    
    for istep in range(nb_step):
        color = cmap(norm(istep))
        
        
        z_prof_half = TOPO[istep][mid_y, mid_x:]
        ax_topo.plot(
            x_km_half, z_prof_half,
            color=color,
            label=f"t = {x_temps[istep]:.2f} Ma"
        )
        
        
        dT_prof_half = mag_prof[istep, mid_x:]
        ax_mag.plot(
            x_km_half, dT_prof_half,
            color=color,
            label=f"t = {x_temps[istep]:.2f} Ma"
        )
    
    
    ax_topo.set_title("Profil topographique (centre → Est)")
    ax_topo.set_xlabel("Distance [km]")
    ax_topo.set_ylabel("Altitude [m]")
    ax_topo.legend(fontsize=12)
    ax_topo.grid(True, alpha=0.25)
    
    ax_mag.set_title(title)
    ax_mag.set_xlabel("Distance [km]")
    ax_mag.set_ylabel(r"$\Delta T$ [nT]")
    ax_mag.legend(fontsize=12)
    ax_mag.grid(True, alpha=0.25)
    
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_layer_evolution_1layer(TOPO, top_rem_all, base_mag, X, Y, params,
                                save_path=None, show=True, dpi=300):
    """MGM internal routine."""

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.collections import PolyCollection

    nb_step = params["nb_step"]

    
    mod = ((params["t"] - params["dt"]) / params["dt"]) / (nb_step - 1)
    x_temps = np.array([i * mod * params["dt"] for i in range(nb_step)]) / 1e6

    rows, colums = TOPO[0].shape
    mid_y = rows // 2
    mid_x = colums // 2

    x_profil = X[mid_y, :]
    x_km_half = x_profil[mid_x:] / 1000.0

    snap_steps = [0, nb_step // 3, 2 * nb_step // 3, nb_step - 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    def build_layer_collection(ax, x_km, top_layer, base_layer, values, ok, cmap, norm, label=None, alpha=0.8):
        polys = []
        vals = []
        for i in range(len(x_km) - 1):
            if not (ok[i] and ok[i + 1]):
                continue
            polys.append([
                (x_km[i], top_layer[i]),
                (x_km[i + 1], top_layer[i + 1]),
                (x_km[i + 1], base_layer[i + 1]),
                (x_km[i], base_layer[i]),
            ])
            vals.append(0.5 * (values[i] + values[i + 1]))
        if not polys:
            return None
        pc = PolyCollection(
            polys,
            array=np.asarray(vals, dtype=float),
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            alpha=alpha
        )
        if label is not None:
            pc.set_label(label)
        ax.add_collection(pc)
        return pc

    def compute_M_map(top0_down, base0_down, X, Y, params_local):
        H0 = base0_down - top0_down
        valid = np.isfinite(H0) & (H0 > 0.0)
        z_center = top0_down + 0.5 * H0

        xc = float(np.nanmean(X))
        yc = float(np.nanmean(Y))
        r = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)

        crater_radius = params_local.get("crater_radius", 7500.0)
        if crater_radius is None:
            crater_radius = float(np.nanmax(r))
        else:
            crater_radius = float(crater_radius)

        r_scale_frac = float(params_local.get("r_scale_frac", 0.35))
        sigma_r = max(1e-6, r_scale_frac * crater_radius)

        if np.any(valid):
            zc = float(np.nanmean(z_center[valid]))
        else:
            zc = 0.0

        z_scale = float(params_local.get("z_scale", 1500.0))
        z_scale = max(z_scale, 1e-6)

        M_center = float(params_local.get("M_center", 0.5))
        M_background = float(params_local.get("M_background", 0.02))

        mag = M_background + (M_center - M_background) * \
              np.exp(-(r ** 2) / (2.0 * sigma_r ** 2)) * \
              np.exp(-np.abs(z_center - zc) / z_scale)

        mag[~valid] = np.nan
        return mag

    
    if isinstance(base_mag, np.ndarray) and base_mag.ndim == 2:
        base0_down = base_mag.copy()
    elif isinstance(base_mag, np.ndarray) and base_mag.ndim == 3:
        base0_down = base_mag[0].copy()
    else:
        raise ValueError("base_mag doit être un array 2D (fixe) ou 3D (dynamique).")

    top0_down = top_rem_all[0].copy()

    M_params = {
        "crater_radius": params.get("crater_radius", 7500.0),
        "r_scale_frac": params.get("r_scale_frac", 0.35),
        "z_scale": params.get("z_scale", 1500.0),
        "M_center": params.get("M_center", 0.5),
        "M_background": params.get("M_background", 0.02),
    }
    M_map = compute_M_map(top0_down, base0_down, X, Y, M_params)
    M_line = M_map[mid_y, mid_x:]

    M_vals = M_line[np.isfinite(M_line)]
    if M_vals.size > 0:
        vmin = float(np.nanmin(M_vals))
        vmax = float(np.nanmax(M_vals))
    else:
        vmin, vmax = 0.0, 1.0
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        vmin, vmax = 0.0, max(1.0, float(vmax) if np.isfinite(vmax) else 1.0)

    cmap = cm.get_cmap("viridis")
    norm = Normalize(vmin=vmin, vmax=vmax)

    for k, istep in enumerate(snap_steps):
        ax = axes[k]

        
        topo_half = TOPO[istep][mid_y, mid_x:]

        
        top_layer_half = -top_rem_all[istep][mid_y, mid_x:].copy()

        if isinstance(base_mag, np.ndarray) and base_mag.ndim == 2:
            base_layer_half = -base_mag[mid_y, mid_x:].copy()
        elif isinstance(base_mag, np.ndarray) and base_mag.ndim == 3:
            base_layer_half = -base_mag[istep][mid_y, mid_x:].copy()
        else:
            raise ValueError("base_mag doit être un array 2D (fixe) ou 3D (dynamique).")

        
        layer_ok = np.isfinite(top_layer_half) & np.isfinite(base_layer_half) & (top_layer_half > base_layer_half)
        top_layer_half[~layer_ok] = np.nan
        base_layer_half[~layer_ok] = np.nan
        M_line_half = M_line

        
        ax.plot(x_km_half, topo_half, lw=2.0, color="k", label="Topography")

        build_layer_collection(
            ax,
            x_km_half,
            top_layer_half,
            base_layer_half,
            M_line_half,
            layer_ok,
            cmap,
            norm,
            label="Magnetized layer" if k == 0 else None,
            alpha=0.85
        )

        ax.plot(
            x_km_half,
            base_layer_half,
            ls=":",
            lw=1.2,
            color="tab:blue",
            label="Layer base"
        )

        ax.set_title(r'$Time = %.2f\,\mathrm{Ma}$' % x_temps[istep])
        ax.grid(True, alpha=0.3)

    for ax in axes[2:]:
        ax.set_xlabel("Distance [km]")
    for ax in axes[::2]:
        ax.set_ylabel("Altitude [m]")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.11)
    )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.25, 0.05, 0.50, 0.02])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Aimantation M [A/m]")

    plt.tight_layout(rect=[0, 0.14, 1, 1])
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_layer_evolution_2layers(TOPO, top1_rem_all, base1, top2_rem_all, base2, X, Y, params,
                                 save_path=None, show=True, dpi=300):
    """MGM internal routine."""
    
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.collections import PolyCollection

    nb_step = params["nb_step"]

    mod = ((params["t"] - params["dt"]) / params["dt"]) / (nb_step - 1)
    x_temps = np.array([i * mod * params["dt"] for i in range(nb_step)]) / 1e6  

    rows, colums = TOPO[0].shape
    mid_y = rows // 2
    mid_x = colums // 2

    x_profil = X[mid_y, :]
    x_km_half = x_profil[mid_x:] / 1000.0

    snap_steps = np.array([0, nb_step // 3, 2 * nb_step // 3, nb_step - 1], dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    
    base1_up_full = -base1
    base2_up_full = (-base2) if (base2 is not None) else None

    def build_layer_collection(ax, x_km, top_layer, base_layer, values, ok, cmap, norm, label=None, alpha=0.8):
        polys = []
        vals = []
        for i in range(len(x_km) - 1):
            if not (ok[i] and ok[i + 1]):
                continue
            polys.append([
                (x_km[i], top_layer[i]),
                (x_km[i + 1], top_layer[i + 1]),
                (x_km[i + 1], base_layer[i + 1]),
                (x_km[i], base_layer[i]),
            ])
            vals.append(0.5 * (values[i] + values[i + 1]))
        if not polys:
            return None
        pc = PolyCollection(
            polys,
            array=np.asarray(vals, dtype=float),
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            alpha=alpha
        )
        if label is not None:
            pc.set_label(label)
        ax.add_collection(pc)
        return pc

    def compute_M_map(top0_down, base0_down, X, Y, params_local):
        H0 = base0_down - top0_down
        valid = np.isfinite(H0) & (H0 > 0.0)
        z_center = top0_down + 0.5 * H0

        xc = float(np.nanmean(X))
        yc = float(np.nanmean(Y))
        r = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)

        crater_radius = params_local.get("crater_radius", 15000.0)
        if crater_radius is None:
            crater_radius = float(np.nanmax(r))
        else:
            crater_radius = float(crater_radius)

        r_scale_frac = float(params_local.get("r_scale_frac", 0.35))
        sigma_r = max(1e-6, r_scale_frac * crater_radius)

        if np.any(valid):
            zc = float(np.nanmean(z_center[valid]))
        else:
            zc = 0.0

        z_scale = float(params_local.get("z_scale", 1500.0))
        z_scale = max(z_scale, 1e-6)

        M_center = float(params_local.get("M_center", 0.5))
        M_background = float(params_local.get("M_background", 0.02))

        mag = M_background + (M_center - M_background) * \
              np.exp(-(r ** 2) / (2.0 * sigma_r ** 2)) * \
              np.exp(-np.abs(z_center - zc) / z_scale)

        mag[~valid] = np.nan
        return mag

    top1_0_down = top1_rem_all[0].copy()
    if isinstance(base1, np.ndarray) and base1.ndim == 2:
        base1_0_down = base1.copy()
    elif isinstance(base1, np.ndarray) and base1.ndim == 3:
        base1_0_down = base1[0].copy()
    else:
        raise ValueError("base1 doit être un array 2D (fixe) ou 3D (dynamique).")
    M1_params = {
        "crater_radius": params.get("crater_radius", 15000.0),
        "r_scale_frac": params.get("r_scale_frac1", params.get("r_scale_frac", 0.35)),
        "z_scale": params.get("z_scale1", params.get("z_scale", 1500.0)),
        "M_center": params.get("M_center1", 0.4),
        "M_background": params.get("M_background1", 0.02),
    }
    M1_map = compute_M_map(top1_0_down, base1_0_down, X, Y, M1_params)
    M1_line = M1_map[mid_y, mid_x:]

    if (top2_rem_all is not None) and (base2 is not None):
        top2_0_down = top2_rem_all[0].copy()
        if isinstance(base2, np.ndarray) and base2.ndim == 2:
            base2_0_down = base2.copy()
        elif isinstance(base2, np.ndarray) and base2.ndim == 3:
            base2_0_down = base2[0].copy()
        else:
            raise ValueError("base2 doit être un array 2D (fixe) ou 3D (dynamique).")
        M2_params = {
            "crater_radius": params.get("crater_radius", 15000.0),
            "r_scale_frac": params.get("r_scale_frac2", params.get("r_scale_frac", 0.35)),
            "z_scale": params.get("z_scale2", params.get("z_scale", 1500.0)),
            "M_center": params.get("M_center2", 0.6),
            "M_background": params.get("M_background2", 0.01),
        }
        M2_map = compute_M_map(top2_0_down, base2_0_down, X, Y, M2_params)
        M2_line = M2_map[mid_y, mid_x:]
    else:
        M2_line = None

    M_vals = [M1_line[np.isfinite(M1_line)]]
    if M2_line is not None:
        M_vals.append(M2_line[np.isfinite(M2_line)])
    if any(v.size > 0 for v in M_vals):
        vmin = float(np.nanmin(np.concatenate(M_vals)))
        vmax = float(np.nanmax(np.concatenate(M_vals)))
    else:
        vmin, vmax = 0.0, 1.0
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        vmin, vmax = 0.0, max(1.0, float(vmax) if np.isfinite(vmax) else 1.0)

    cmap = cm.get_cmap("viridis")
    norm = Normalize(vmin=vmin, vmax=vmax)

    for k, istep in enumerate(snap_steps):
        ax = axes[k]

        topo_half_up = TOPO[istep][mid_y, mid_x:]

        
        top1_half_up = -top1_rem_all[istep][mid_y, mid_x:]
        if isinstance(base1_up_full, np.ndarray) and base1_up_full.ndim == 2:
            base1_half_up = base1_up_full[mid_y, mid_x:]
        elif isinstance(base1_up_full, np.ndarray) and base1_up_full.ndim == 3:
            base1_half_up = base1_up_full[istep][mid_y, mid_x:]
        else:
            raise ValueError("base1 doit être un array 2D (fixe) ou 3D (dynamique).")
        ok1 = np.isfinite(top1_half_up) & np.isfinite(base1_half_up) & (top1_half_up > base1_half_up)
        M1_line_half = M1_line

        
        ok2 = None
        if (top2_rem_all is not None) and (base2_up_full is not None):
            top2_half_up = -top2_rem_all[istep][mid_y, mid_x:]
            if isinstance(base2_up_full, np.ndarray) and base2_up_full.ndim == 2:
                base2_half_up = base2_up_full[mid_y, mid_x:]
            elif isinstance(base2_up_full, np.ndarray) and base2_up_full.ndim == 3:
                base2_half_up = base2_up_full[istep][mid_y, mid_x:]
            else:
                raise ValueError("base2 doit être un array 2D (fixe) ou 3D (dynamique).")
            ok2 = np.isfinite(top2_half_up) & np.isfinite(base2_half_up) & (top2_half_up > base2_half_up)
            M2_line_half = M2_line
        else:
            top2_half_up = None
            base2_half_up = None

        
        ax.plot(x_km_half, topo_half_up, lw=2, label="Topographie")

        build_layer_collection(
            ax,
            x_km_half,
            top1_half_up,
            base1_half_up,
            M1_line_half,
            ok1,
            cmap,
            norm,
            label="Couche 1 restante" if k == 0 else None,
            alpha=0.85
        )
        ax.plot(
            x_km_half,
            np.where(ok1, base1_half_up, np.nan),
            ls=":",
            lw=1.0,
            label="Interface inférieure couche 1"
        )

        if ok2 is not None:
            build_layer_collection(
                ax,
                x_km_half,
                top2_half_up,
                base2_half_up,
                M2_line_half,
                ok2,
                cmap,
                norm,
                label="Couche 2 restante" if k == 0 else None,
                alpha=0.70
            )
            ax.plot(
                x_km_half,
                np.where(ok2, base2_half_up, np.nan),
                ls="--",
                lw=1.0,
                label="Interface inférieure couche 2"
            )

        ax.set_title(f"t = {x_temps[istep]:.2f} Ma")
        ax.grid(True, alpha=0.25)

    for ax in axes[2:]:
        ax.set_xlabel("Distance [km]")
    for ax in axes[::2]:
        ax.set_ylabel("Altitude [m]")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.08)
    )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.25, 0.03, 0.50, 0.02])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Aimantation M [A/m]")

    plt.tight_layout(rect=[0, 0.18, 1, 1])
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)






def plot_3d_pyvista(TOPO, mag_maps, X, Y, params,
                    save_path=None, show=True):
    """MGM internal routine."""
    import pyvista as pv

    z_init = TOPO[0]
    z_final = TOPO[-1]
    nby, nbx = z_final.shape
    mag_final = mag_maps[-1]

    
    Xkm = X / 1000.0
    Ykm = Y / 1000.0

    
    extent_km = (X.max() - X.min()) / 1000.0
    z_range = max(np.nanmax(z_final) - np.nanmin(z_final), 1.0)
    z_factor = extent_km / z_range * 0.15

    
    dz = z_final - z_init

    
    sediment = np.clip(dz, 0, None)

    
    def make_grid(Zvals, z_exag):
        grid = pv.StructuredGrid(Xkm, Ykm, Zvals * z_exag)
        return grid

    pv.global_theme.background = "white"
    pv.global_theme.font.color = "black"

    plotter = pv.Plotter(shape=(2, 2), window_size=[1600, 1200],
                         off_screen=(not show))

    
    plotter.subplot(0, 0)
    grid_topo = make_grid(z_final, z_factor)
    grid_topo["Altitude [m]"] = z_final.ravel(order="F")
    plotter.add_mesh(grid_topo, scalars="Altitude [m]", cmap="terrain",
                     show_edges=False, lighting=True)
    plotter.add_text("Topographie (finale)", font_size=10, position="upper_left")
    plotter.show_axes()

    
    plotter.subplot(0, 1)
    grid_dz = make_grid(z_final, z_factor)
    grid_dz["dz [m]"] = dz.ravel(order="F")
    clim = max(np.nanmax(np.abs(dz)), 1.0)
    plotter.add_mesh(grid_dz, scalars="dz [m]", cmap="RdBu_r",
                     clim=[-clim, clim], show_edges=False, lighting=True)
    plotter.add_text("Changement topo (final - init)", font_size=10,
                     position="upper_left")
    plotter.show_axes()

    
    plotter.subplot(1, 0)
    grid_sed = make_grid(z_final, z_factor)
    grid_sed["Sédiments [m]"] = sediment.ravel(order="F")
    plotter.add_mesh(grid_sed, scalars="Sédiments [m]", cmap="YlOrBr",
                     show_edges=False, lighting=True)
    plotter.add_text("Épaisseur sédimentaire", font_size=10,
                     position="upper_left")
    plotter.show_axes()

    
    plotter.subplot(1, 1)
    grid_mag = make_grid(z_final, z_factor)
    grid_mag["Mag [nT]"] = mag_final.ravel(order="F")
    plotter.add_mesh(grid_mag, scalars="Mag [nT]", cmap="coolwarm",
                     show_edges=False, lighting=True)
    plotter.add_text("Anomalie magnétique (1 couche)", font_size=10,
                     position="upper_left")
    plotter.show_axes()

    
    plotter.link_views()

    
    if save_path is not None:
        plotter.screenshot(save_path)
    if show:
        plotter.show()
    else:
        plotter.close()






def main(
    Te=None,
    uplift_rate_m_per_Ma=None,
    run_magnetics=True,
    run_gravity=None,
    gravity_method=None,
    gravity_kernel=None,
    gravity_spc=None,
    gravity_terrain_spc=None,
    gravity_terrain_density=None,
    gravity_density_mode=None,
    gravity_constant_delta_rho=None,
    gravity_compaction_lithology=None,
    gravity_compaction_layer_thickness=None,
    gravity_host_density=None,
    gravity_grain_density=None,
    gravity_fluid_density=None,
    gravity_add_initial_background=None,
    save_figures=None,
    show_figures=None,
    show_progress=None,
    log_progress_interval_yr=None,
    save_topography_maps=None,
    random_seed=None,
    initial_seed_path=None,
    surface_profile_path=None,
    xy_space_m=None,
    output_root=None,
    total_time_yr=None,
    time_step_yr=None,
    flexure_time_step_yr=None,
    topography_snapshot_times_Ma=None,
    boundary_mode=None,
    stop_on_crater_breach=None,
    crater_radius_m=None,
    crater_mask_radius_m=None,
    event_check_interval_yr=None,
    crater_event_min_area_km2=None,
    flexure_include_direct_uplift_load=None,
):
    """MGM internal routine."""
    
    
    

    print("═" * 80)
    print("INITIALISATION")
    print("═" * 80)

    
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    
    params = initialize_simulation()

    
    if Te is not None:
        params['Te'] = Te
    if uplift_rate_m_per_Ma is not None:
        params['uplift_rate_m_per_Ma'] = uplift_rate_m_per_Ma
        params['uplift_rate'] = uplift_rate_m_per_Ma * 1e-6
    if xy_space_m is not None:
        params['xy_space'] = float(xy_space_m)
    if output_root is not None:
        params['output_root'] = str(output_root)
    if boundary_mode is not None:
        params['boundary_mode'] = str(boundary_mode).lower()
    if total_time_yr is not None:
        params['t'] = int(total_time_yr)
        params['nt'] = int((params['t'] - params['dt']) // params['dt']) + 1
    if time_step_yr is not None:
        params['dt'] = int(time_step_yr)
        if params['dt'] <= 0:
            raise ValueError('Le pas de temps doit être strictement positif.')
        params['nt'] = int((params['t'] - params['dt']) // params['dt']) + 1
    if flexure_time_step_yr is not None:
        params['dt_flex'] = int(flexure_time_step_yr)
        if params['dt_flex'] <= 0:
            raise ValueError("Le pas de temps de flexure doit être strictement positif.")
    if topography_snapshot_times_Ma is not None:
        snapshot_times_Ma = np.asarray(topography_snapshot_times_Ma, dtype=float).reshape(-1)
        if snapshot_times_Ma.size == 0:
            raise ValueError("topography_snapshot_times_Ma ne peut pas être vide.")
        params['topography_snapshot_times_yr'] = (snapshot_times_Ma * 1.0e6).tolist()
        params['nb_step'] = int(snapshot_times_Ma.size)
    if save_figures is not None:
        params['save_figures'] = bool(save_figures)
    if show_figures is not None:
        params['show_figures'] = bool(show_figures)
    if run_gravity is not None:
        params['run_gravity'] = bool(run_gravity)
    if gravity_method is not None:
        params['gravity_method'] = str(gravity_method).lower()
    if gravity_kernel is not None:
        params['gravity_kernel'] = str(gravity_kernel).lower()
    if gravity_spc is not None:
        params['gravity_spc'] = int(gravity_spc)
    if gravity_terrain_spc is not None:
        params['gravity_terrain_spc'] = int(gravity_terrain_spc)
    if gravity_terrain_density is not None:
        params['gravity_terrain_density'] = float(gravity_terrain_density)
    if gravity_density_mode is not None:
        params['gravity_density_mode'] = str(gravity_density_mode).lower()
    if gravity_constant_delta_rho is not None:
        params['gravity_constant_delta_rho'] = float(gravity_constant_delta_rho)
    if gravity_compaction_lithology is not None:
        params['gravity_compaction_lithology'] = str(gravity_compaction_lithology).lower()
    if gravity_compaction_layer_thickness is not None:
        params['gravity_compaction_layer_thickness'] = float(gravity_compaction_layer_thickness)
    if gravity_host_density is not None:
        params['gravity_host_density'] = float(gravity_host_density)
    if gravity_grain_density is not None:
        params['gravity_grain_density'] = float(gravity_grain_density)
    if gravity_fluid_density is not None:
        params['gravity_fluid_density'] = float(gravity_fluid_density)
    if gravity_add_initial_background is not None:
        params['gravity_add_initial_background'] = bool(gravity_add_initial_background)
    if show_progress is not None:
        params['show_progress'] = bool(show_progress)
    if log_progress_interval_yr is not None:
        params['log_progress_interval_yr'] = max(int(log_progress_interval_yr), params['dt'])
    if save_topography_maps is not None:
        params['save_topography_maps'] = bool(save_topography_maps)
    if random_seed is not None:
        params['random_seed'] = random_seed
    if initial_seed_path is not None:
        params['initial_seed_path'] = str(initial_seed_path)
    if surface_profile_path is not None:
        params['file_path'] = str(surface_profile_path)
    if stop_on_crater_breach is not None:
        params['stop_on_crater_breach'] = bool(stop_on_crater_breach)
    if crater_radius_m is not None:
        params['crater_radius_m'] = float(crater_radius_m)
    if crater_mask_radius_m is not None:
        params['crater_mask_radius_m'] = float(crater_mask_radius_m)
    if event_check_interval_yr is not None:
        params['event_check_interval_yr'] = int(event_check_interval_yr)
    if crater_event_min_area_km2 is not None:
        params['crater_event_min_area_m2'] = float(crater_event_min_area_km2) * 1.0e6
    if flexure_include_direct_uplift_load is not None:
        params['flexure_include_direct_uplift_load'] = bool(flexure_include_direct_uplift_load)

    if params.get('stop_on_crater_breach', False) and run_magnetics:
        raise ValueError("stop_on_crater_breach=True nécessite run_magnetics=False pour éviter un post-traitement incomplet.")
    if int(params.get("gravity_spc", 1)) < 0:
        raise ValueError("gravity_spc doit être positif ou nul.")
    if int(params.get("gravity_terrain_spc", 1)) < 0:
        raise ValueError("gravity_terrain_spc doit être positif ou nul.")
    if float(params.get("gravity_terrain_density", 2670.0)) <= 0.0:
        raise ValueError("gravity_terrain_density doit être strictement positif.")
    if str(params.get("gravity_density_mode", "decreasing")).lower() not in {
        "decreasing", "constant", "compaction", "compaction_layered"
    }:
        raise ValueError(
            "gravity_density_mode doit valoir 'decreasing', 'constant', 'compaction' ou 'compaction_layered'."
        )
    if not np.isfinite(float(params.get("gravity_constant_delta_rho", -170.0))):
        raise ValueError("gravity_constant_delta_rho doit être un nombre fini.")
    if str(params.get("gravity_compaction_lithology", "shaly_sand")).lower() not in {"sand", "shaly_sand", "shale"}:
        raise ValueError("gravity_compaction_lithology doit valoir 'sand', 'shaly_sand' ou 'shale'.")
    if float(params.get("gravity_compaction_layer_thickness", 50.0)) <= 0.0:
        raise ValueError("gravity_compaction_layer_thickness doit être strictement positif.")
    if float(params.get("gravity_host_density", 2670.0)) <= 0.0:
        raise ValueError("gravity_host_density doit être strictement positif.")
    if float(params.get("gravity_grain_density", 2650.0)) <= 0.0:
        raise ValueError("gravity_grain_density doit être strictement positif.")
    if float(params.get("gravity_fluid_density", 1000.0)) < 0.0:
        raise ValueError("gravity_fluid_density doit être positif ou nul.")

    
    params['nb_step_flex'] = max(1, int(round(params['t'] / params['dt_flex'])))
    params['mod_flex'] = max(1, int(round(params['dt_flex'] / params['dt'])))
    params['figures_root'] = build_run_output_dir(params)
    
    seed_path = params.get("initial_seed_path")
    if seed_path:
        seed_payload = load_initial_seed(seed_path)
        seed_xy_space = seed_payload.get("xy_space")
        if seed_xy_space is not None and abs(float(params["xy_space"]) - seed_xy_space) > 1e-9:
            raise ValueError(
                f"Le seed {seed_path} a xy_space={seed_xy_space:g} m, incompatible avec "
                f"--xy-space={params['xy_space']:g} m."
            )
        if seed_xy_space is not None:
            params["xy_space"] = float(seed_xy_space)
        if seed_payload.get("limit") is not None:
            params["limit"] = float(seed_payload["limit"])
        params["initial_bedrock_seed"] = seed_payload.get("bedrock_final")
        params["initial_soil_seed"] = seed_payload.get("soil_depth_final")
        z = seed_payload["z_final"].copy()
        if seed_payload.get("X") is not None and seed_payload.get("Y") is not None:
            X = seed_payload["X"].copy()
            Y = seed_payload["Y"].copy()
        else:
            _, X, Y = generate_topography(
                params['file_path'],
                params['xy_space'],
                params['limit']
            )
        print(f"Topo initiale chargée depuis le seed : {seed_path}")
    else:
        params["initial_bedrock_seed"] = None
        params["initial_soil_seed"] = None
        z, X, Y = generate_topography(
            params['file_path'],
            params['xy_space'],
            params['limit']
        )
        print(f"Topo initiale chargée depuis le profil source : {params['file_path']}")

    
    barringer, components = initialize_grid(z, params['xy_space'], params)
    
    print(f"Grille : {z.shape[0]} × {z.shape[1]} nœuds")
    print(f"Espacement : {params['xy_space']} m")
    print(f"Durée simulation : {params['t']} ans")
    print(f"Pas de temps : {params['dt']} ans")
    print(f"Nombre d'itérations : {params['nt']}")
    print()
    
    
    
    
    
    print("═" * 80)
    print("SIMULATION")
    print("═" * 80)
    
    results = run_simulation(barringer, components, z, X, Y, params)
    params["snapshot_times_yr"] = np.asarray(results.get("topography_snapshot_times_yr", []), dtype=float)

    
    components['frr'].run_one_step()
    drainage_summary = compute_drainage_summary(barringer, params)
    drainage_summary["requested_time_yr"] = int(params["t"])
    drainage_summary["executed_time_yr"] = int(results.get("executed_time_yr", params["t"]))
    drainage_summary["time_step_yr"] = int(params["dt"])
    drainage_summary["flexure_time_step_yr"] = int(params["dt_flex"])
    drainage_summary["stop_reason"] = results.get("stop_reason", "completed")
    drainage_summary["stop_on_crater_breach"] = bool(params.get("stop_on_crater_breach", False))
    drainage_summary["crater_radius_m"] = float(params.get("crater_radius_m", 0.0))
    drainage_summary["crater_mask_radius_m"] = float(params.get("crater_mask_radius_m", 0.0))
    drainage_summary["event_check_interval_yr"] = int(params.get("event_check_interval_yr", params["dt"]))
    drainage_summary["crater_event_min_area_m2"] = float(params.get("crater_event_min_area_m2", 0.0))
    drainage_summary["flexure_include_direct_uplift_load"] = bool(params.get("flexure_include_direct_uplift_load", True))
    drainage_summary["crater_event"] = results.get("crater_event", {"detected": False})
    summary_path = save_drainage_summary(drainage_summary, params['figures_root'])
    params["executed_time_yr"] = drainage_summary["executed_time_yr"]

    print()
    print("Simulation terminée !")
    print(
        "Drainage final : "
        f"{drainage_summary['north_south_label']} "
        f"(score Nord-Sud = {drainage_summary['north_south_score']:+.3f}, "
        f"côté dominant global = {drainage_summary['dominant_outlet_side']})"
    )
    print(f"Résumé drainage : {summary_path}")
    print()

    payload = {
        "params": params,
        "results": results,
        "drainage_summary": drainage_summary,
        "X": X,
        "Y": Y,
    }

    if params.get("save_topography_maps", False):
        topo_maps_path = os.path.join(params["figures_root"], "maps_topography.png")
        plot_topography_maps(
            results["TOPO"], X, Y, params,
            title="Evolution topographique",
            save_path=topo_maps_path,
            show=params.get("show_figures", False),
            dpi=params.get("save_dpi", 300),
        )
        print(f"Figure topographie : {topo_maps_path}")

        flexure_maps_path = os.path.join(params["figures_root"], "maps_flexure.png")
        plot_flexure_maps(
            results["FLEX"], X, Y, params,
            title="Evolution de la flexure cumulée",
            save_path=flexure_maps_path,
            show=params.get("show_figures", False),
            dpi=params.get("save_dpi", 300),
        )
        print(f"Figure flexure : {flexure_maps_path}")
        print()

    gravity_outputs = {}
    if params.get("run_gravity", False):
        print("═" * 80)
        print("POST-TRAITEMENT : ANOMALIES GRAVIMETRIQUES")
        print("═" * 80)

        gravity_outputs = compute_gravity_anomalies(
            results["TOPO"],
            results["EPAIS"],
            X,
            Y,
            params,
        )
        payload["gravity"] = gravity_outputs

        save_figures = params.get("save_figures", False)
        show_figures = params.get("show_figures", True)
        save_dpi = params.get("save_dpi", 300)
        gravity_root = os.path.join(params["figures_root"], "gravity")
        gravity_field_order = [
            "legacy",
            "reference",
            "complete",
            "simple_bouguer",
            "terrain_correction",
            "terrain_exact",
            "bouguer_slab",
        ]
        gravity_comparison_order = ["comparison", "comparison_legacy_complete"]

        if save_figures:
            for folder_name in gravity_outputs:
                os.makedirs(os.path.join(gravity_root, folder_name), exist_ok=True)

        for field_name in gravity_field_order:
            if field_name not in gravity_outputs:
                continue

            field_payload = gravity_outputs[field_name]
            field_maps = field_payload["maps"]
            field_final = field_maps[-1]
            backend = field_payload.get("backend")
            print(
                f"{field_payload['title']} : "
                f"{'backend=' + str(backend) + ', ' if backend else ''}"
                f"min={np.nanmin(field_final):.3f} mGal, "
                f"max={np.nanmax(field_final):.3f} mGal"
            )
            if save_figures:
                export_gravity_field_data(
                    results["TOPO"],
                    field_maps,
                    X,
                    Y,
                    params,
                    field_name=field_name,
                    title=field_payload["title"],
                    backend=backend,
                    save_path=os.path.join(gravity_root, field_name, "gravity_data.npz"),
                )
            plot_gravity_maps(
                field_maps,
                X,
                Y,
                params,
                title=field_payload["title"],
                save_path=os.path.join(gravity_root, field_name, "gravity_maps.png") if save_figures else None,
                show=show_figures,
                dpi=save_dpi,
            )
            plot_gravity_profiles_center_west(
                results["TOPO"],
                field_maps,
                X,
                params,
                title=f"Profil gravimetrique - {field_payload['title']}",
                save_path=(
                    os.path.join(gravity_root, field_name, "gravity_profiles_center_west.png")
                    if save_figures else None
                ),
                show=show_figures,
                dpi=save_dpi,
            )

        for comparison_name in gravity_comparison_order:
            if comparison_name not in gravity_outputs:
                continue

            comparison_payload = gravity_outputs[comparison_name]
            diff_maps = comparison_payload["maps"]
            diff_final = diff_maps[-1]
            first_label = comparison_payload.get("first_label", "First")
            second_label = comparison_payload.get("second_label", "Second")
            first_key = "legacy" if first_label.lower() == "legacy" else first_label.lower()
            second_key = "reference" if second_label.lower() == "reference" else second_label.lower()
            if first_key not in gravity_outputs or second_key not in gravity_outputs:
                continue

            print(
                f"{comparison_payload['title']} : "
                f"rms final={np.sqrt(np.mean(diff_final ** 2)):.3f} mGal, "
                f"max abs final={np.max(np.abs(diff_final)):.3f} mGal"
            )
            plot_gravity_maps(
                diff_maps,
                X,
                Y,
                params,
                title=comparison_payload["title"],
                colorbar_label="Difference [mGal]",
                save_path=os.path.join(gravity_root, comparison_name, "gravity_difference_maps.png") if save_figures else None,
                show=show_figures,
                dpi=save_dpi,
            )
            plot_gravity_profiles_comparison_center_west(
                results["TOPO"],
                gravity_outputs[first_key]["maps"],
                gravity_outputs[second_key]["maps"],
                X,
                params,
                first_label=first_label,
                second_label=second_label,
                save_path=(
                    os.path.join(gravity_root, comparison_name, "gravity_profiles_comparison_center_west.png")
                    if save_figures else None
                ),
                show=show_figures,
                dpi=save_dpi,
            )
            if save_figures:
                export_gravity_comparison_data(
                    results["TOPO"],
                    gravity_outputs[first_key]["maps"],
                    gravity_outputs[second_key]["maps"],
                    X,
                    Y,
                    params,
                    comparison_name=comparison_name,
                    first_label=first_label,
                    second_label=second_label,
                    save_path=os.path.join(gravity_root, comparison_name, "gravity_comparison_data.npz"),
                )
                summary_path_gravity = os.path.join(gravity_root, comparison_name, "gravity_comparison_summary.txt")
                write_gravity_comparison_summary(
                    gravity_outputs[first_key]["maps"],
                    gravity_outputs[second_key]["maps"],
                    params,
                    summary_path_gravity,
                    first_label=first_label,
                    second_label=second_label,
                )
                print(f"Résumé comparaison gravi : {summary_path_gravity}")

        print()

    if not run_magnetics:
        print("Mode drainage-only : post-traitements magnétiques ignorés.")
        print()
        print("═" * 80)
        print("TERMINÉ")
        print("═" * 80)
        return payload
    
    
    
    
    
    print("═" * 80)
    print("POST-TRAITEMENT : ANOMALIES MAGNÉTIQUES (1 COUCHE)")
    print("═" * 80)
    
    mag_maps, mag_prof, H_map, H_rem_all, top_rem_all, base_mag = \
        compute_magnetic_anomalies_1layer(results['TOPO'], X, Y, params)
    
    print("Calcul terminé - 1 couche")
    print(f"Amplitude max : {np.nanmax(np.abs(mag_maps)):.2f} nT")
    print()

    
    
    

    print("═" * 80)
    print("POST-TRAITEMENT : ANOMALIES MAGNÉTIQUES (CAS 1)")
    print("═" * 80)

    mag_maps_case1, mag_prof_case1, H0_map_case1, H_rem_all_case1, top_rem_all_case1, base_rem_all_case1 = \
        compute_magnetic_anomalies_case1(results['TOPO'], X, Y, params)

    print("Calcul terminé - Cas 1")
    print(f"Amplitude max : {np.nanmax(np.abs(mag_maps_case1)):.2f} nT")
    print()

    
    
    

    print("═" * 80)
    print("POST-TRAITEMENT : ANOMALIES MAGNÉTIQUES (CAS 3)")
    print("═" * 80)

    mag_maps_case3, mag_prof_case3, H0_map_case3, top_rem_all_case3, base_rem_all_case3 = \
        compute_magnetic_anomalies_case3(results['TOPO'], X, Y, params)


    print("Calcul terminé - Cas 3")
    print(f"Amplitude max : {np.nanmax(np.abs(mag_maps_case3)):.2f} nT")
    print()
    
    
    
    
    
    print("═" * 80)
    print("POST-TRAITEMENT : ANOMALIES MAGNÉTIQUES (2 COUCHES)")
    print("═" * 80)
    
    mag_maps_layer, mag_prof_layer, base1, top1_rem_all, base2, top2_rem_all = \
        compute_magnetic_anomalies_2layers(results['TOPO'], X, Y, params)
    
    print("Calcul terminé - 2 couches")
    print(f"Amplitude max : {np.nanmax(np.abs(mag_maps_layer)):.2f} nT")
    print()
    
    
    
    

    print("═" * 80)
    print("POST-TRAITEMENT : ANOMALIES MAGNÉTIQUES (CAS 2 - LENTILLES + LOBE)")
    print("═" * 80)

    (
        mag_maps_case2, mag_prof_case2, H0_map_case2, H_rem_all_case2, top_rem_all_case2, base_rem_all_case2,
        top_lens_rem_all_case2, base_lens_rem_all_case2, top_lobe_rem_all_case2, base_lobe_rem_all_case2
    ) = compute_magnetic_anomalies_case2(results["TOPO"], X, Y, params, return_components=True)

    print("Calcul terminé - Cas 2 (lentilles + lobe)")
    print(f"Amplitude max : {np.nanmax(np.abs(mag_maps_case2)):.2f} nT")
    print()
    
    
    
    
    
    print("═" * 80)
    print("VISUALISATIONS")
    print("═" * 80)

    save_figures = params.get("save_figures", False)
    show_figures = params.get("show_figures", True)
    save_dpi = params.get("save_dpi", 300)
    figures_root = params.get("figures_root", "output_directory/figures")

    case_dirs = {
        "1_couche": os.path.join(figures_root, "1_couche"),
        "2_couches": os.path.join(figures_root, "2_couches"),
        "cas1": os.path.join(figures_root, "cas1"),
        "cas2": os.path.join(figures_root, "cas2"),
        "cas3": os.path.join(figures_root, "cas3"),
    }

    if save_figures:
        for path in case_dirs.values():
            os.makedirs(path, exist_ok=True)

    def fig_path(case_key, filename):
        if not save_figures:
            return None
        return os.path.join(case_dirs[case_key], filename)
    
    
    plot_magnetic_maps(
        mag_maps, X, Y, params,
        title="Anomalies magnétiques - 1 couche",
        save_path=fig_path("1_couche", "mag_maps.png"),
        show=show_figures,
        dpi=save_dpi
    )
    
    plot_magnetic_maps(
        mag_maps_layer, X, Y, params,
        title="Anomalies magnétiques - 2 couches",
        save_path=fig_path("2_couches", "mag_maps.png"),
        show=show_figures,
        dpi=save_dpi
    )
    
    
    plot_magnetic_profiles(
        results['TOPO'], mag_prof, X, Y, params,
        title="Profil magnétique - 1 couche",
        save_path=fig_path("1_couche", "mag_profiles.png"),
        show=show_figures,
        dpi=save_dpi
    )
    
    plot_magnetic_profiles(
        results['TOPO'], mag_prof_layer, X, Y, params,
        title="Profil magnétique - 2 couches",
        save_path=fig_path("2_couches", "mag_profiles.png"),
        show=show_figures,
        dpi=save_dpi
    )
    
    
    plot_layer_evolution_1layer(
        results['TOPO'], top_rem_all, base_mag, X, Y, params,
        save_path=fig_path("1_couche", "layer_evolution.png"),
        show=show_figures,
        dpi=save_dpi
    )
    
    plot_layer_evolution_2layers(
        results['TOPO'], top1_rem_all, base1, top2_rem_all, base2, X, Y, params,
        save_path=fig_path("2_couches", "layer_evolution.png"),
        show=show_figures,
        dpi=save_dpi
    )

    
    plot_magnetic_maps(
        mag_maps_case1, X, Y, params,
        title="Anomalies magnétiques - Cas 1 (couche attachée à la topographie)",
        save_path=fig_path("cas1", "mag_maps.png"),
        show=show_figures,
        dpi=save_dpi
    )

    plot_magnetic_profiles(
        results['TOPO'], mag_prof_case1, X, Y, params,
        title="Profil magnétique - Cas 1 (couche attachée à la topographie)",
        save_path=fig_path("cas1", "mag_profiles.png"),
        show=show_figures,
        dpi=save_dpi
    )

    plot_layer_evolution_1layer(
        results['TOPO'], top_rem_all_case1, base_rem_all_case1, X, Y, params,
        save_path=fig_path("cas1", "layer_evolution.png"),
        show=show_figures,
        dpi=save_dpi
    )
    
    
    
    plot_magnetic_maps(
        mag_maps_case2, X, Y, params,
        title="Anomalies magnétiques - Cas 2 (lentilles + lobe)",
        save_path=fig_path("cas2", "mag_maps.png"),
        show=show_figures,
        dpi=save_dpi
    )
    
    plot_magnetic_profiles(
        results['TOPO'], mag_prof_case2, X, Y, params,
        title="Profil magnétique - Cas 2 (lentilles + lobe)",
        save_path=fig_path("cas2", "mag_profiles.png"),
        show=show_figures,
        dpi=save_dpi
    )
    
    plot_layer_evolution_2layers(
        results['TOPO'],
        top_lens_rem_all_case2, base_lens_rem_all_case2,
        top_lobe_rem_all_case2, base_lobe_rem_all_case2,
        X, Y, params,
        save_path=fig_path("cas2", "layer_evolution.png"),
        show=show_figures,
        dpi=save_dpi
    )
    
    
    plot_magnetic_maps(
        mag_maps_case3, X, Y, params,
        title="Anomalies magnétiques - Cas 3 (couche profonde bombée)",
        save_path=fig_path("cas3", "mag_maps.png"),
        show=show_figures,
        dpi=save_dpi
    )

    plot_magnetic_profiles(
        results['TOPO'], mag_prof_case3, X, Y, params,
        title="Profil magnétique - Cas 3 (couche profonde bombée)",
        save_path=fig_path("cas3", "mag_profiles.png"),
        show=show_figures,
        dpi=save_dpi
    )

    plot_layer_evolution_1layer(
        results['TOPO'], top_rem_all_case3, base_rem_all_case3, X, Y, params,
        save_path=fig_path("cas3", "layer_evolution.png"),
        show=show_figures,
        dpi=save_dpi
    )

    
    
    

    if params.get("use_pyvista_3d", False):
        print()
        print("═" * 80)
        print("VISUALISATION 3D (PyVista)")
        print("═" * 80)

        pyvista_dir = os.path.join(figures_root, "3d")
        if save_figures:
            os.makedirs(pyvista_dir, exist_ok=True)

        plot_3d_pyvista(
            results['TOPO'], mag_maps, X, Y, params,
            save_path=os.path.join(pyvista_dir, "overview_3d.png") if save_figures else None,
            show=show_figures,
        )

    print()
    print("═" * 80)
    print("DONE")
    print("═" * 80)
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crater simulation with uplift and flexure")
    parser.add_argument("--Te", type=float, default=None,
                        help="Elastic thickness in meters, for example 30e3.")
    parser.add_argument("--uplift", type=float, default=None,
                        help="Uplift rate in m/Ma, for example 15.")
    parser.add_argument("--drainage-only", action="store_true",
                        help="Skip magnetic post-processing; gravity can still be enabled with --run-gravity.")
    parser.add_argument("--run-gravity", action="store_true",
                        help="Enable gravity post-processing.")
    parser.add_argument("--gravity-method", type=str, choices=["legacy", "reference", "complete", "both", "all"], default=None,
                        help="Gravity method: legacy, reference, complete, both, or all.")
    parser.add_argument("--gravity-kernel", type=str, choices=["numba", "python"], default=None,
                        help="Gravity numerical backend: numba or python.")
    parser.add_argument("--gravity-spc", type=int, default=None,
                        help="Sedimentary-contrast computation radius in grid nodes.")
    parser.add_argument("--gravity-terrain-spc", type=int, default=None,
                        help="Terrain-correction computation radius in grid nodes.")
    parser.add_argument("--gravity-terrain-density", type=float, default=None,
                        help="Density used for the Bouguer slab and terrain correction (kg/m^3).")
    parser.add_argument("--gravity-density-mode", type=str, choices=["decreasing", "constant", "compaction", "compaction_layered"], default=None,
                        help="Sedimentary density-contrast law: decreasing, constant, compaction, or compaction_layered.")
    parser.add_argument("--gravity-constant-delta-rho", type=float, default=None,
                        help="Constant sedimentary density contrast (kg/m^3) used with --gravity-density-mode=constant.")
    parser.add_argument("--gravity-compaction-lithology", type=str, choices=["sand", "shaly_sand", "shale"], default=None,
                        help="Lithology preset for --gravity-density-mode=compaction or compaction_layered.")
    parser.add_argument("--gravity-compaction-layer-thickness", type=float, default=None,
                        help="Maximum sublayer thickness (m) for --gravity-density-mode=compaction_layered.")
    parser.add_argument("--gravity-host-density", type=float, default=None,
                        help="Host-rock density (kg/m^3) for the compaction law.")
    parser.add_argument("--gravity-grain-density", type=float, default=None,
                        help="Sediment grain density (kg/m^3) for the compaction law.")
    parser.add_argument("--gravity-fluid-density", type=float, default=None,
                        help="Pore-fluid density (kg/m^3) for the compaction law.")
    parser.add_argument("--no-initial-gravity-background", action="store_true",
                        help="Do not add the initial Bouguer grid from the input CSV.")
    parser.add_argument("--no-save-figures", action="store_true",
                        help="Do not write figures to disk.")
    parser.add_argument("--show-figures", action="store_true",
                        help="Display figures on screen.")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable the progress bar.")
    parser.add_argument("--log-progress-interval", type=int, default=None,
                        help="Progress-log interval in years.")
    parser.add_argument("--save-topography-maps", action="store_true",
                        help="Save maps_topography.png for this run.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Initial-noise random seed. Defaults to the value set in params.")
    parser.add_argument("--initial-seed-path", type=str, default=None,
                        help="NPZ seed file used to initialize topography, bedrock, and soil.")
    parser.add_argument("--surface-profile", type=str, default=None,
                        help="Radial CSV profile used to generate the initial topography if no seed file is provided.")
    parser.add_argument("--xy-space", type=float, default=None,
                        help="Horizontal grid spacing in meters.")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root directory for simulation outputs.")
    parser.add_argument("--total-time", type=int, default=None,
                        help="Maximum simulation duration in years.")
    parser.add_argument("--dt", type=int, default=None,
                        help="Numerical time step in years.")
    parser.add_argument("--dt-flex", type=int, default=None,
                        help="Interval between flexure calculations, in years.")
    parser.add_argument("--topography-snapshot-times-ma", type=float, nargs="+", default=None,
                        help="Exact topography snapshot times to keep/display, in Ma.")
    parser.add_argument("--boundary-mode", type=str,
                        choices=["all_open", "south_open", "southwest_corner", "southeast_corner"],
                        default=None,
                        help="Boundary-condition mode for drainage.")
    parser.add_argument("--stop-on-crater-breach", action="store_true",
                        help="Stop the simulation once drainage crosses the crater rim.")
    parser.add_argument("--crater-radius-m", type=float, default=None,
                        help="Crater radius (m) used for figures and geometric markers.")
    parser.add_argument("--crater-mask-radius-m", type=float, default=None,
                        help="Crater-mask radius (m) used for the drainage indicator and optional stop condition.")
    parser.add_argument("--event-check-interval", type=int, default=None,
                        help="Event-check interval in years.")
    parser.add_argument("--crater-event-min-area-km2", type=float, default=None,
                        help="Minimum drained area (km^2) required to validate a rim-crossing event.")
    parser.add_argument("--no-direct-uplift-flexure-load", action="store_true",
                        help="Remove the imposed tectonic-uplift component from the flexural load.")
    args = parser.parse_args()
    main(
        Te=args.Te,
        uplift_rate_m_per_Ma=args.uplift,
        run_magnetics=not args.drainage_only,
        run_gravity=(True if args.run_gravity else None),
        gravity_method=args.gravity_method,
        gravity_kernel=args.gravity_kernel,
        gravity_spc=args.gravity_spc,
        gravity_terrain_spc=args.gravity_terrain_spc,
        gravity_terrain_density=args.gravity_terrain_density,
        gravity_density_mode=args.gravity_density_mode,
        gravity_constant_delta_rho=args.gravity_constant_delta_rho,
        gravity_compaction_lithology=args.gravity_compaction_lithology,
        gravity_compaction_layer_thickness=args.gravity_compaction_layer_thickness,
        gravity_host_density=args.gravity_host_density,
        gravity_grain_density=args.gravity_grain_density,
        gravity_fluid_density=args.gravity_fluid_density,
        gravity_add_initial_background=(False if args.no_initial_gravity_background else None),
        save_figures=not args.no_save_figures,
        show_figures=args.show_figures,
        show_progress=not args.no_progress,
        log_progress_interval_yr=args.log_progress_interval,
        save_topography_maps=args.save_topography_maps,
        random_seed=args.seed,
        initial_seed_path=args.initial_seed_path,
        surface_profile_path=args.surface_profile,
        xy_space_m=args.xy_space,
        output_root=args.output_root,
        total_time_yr=args.total_time,
        time_step_yr=args.dt,
        flexure_time_step_yr=args.dt_flex,
        topography_snapshot_times_Ma=args.topography_snapshot_times_ma,
        boundary_mode=args.boundary_mode,
        stop_on_crater_breach=args.stop_on_crater_breach,
        crater_radius_m=args.crater_radius_m,
        crater_mask_radius_m=args.crater_mask_radius_m,
        event_check_interval_yr=args.event_check_interval,
        crater_event_min_area_km2=args.crater_event_min_area_km2,
        flexure_include_direct_uplift_load=(False if args.no_direct_uplift_flexure_load else None),
    )

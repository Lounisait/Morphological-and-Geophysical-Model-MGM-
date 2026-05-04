#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Magnetic anomaly workflows used by MGM."""

import numpy as np
from mag_crater_dipole_harmonized import (
    assign_magnetization_radial,
    total_field_anomaly_nT,
    upward_continuation_2d,
    compute_volumes_from_H,
    compute_B_prism_auto,
)





def compute_mag_evolution_column_prism(
    TOPO, X, Y, xy_space, nb_step,
    
    z_comp_down=500.0,
    crater_radius=15000.0,
    thickness_mode="taper_to_zero",
    taper_power=1.0,
    M_center=0.5,
    M_background=0.02,
    r_scale_frac=0.35,
    z_scale=1500.0,
    I_deg=90.0,
    D_deg=0.0,
    
    obs_mode="flat",
    z_obs_up=1000.0,
    h_above_topo_up=0.0,
    
    erosion_mode="linear",
    
    return_layer=True,
    force_negative=True,
    use_numba=True,
    upward_dz=0.0,

):
    """MGM internal routine."""
    
    topo0 = TOPO[0]
    nby, nbx = topo0.shape
    
    x_obs = X[0, :]
    y_obs = Y[:, 0]
    dx = float(xy_space)
    dy = float(xy_space)
    
    obsX, obsY = np.meshgrid(x_obs, y_obs, indexing="xy")
    CX, CY = np.meshgrid(x_obs, y_obs, indexing="xy")
    
    xc = float(x_obs.mean())
    yc = float(y_obs.mean())
    
    
    
    
    
    r = np.sqrt((CX - xc)**2 + (CY - yc)**2)
    Hmax = float(z_comp_down)
    
    if thickness_mode == "constant":
        H_map = np.full_like(topo0, Hmax, dtype=np.float64)
    elif thickness_mode == "taper_to_zero":
        t = 1.0 - r / float(crater_radius)
        t = np.clip(t, 0.0, 1.0)
        t = t**float(taper_power)
        H_map = Hmax * t
    else:
        raise ValueError("thickness_mode: 'constant' ou 'taper_to_zero'")
    
    z_top_layer = topo0
    z_base_layer = topo0 + H_map
    z_center_layer = topo0 + 0.5 * H_map
    
    
    
    
    
    centers = np.column_stack([
        CX.ravel(),
        CY.ravel(),
        z_center_layer.ravel(),
    ]).astype(np.float64)
    
    crater_center = (xc, yc, float(np.mean(z_center_layer)))
    
    M_vecs0 = assign_magnetization_radial(
        centers=centers,
        shape=(nbx, nby, 1),
        dims=(dx, dy, Hmax),
        crater_center=crater_center,
        crater_radius=crater_radius,
        M_center=M_center,
        M_background=M_background,
        r_scale_frac=r_scale_frac,
        z_scale=z_scale,
        I_deg=I_deg,
        D_deg=D_deg,
    ).astype(np.float64)

    
    
    
    
    
    mag_maps = np.zeros((nb_step, nby, nbx), dtype=np.float64)
    mag_prof = np.zeros((nb_step, nbx), dtype=np.float64)
    mid_y = nby // 2
    
    if return_layer:
        H_rem_all = np.zeros((nb_step, nby, nbx), dtype=np.float64)
        top_rem_all = np.zeros((nb_step, nby, nbx), dtype=np.float64)
    
    topo_init = topo0.copy()
    E_max = np.zeros_like(topo0, dtype=np.float64)
    
    for istep in range(nb_step):
        topo = TOPO[istep].astype(np.float64)
        
        
        if obs_mode == "topography":
            z_obs = topo
        elif obs_mode == "flat":
            z_obs = np.full_like(topo, -float(z_obs_up), dtype=np.float64)
        elif obs_mode == "topography_plus_h":
            z_obs = topo - float(h_above_topo_up)
        else:
            raise ValueError("obs_mode invalide")
        
        obsZ = z_obs
        
        
        erosion_inst = np.maximum(topo - topo_init, 0.0)
        E_max = np.maximum(E_max, erosion_inst)
        erosion = E_max
        
        
        if erosion_mode == "linear":
            H_rem = np.clip(H_map - erosion, 0.0, H_map)
        elif erosion_mode == "binary":
            H_rem = H_map.copy()
            H_rem[erosion >= H_map] = 0.0
        else:
            raise ValueError("erosion_mode: 'linear' ou 'binary'")
        
        if return_layer:
            
            
            z_top_rem = z_base_layer - H_rem
            top_rem_all[istep] = z_top_rem
            H_rem_all[istep] = H_rem
        
        
        
        

        volumes_t = compute_volumes_from_H(dx, dy, H_rem)

        if np.all(H_rem == 0):
            continue

        
        
        
        z_center_rem = z_base_layer - 0.5 * H_rem
        centers_t = np.column_stack([
            CX.ravel(),
            CY.ravel(),
            z_center_rem.ravel(),
        ]).astype(np.float64)

        
        Bx, By, Bz = compute_B_prism_auto(
            obsX, obsY, obsZ,
            centers_t, M_vecs0, volumes_t,
            dx, dy, use_numba=use_numba
        )
        
        dT = total_field_anomaly_nT(Bx, By, Bz, I_deg=I_deg, D_deg=D_deg)
        
        if force_negative:
            dT = -np.abs(dT)

        if upward_dz > 0:
            dT = upward_continuation_2d(dT, dx, dy, upward_dz)

        mag_maps[istep] = dT
        mag_prof[istep] = dT[mid_y]

    if return_layer:
        return mag_maps, mag_prof, H_map, H_rem_all, top_rem_all, z_base_layer
    else:
        return mag_maps, mag_prof






def compute_mag_evolution_column_prism_2layers(
    TOPO, X, Y, xy_space, nb_step,
    
    z_comp_down1=50.0,
    thickness_mode1="taper_to_zero",
    taper_power1=1.0,
    crater_radius1=15000.0,
    M_center1=0.5,
    M_background1=0.02,
    r_scale_frac1=0.35,
    z_scale1=1500.0,
    
    use_layer2=True,
    z_comp_down2=200.0,
    thickness_mode2="taper_to_zero",
    taper_power2=1.0,
    crater_radius2=15000.0,
    M_center2=0.1,
    M_background2=0.01,
    r_scale_frac2=0.35,
    z_scale2=1500.0,
    
    I_deg=90.0,
    D_deg=0.0,
    
    obs_mode="flat",
    z_obs_up=1000.0,
    h_above_topo_up=0.0,
    
    erosion_mode="linear",
    
    force_negative=True,
    return_layers=True,
    use_numba=True,
    upward_dz=0.0,

):
    """MGM internal routine."""
    
    topo0 = TOPO[0]
    nby, nbx = topo0.shape
    
    x_obs = X[0, :]
    y_obs = Y[:, 0]
    dx = float(xy_space)
    dy = float(xy_space)
    
    obsX, obsY = np.meshgrid(x_obs, y_obs, indexing="xy")
    CX, CY = np.meshgrid(x_obs, y_obs, indexing="xy")
    
    xc = float(x_obs.mean())
    yc = float(y_obs.mean())
    r = np.sqrt((CX - xc)**2 + (CY - yc)**2)
    
    
    
    
    
    def build_H_map(Hmax, thickness_mode, taper_power, crater_radius_local):
        Hmax = float(Hmax)
        if thickness_mode == "constant":
            return np.full_like(topo0, Hmax, dtype=np.float64)
        elif thickness_mode == "taper_to_zero":
            t = 1.0 - (r / float(crater_radius_local))
            t = np.clip(t, 0.0, 1.0)
            t = t**float(taper_power)
            return Hmax * t
        else:
            raise ValueError("thickness_mode invalide")
    
    
    
    
    
    H1 = build_H_map(z_comp_down1, thickness_mode1, taper_power1, crater_radius1)
    H2 = build_H_map(z_comp_down2, thickness_mode2, taper_power2, crater_radius2) if use_layer2 else np.zeros_like(H1)
    
    z_top_layer1 = topo0
    z_base_layer1 = topo0 + H1
    z_top_layer2 = z_base_layer1
    z_base_layer2 = z_top_layer2 + H2 if use_layer2 else None
    
    
    
    
    
    z_center1 = topo0 + 0.5 * H1
    centers1 = np.column_stack([
        CX.ravel(), 
        CY.ravel(), 
        z_center1.ravel()
    ]).astype(np.float64)
    
    M1 = assign_magnetization_radial(
        centers1, (nbx, nby, 1), (dx, dy, z_comp_down1),
        (xc, yc, float(np.mean(z_center1))), crater_radius1,
        M_center1, M_background1, r_scale_frac1, z_scale1,
        I_deg, D_deg
    ).astype(np.float64)
    
    
    
    
    
    if use_layer2:
        z_center2 = z_top_layer2 + 0.5 * H2
        centers2 = np.column_stack([
            CX.ravel(), 
            CY.ravel(), 
            z_center2.ravel()
        ]).astype(np.float64)
        
        M2 = assign_magnetization_radial(
            centers2, (nbx, nby, 1), (dx, dy, z_comp_down2),
            (xc, yc, float(np.mean(z_center2))), crater_radius2,
            M_center2, M_background2, r_scale_frac2, z_scale2,
            I_deg, D_deg
        ).astype(np.float64)
        
    else:
        centers2, M2= None, None
    
    
    
    
    
    mag_maps = np.zeros((nb_step, nby, nbx), dtype=np.float64)
    mag_prof = np.zeros((nb_step, nbx), dtype=np.float64)
    mid_y = nby // 2
    
    if return_layers:
        top1_rem_all = np.zeros((nb_step, nby, nbx), dtype=np.float64)
        top2_rem_all = np.zeros((nb_step, nby, nbx), dtype=np.float64) if use_layer2 else None
    
    topo_init = topo0.copy()
    E_max = np.zeros_like(topo0, dtype=np.float64)
    
    
    
    
    
    for istep in range(nb_step):
        topo = TOPO[istep].astype(np.float64)
        
        
        if obs_mode == "topography":
            z_obs = topo
        elif obs_mode == "flat":
            z_obs = np.full_like(topo, -float(z_obs_up), dtype=np.float64)
        elif obs_mode == "topography_plus_h":
            z_obs = topo - float(h_above_topo_up)
        else:
            raise ValueError("obs_mode invalide")
        
        obsZ = z_obs
        
        
        erosion_inst = np.maximum(topo - topo_init, 0.0)
        E_max = np.maximum(E_max, erosion_inst)
        erosion = E_max
        
        
        
        
        
        if erosion_mode == "linear":
            H1_rem = np.clip(H1 - erosion, 0.0, H1)
        elif erosion_mode == "binary":
            H1_rem = H1.copy()
            H1_rem[erosion >= H1] = 0.0
        else:
            raise ValueError("erosion_mode invalide")
        
        if return_layers:
            
            
            z_top1_rem = z_base_layer1 - H1_rem
            top1_rem_all[istep] = z_top1_rem
        
        volumes1_t = compute_volumes_from_H(dx, dy, H1_rem)
        
        
        
        
        
        if use_layer2:
            erosion_into_2 = np.maximum(erosion - H1, 0.0)
            
            if erosion_mode == "linear":
                H2_rem = np.clip(H2 - erosion_into_2, 0.0, H2)
            elif erosion_mode == "binary":
                H2_rem = H2.copy()
                H2_rem[erosion >= (H1 + H2)] = 0.0
            
            if return_layers:
                
                
                z_top2_rem = z_base_layer2 - H2_rem
                top2_rem_all[istep] = z_top2_rem
            
            volumes2_t = compute_volumes_from_H(dx, dy, H2_rem)
        else:
            volumes2_t = None
        
        
        
        

        
        z_center1_rem = z_base_layer1 - 0.5 * H1_rem
        centers1_t = np.column_stack([
            CX.ravel(), CY.ravel(), z_center1_rem.ravel()
        ]).astype(np.float64)

        Bx1, By1, Bz1 = compute_B_prism_auto(
            obsX, obsY, obsZ,
            centers1_t, M1, volumes1_t,
            dx, dy, use_numba=use_numba
        )

        if use_layer2:
            
            z_center2_rem = z_base_layer2 - 0.5 * H2_rem
            centers2_t = np.column_stack([
                CX.ravel(), CY.ravel(), z_center2_rem.ravel()
            ]).astype(np.float64)

            Bx2, By2, Bz2 = compute_B_prism_auto(
                obsX, obsY, obsZ,
                centers2_t, M2, volumes2_t,
                dx, dy, use_numba=use_numba
            )
            Bx, By, Bz = Bx1 + Bx2, By1 + By2, Bz1 + Bz2
        else:
            Bx, By, Bz = Bx1, By1, Bz1
        
        dT = total_field_anomaly_nT(Bx, By, Bz, I_deg=I_deg, D_deg=D_deg)
        
        if force_negative:
            dT = -np.abs(dT)

        if upward_dz > 0:
            dT = upward_continuation_2d(dT, dx, dy, upward_dz)

        mag_maps[istep] = dT
        mag_prof[istep] = dT[mid_y, :]

    if return_layers:
        return mag_maps, mag_prof, z_base_layer1, top1_rem_all, z_base_layer2, top2_rem_all
    else:
        return mag_maps, mag_prof
    
def compute_mag_evolution_column_prism_case1(
    TOPO, X, Y, xy_space, nb_step,
    
    z_comp_down=500.0,
    crater_radius=15000.0,
    base_mode="flat_at_center",  
    z_base_abs_down=None,         
    
    M_center=0.5,
    M_background=0.02,
    r_scale_frac=0.35,
    z_scale=1500.0,
    I_deg=90.0,
    D_deg=0.0,
    
    obs_mode="flat",          
    z_obs_up=1000.0,
    h_above_topo_up=0.0,
    
    erosion_mode="linear",
    
    return_layer=True,
    force_negative=True,
    use_numba=True,
    upward_dz=0.0,

):
    """MGM internal routine."""

    topo0 = TOPO[0].astype(np.float64)
    nby, nbx = topo0.shape

    x_obs = X[0, :]
    y_obs = Y[:, 0]
    dx = float(xy_space)
    dy = float(xy_space)

    obsX, obsY = np.meshgrid(x_obs, y_obs, indexing="xy")
    CX, CY = np.meshgrid(x_obs, y_obs, indexing="xy")

    xc = float(x_obs.mean())
    yc = float(y_obs.mean())
    r = np.sqrt((CX - xc) ** 2 + (CY - yc) ** 2)
    mask_disk = (r <= float(crater_radius))

    
    
    
    Hmax = float(z_comp_down)

    if base_mode == "flat_at_center":
        ic = int(np.argmin(np.abs(x_obs - xc)))
        jc = int(np.argmin(np.abs(y_obs - yc)))
        z_base0 = float(topo0[jc, ic] + Hmax)
    elif base_mode == "flat_at_mean":
        z_base0 = float(np.mean(topo0) + Hmax)
    elif base_mode == "absolute":
        if z_base_abs_down is None:
            raise ValueError("base_mode='absolute' requiert z_base_abs_down (z-down)")
        z_base0 = float(z_base_abs_down)
    else:
        raise ValueError("base_mode invalide: 'flat_at_center' | 'flat_at_mean' | 'absolute'")

    z_base0_plane = np.full_like(topo0, z_base0, dtype=np.float64)

    
    H0_map = np.zeros_like(topo0, dtype=np.float64)
    H0_map[mask_disk] = np.maximum(z_base0_plane[mask_disk] - topo0[mask_disk], 0.0)

    
    z_center0 = topo0 + 0.5 * H0_map
    centers0 = np.column_stack([
        CX.ravel(),
        CY.ravel(),
        z_center0.ravel(),
    ]).astype(np.float64)

    crater_center = (xc, yc, float(np.mean(z_center0)))

    M_vecs0 = assign_magnetization_radial(
        centers=centers0,
        shape=(nbx, nby, 1),
        dims=(dx, dy, Hmax),
        crater_center=crater_center,
        crater_radius=crater_radius,
        M_center=M_center,
        M_background=M_background,
        r_scale_frac=r_scale_frac,
        z_scale=z_scale,
        I_deg=I_deg,
        D_deg=D_deg,
    ).astype(np.float64)

    
    
    
    mag_maps = np.zeros((nb_step, nby, nbx), dtype=np.float64)
    mag_prof = np.zeros((nb_step, nbx), dtype=np.float64)
    mid_y = nby // 2

    if return_layer:
        H_rem_all = np.zeros((nb_step, nby, nbx), dtype=np.float64)
        top_rem_all = np.full((nb_step, nby, nbx), np.nan, dtype=np.float64)
        base_rem_all = np.full((nb_step, nby, nbx), np.nan, dtype=np.float64)

    E_max = np.zeros_like(topo0, dtype=np.float64)

    for istep in range(nb_step):
        topo = TOPO[istep].astype(np.float64)

        
        if obs_mode == "topography":
            obsZ = topo
        elif obs_mode == "flat":
            obsZ = np.full_like(topo, -float(z_obs_up), dtype=np.float64)
        elif obs_mode == "topography_plus_h":
            obsZ = topo - float(h_above_topo_up)
        else:
            raise ValueError("obs_mode invalide")

        
        erosion_inst = np.maximum(topo - topo0, 0.0)
        E_max = np.maximum(E_max, erosion_inst)
        erosion = E_max

        
        if erosion_mode == "linear":
            H_rem = np.clip(H0_map - erosion, 0.0, H0_map)
        elif erosion_mode == "binary":
            H_rem = H0_map.copy()
            H_rem[erosion >= H0_map] = 0.0
        else:
            raise ValueError("erosion_mode: 'linear' ou 'binary'")

        
        H_rem[~mask_disk] = 0.0

        if np.all(H_rem == 0.0):
            if return_layer:
                H_rem_all[istep] = H_rem
            continue

        
        z_surface_min = topo0 + erosion

        
        present = H_rem > 0.0
        top_mag = np.full_like(topo, np.nan, dtype=np.float64)
        base_mag = np.full_like(topo, np.nan, dtype=np.float64)
        top_mag[present] = z_surface_min[present]
        base_mag[present] = z_surface_min[present] + H_rem[present]

        if return_layer:
            H_rem_all[istep] = H_rem
            top_rem_all[istep] = top_mag
            base_rem_all[istep] = base_mag

        
        centers_t = np.column_stack([
            CX.ravel(),
            CY.ravel(),
            (z_surface_min + 0.5 * H_rem).ravel(),
        ]).astype(np.float64)

        volumes_t = compute_volumes_from_H(dx, dy, H_rem)

        Bx, By, Bz = compute_B_prism_auto(
            obsX, obsY, obsZ,
            centers_t, M_vecs0, volumes_t,
            dx, dy, use_numba=use_numba
        )

        dT = total_field_anomaly_nT(Bx, By, Bz, I_deg=I_deg, D_deg=D_deg)
        if force_negative:
            dT = -np.abs(dT)

        if upward_dz > 0:
            dT = upward_continuation_2d(dT, dx, dy, upward_dz)

        mag_maps[istep] = dT
        mag_prof[istep] = dT[mid_y]

    if return_layer:
        return mag_maps, mag_prof, H0_map, H_rem_all, top_rem_all, base_rem_all
    else:
        return mag_maps, mag_prof

def compute_mag_simple_flat_layer(
    TOPO, X, Y, xy_space, nb_step,
    z_top_layer_down=3500.0,     
    thickness_down=500.0,        
    crater_radius=None,          
    
    M_center=0.5,
    M_background=0.02,
    r_scale_frac=0.35,
    z_scale=1500.0,
    I_deg=90.0,
    D_deg=0.0,
    
    obs_mode="flat",
    z_obs_up=1000.0,
    h_above_topo_up=0.0,
    
    return_layer=True,
    force_negative=True,
    use_numba=True,
    upward_dz=0.0,

):
    """MGM internal routine."""

    topo0 = TOPO[0].astype(np.float64)
    nby, nbx = topo0.shape

    x_obs = X[0, :]
    y_obs = Y[:, 0]
    dx = float(xy_space)
    dy = float(xy_space)

    obsX, obsY = np.meshgrid(x_obs, y_obs, indexing="xy")
    CX, CY = np.meshgrid(x_obs, y_obs, indexing="xy")

    xc = float(x_obs.mean())
    yc = float(y_obs.mean())
    r = np.sqrt((CX - xc)**2 + (CY - yc)**2)

    if crater_radius is None:
        in_mask = np.ones_like(topo0, dtype=bool)
        crater_radius_for_M = np.nanmax(r)
    else:
        in_mask = (r <= float(crater_radius))
        crater_radius_for_M = float(crater_radius)

    
    z_top0 = np.full_like(topo0, z_top_layer_down)
    z_base0 = z_top0 + thickness_down
    H0_map = np.zeros_like(topo0)
    H0_map[in_mask] = thickness_down

    z_center0 = z_top0 + 0.5 * thickness_down

    centers = np.column_stack([
        CX.ravel(),
        CY.ravel(),
        z_center0.ravel()
    ]).astype(np.float64)

    crater_center = (xc, yc, float(z_center0.mean()))

    M_vecs0 = assign_magnetization_radial(
        centers=centers,
        shape=(nbx, nby, 1),
        dims=(dx, dy, thickness_down),
        crater_center=crater_center,
        crater_radius=crater_radius_for_M,
        M_center=M_center,
        M_background=M_background,
        r_scale_frac=r_scale_frac,
        z_scale=z_scale,
        I_deg=I_deg,
        D_deg=D_deg,
    ).astype(np.float64)

    mag_maps = np.zeros((nb_step, nby, nbx))
    mag_prof = np.zeros((nb_step, nbx))
    mid_y = nby // 2

    if return_layer:
        top_rem_all = np.full((nb_step, nby, nbx), np.nan)
        base_rem_all = np.full((nb_step, nby, nbx), np.nan)

    for istep in range(nb_step):
        topo = TOPO[istep].astype(np.float64)

        if obs_mode == "topography":
            obsZ = topo
        elif obs_mode == "flat":
            obsZ = np.full_like(topo, -float(z_obs_up))
        elif obs_mode == "topography_plus_h":
            obsZ = topo - float(h_above_topo_up)
        else:
            raise ValueError("obs_mode invalide")

        if return_layer:
            top_rem_all[istep][in_mask] = z_top0[in_mask]
            base_rem_all[istep][in_mask] = z_base0[in_mask]

        volumes = compute_volumes_from_H(dx, dy, H0_map)

        Bx, By, Bz = compute_B_prism_auto(
            obsX, obsY, obsZ,
            centers, M_vecs0, volumes,
            dx, dy, use_numba=use_numba
        )

        dT = total_field_anomaly_nT(Bx, By, Bz, I_deg=I_deg, D_deg=D_deg)
        if force_negative:
            dT = -np.abs(dT)

        if upward_dz > 0:
            dT = upward_continuation_2d(dT, dx, dy, upward_dz)

        mag_maps[istep] = dT
        mag_prof[istep] = dT[mid_y]

    if return_layer:
        return mag_maps, mag_prof, H0_map, top_rem_all, base_rem_all
    return mag_maps, mag_prof

def compute_mag_evolution_column_prism_case2_lenses_lobe(
    TOPO, X, Y, xy_space, nb_step,
    
    crater_radius=7500.0,

    
    lens_offset_frac=0.45,              
    lens_offset_km=None,                
    lens_offset_range_km=None,          
    lens_sigma_frac=0.18,
    lens_sigma_km=None,                 
    lens_thickness_down=400.0,

    
    lobe_thickness_down=250.0,
    lobe_inner_hole_frac=0.0,
    lobe_taper_power=1.0,

    
    M_center_lens=0.6,
    M_background_lens=0.02,
    r_scale_frac_lens=0.35,
    z_scale_lens=1500.0,

    
    M_center_lobe=0.35,
    M_background_lobe=0.01,
    r_scale_frac_lobe=0.50,
    z_scale_lobe=2000.0,

    
    I_deg=90.0,
    D_deg=0.0,

    
    obs_mode="flat",
    z_obs_up=1000.0,
    h_above_topo_up=0.0,

    
    erosion_mode="linear",

    
    return_layer=True,
    return_components=False,
    force_negative=True,
    use_numba=True,
    upward_dz=0.0,

):
    """MGM internal routine."""

    topo0 = TOPO[0].astype(np.float64)
    nby, nbx = topo0.shape

    x_obs = X[0, :]
    y_obs = Y[:, 0]
    dx = float(xy_space)
    dy = float(xy_space)

    obsX, obsY = np.meshgrid(x_obs, y_obs, indexing="xy")
    CX, CY = np.meshgrid(x_obs, y_obs, indexing="xy")

    xc = float(x_obs.mean())
    yc = float(y_obs.mean())

    r = np.sqrt((CX - xc) ** 2 + (CY - yc) ** 2)
    crater_radius = float(crater_radius)
    in_mask = (r <= crater_radius)

    
    if lens_offset_range_km is not None:
        a, b = float(lens_offset_range_km[0]), float(lens_offset_range_km[1])
        lens_offset_m = 0.5 * (a + b) * 1000.0
    elif lens_offset_km is not None:
        lens_offset_m = float(lens_offset_km) * 1000.0
    else:
        lens_offset_m = float(lens_offset_frac) * crater_radius

    
    if lens_sigma_km is not None:
        lens_sigma = float(lens_sigma_km) * 1000.0
    else:
        lens_sigma = float(lens_sigma_frac) * crater_radius

    
    x1, y1 = xc - lens_offset_m, yc
    x2, y2 = xc + lens_offset_m, yc
    r1 = np.sqrt((CX - x1) ** 2 + (CY - y1) ** 2)
    r2 = np.sqrt((CX - x2) ** 2 + (CY - y2) ** 2)

    
    
    
    H0_lens = np.zeros_like(topo0, dtype=np.float64)
    H0_lens[in_mask] = float(lens_thickness_down) * (
        np.exp(-0.5 * (r1[in_mask] / lens_sigma) ** 2) +
        np.exp(-0.5 * (r2[in_mask] / lens_sigma) ** 2)
    )

    
    t = 1.0 - (r / crater_radius)
    t = np.clip(t, 0.0, 1.0)
    t = t ** float(lobe_taper_power)

    H0_lobe = np.zeros_like(topo0, dtype=np.float64)
    H0_lobe[in_mask] = float(lobe_thickness_down) * t[in_mask]

    if float(lobe_inner_hole_frac) > 0.0:
        hole_r = float(lobe_inner_hole_frac) * crater_radius
        H0_lobe[r < hole_r] = 0.0

    
    H0_total = H0_lens + H0_lobe

    
    
    
    
    z_center_lens0 = topo0 + 0.5 * H0_lens
    z_center_lobe0 = topo0 + 0.5 * H0_lobe

    centers_lens0 = np.column_stack([CX.ravel(), CY.ravel(), z_center_lens0.ravel()]).astype(np.float64)
    centers_lobe0 = np.column_stack([CX.ravel(), CY.ravel(), z_center_lobe0.ravel()]).astype(np.float64)

    crater_center = (xc, yc, float(np.mean(topo0)))

    Hmax_lens = float(max(1.0, lens_thickness_down))
    Hmax_lobe = float(max(1.0, lobe_thickness_down))

    M_lens = assign_magnetization_radial(
        centers=centers_lens0,
        shape=(nbx, nby, 1),
        dims=(dx, dy, Hmax_lens),
        crater_center=crater_center,
        crater_radius=crater_radius,
        M_center=M_center_lens,
        M_background=M_background_lens,
        r_scale_frac=r_scale_frac_lens,
        z_scale=z_scale_lens,
        I_deg=I_deg,
        D_deg=D_deg,
    ).astype(np.float64)

    M_lobe = assign_magnetization_radial(
        centers=centers_lobe0,
        shape=(nbx, nby, 1),
        dims=(dx, dy, Hmax_lobe),
        crater_center=crater_center,
        crater_radius=crater_radius,
        M_center=M_center_lobe,
        M_background=M_background_lobe,
        r_scale_frac=r_scale_frac_lobe,
        z_scale=z_scale_lobe,
        I_deg=I_deg,
        D_deg=D_deg,
    ).astype(np.float64)

    
    
    
    mag_maps = np.zeros((nb_step, nby, nbx), dtype=np.float64)
    mag_prof = np.zeros((nb_step, nbx), dtype=np.float64)
    mid_y = nby // 2

    if return_components and (not return_layer):
        raise ValueError("return_components=True nécessite return_layer=True.")

    if return_layer:
        H_rem_all = np.zeros((nb_step, nby, nbx), dtype=np.float64)
        top_rem_all = np.full((nb_step, nby, nbx), np.nan, dtype=np.float64)
        base_rem_all = np.full((nb_step, nby, nbx), np.nan, dtype=np.float64)
        if return_components:
            top_lens_rem_all = np.full((nb_step, nby, nbx), np.nan, dtype=np.float64)
            base_lens_rem_all = np.full((nb_step, nby, nbx), np.nan, dtype=np.float64)
            top_lobe_rem_all = np.full((nb_step, nby, nbx), np.nan, dtype=np.float64)
            base_lobe_rem_all = np.full((nb_step, nby, nbx), np.nan, dtype=np.float64)

    E_max = np.zeros_like(topo0, dtype=np.float64)

    for istep in range(nb_step):
        topo = TOPO[istep].astype(np.float64)

        
        if obs_mode == "topography":
            obsZ = topo
        elif obs_mode == "flat":
            obsZ = np.full_like(topo, -float(z_obs_up), dtype=np.float64)
        elif obs_mode == "topography_plus_h":
            obsZ = topo - float(h_above_topo_up)
        else:
            raise ValueError("obs_mode invalide")

        
        erosion_inst = np.maximum(topo - topo0, 0.0)
        E_max = np.maximum(E_max, erosion_inst)
        erosion = E_max

        
        if erosion_mode == "linear":
            H_lens = np.clip(H0_lens - erosion, 0.0, H0_lens)
            H_lobe = np.clip(H0_lobe - erosion, 0.0, H0_lobe)
        elif erosion_mode == "binary":
            H_lens = H0_lens.copy()
            H_lens[erosion >= H0_lens] = 0.0
            H_lobe = H0_lobe.copy()
            H_lobe[erosion >= H0_lobe] = 0.0
        else:
            raise ValueError("erosion_mode: 'linear' ou 'binary'")

        
        H_lens[~in_mask] = 0.0
        H_lobe[~in_mask] = 0.0

        H_total = H_lens + H_lobe
        alive = (H_total > 0.0) & in_mask

        
        z_surface_min = topo0 + erosion

        if return_layer:
            H_rem_all[istep] = H_total
            top_rem_all[istep][alive] = z_surface_min[alive]
            base_rem_all[istep][alive] = z_surface_min[alive] + H_total[alive]
            if return_components:
                alive_lens = (H_lens > 0.0) & in_mask
                alive_lobe = (H_lobe > 0.0) & in_mask
                top_lens_rem_all[istep][alive_lens] = z_surface_min[alive_lens]
                base_lens_rem_all[istep][alive_lens] = z_surface_min[alive_lens] + H_lens[alive_lens]
                top_lobe_rem_all[istep][alive_lobe] = z_surface_min[alive_lobe]
                base_lobe_rem_all[istep][alive_lobe] = z_surface_min[alive_lobe] + H_lobe[alive_lobe]

        if np.all(H_total == 0.0):
            continue

        
        centers_lens_t = np.column_stack([CX.ravel(), CY.ravel(), (z_surface_min + 0.5 * H_lens).ravel()]).astype(np.float64)
        centers_lobe_t = np.column_stack([CX.ravel(), CY.ravel(), (z_surface_min + 0.5 * H_lobe).ravel()]).astype(np.float64)

        
        vol_lens = compute_volumes_from_H(dx, dy, H_lens)
        vol_lobe = compute_volumes_from_H(dx, dy, H_lobe)

        
        Bx1, By1, Bz1 = compute_B_prism_auto(
            obsX, obsY, obsZ,
            centers_lens_t, M_lens, vol_lens,
            dx, dy, use_numba=use_numba,
        )

        Bx2, By2, Bz2 = compute_B_prism_auto(
            obsX, obsY, obsZ,
            centers_lobe_t, M_lobe, vol_lobe,
            dx, dy, use_numba=use_numba,
        )

        Bx, By, Bz = Bx1 + Bx2, By1 + By2, Bz1 + Bz2

        dT = total_field_anomaly_nT(Bx, By, Bz, I_deg=I_deg, D_deg=D_deg)
        if force_negative:
            dT = -np.abs(dT)

        if upward_dz > 0:
            dT = upward_continuation_2d(dT, dx, dy, upward_dz)

        mag_maps[istep] = dT
        mag_prof[istep] = dT[mid_y, :]

    if return_layer:
        if return_components:
            return (
                mag_maps, mag_prof, H0_total, H_rem_all, top_rem_all, base_rem_all,
                top_lens_rem_all, base_lens_rem_all, top_lobe_rem_all, base_lobe_rem_all
            )
        return mag_maps, mag_prof, H0_total, H_rem_all, top_rem_all, base_rem_all
    return mag_maps, mag_prof

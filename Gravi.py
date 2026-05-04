#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gravity anomaly routines used by MGM."""

import numpy as np
from sub_routine import anomaly
from tqdm import tqdm




try:
    from numba import njit

    @njit
    def _anomaly_numba(x0, y0, z0, x1, y1, z1, x2, y2, z2, rho):
        gamma = 6.673e-11
        twopi = 2.0 * np.pi
        si2mg = 1e5

        xx0 = x0 - x1
        xx1 = x0 - x2
        yy0 = y0 - y1
        yy1 = y0 - y2
        zz0 = z0 - z1
        zz1 = z0 - z2

        xv = (xx0, xx1)
        yv = (yy0, yy1)
        zv = (zz0, zz1)
        isign = (-1.0, 1.0)

        som = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    rijk = np.sqrt(xv[i]**2 + yv[j]**2 + zv[k]**2)
                    ijk = isign[i] * isign[j] * isign[k]
                    arg1 = np.arctan2(xv[i] * yv[j], zv[k] * rijk)
                    if arg1 < 0.0:
                        arg1 = arg1 + twopi
                    arg2 = np.log(rijk + yv[j])
                    arg3 = np.log(rijk + xv[i])
                    val = (zv[k] * arg1 - xv[i] * arg2 - yv[j] * arg3) * ijk
                    if np.isnan(val):
                        val = 0.0
                    som += val

        return rho * gamma * som * si2mg

    @njit
    def _compute_gravity_grid_numba(d_Lx, d_Ly, dx, dy, z0, z1, dz1, rho_sed):
        ny = len(d_Ly)
        nx = len(d_Lx)
        g_norm = np.zeros((ny, nx))
        centre_x = dx / 2.0
        centre_y = dy / 2.0

        for m in range(ny):
            for i in range(nx):
                dz_ep_i = dz1[m, i]
                if dz_ep_i == 0.0:
                    continue
                for k in range(ny):
                    for j in range(nx):
                        g_norm[k, j] += _anomaly_numba(
                            centre_x + d_Lx[j],
                            centre_y + d_Ly[k],
                            z0[k, j],
                            d_Lx[i], d_Ly[m],
                            z1[m, i],
                            d_Lx[i] + dx,
                            d_Ly[m] + dy,
                            dz_ep_i,
                            rho_sed[m, i],
                        )
        return g_norm

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


def _normalize_density_mode(density_mode):
    """MGM internal routine."""
    mode = str(density_mode).strip().lower()
    if mode in {"decreasing", "decay", "variable"}:
        return "decreasing"
    if mode in {"constant", "uniform"}:
        return "constant"
    if mode in {"compaction", "porosity_depth", "athy"}:
        return "compaction"
    if mode in {"compaction_layered", "layered_compaction", "compaction_profile", "depth_profile"}:
        return "compaction_layered"
    raise ValueError(
        "density_mode doit valoir 'decreasing', 'constant', 'compaction' ou 'compaction_layered'."
    )


_COMPACTION_LITHOLOGY_PRESETS = {
    "sand": {"phi0": 0.49, "z0": 3703.0},
    "shaly_sand": {"phi0": 0.56, "z0": 2464.0},
    "shale": {"phi0": 0.63, "z0": 1960.0},
}


def _normalize_compaction_lithology(compaction_lithology):
    """MGM internal routine."""
    lithology = str(compaction_lithology).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "sandy": "sand",
        "sandstone": "sand",
        "shalysand": "shaly_sand",
        "shalysandstone": "shaly_sand",
        "shaly_sandstone": "shaly_sand",
        "mudstone": "shale",
        "clay": "shale",
    }
    lithology = aliases.get(lithology, lithology)
    if lithology not in _COMPACTION_LITHOLOGY_PRESETS:
        allowed = ", ".join(sorted(_COMPACTION_LITHOLOGY_PRESETS))
        raise ValueError(f"compaction_lithology doit valoir l'un de: {allowed}.")
    return lithology


def _compaction_preset_params(compaction_lithology):
    """MGM internal routine."""
    lithology = _normalize_compaction_lithology(compaction_lithology)
    return _COMPACTION_LITHOLOGY_PRESETS[lithology]


def _average_density_between_depths_from_compaction(
    z_top,
    z_bottom,
    phi0,
    z0,
    grain_density,
    fluid_density,
):
    """MGM internal routine."""
    z_top = np.asarray(z_top, dtype=float)
    z_bottom = np.asarray(z_bottom, dtype=float)
    dz = z_bottom - z_top
    rho_avg = np.zeros_like(dz, dtype=float)
    active_mask = dz > 0.0
    if not np.any(active_mask):
        return rho_avg

    dz_active = dz[active_mask]
    decay_term = (
        float(z0) / dz_active
    ) * (
        np.exp(-z_top[active_mask] / float(z0)) - np.exp(-z_bottom[active_mask] / float(z0))
    )
    rho_avg[active_mask] = float(grain_density) - (
        (float(grain_density) - float(fluid_density)) * float(phi0) * decay_term
    )
    return rho_avg


def _average_density_from_compaction(
    ep_sedo,
    phi0,
    z0,
    grain_density,
    fluid_density,
):
    """MGM internal routine."""
    ep_sedo = np.asarray(ep_sedo, dtype=float)
    rho_avg = np.zeros_like(ep_sedo)
    active_mask = ep_sedo > 0.0

    if not np.any(active_mask):
        return rho_avg

    thickness = ep_sedo[active_mask]
    compaction_factor = (float(z0) / thickness) * (1.0 - np.exp(-thickness / float(z0)))
    rho_avg[active_mask] = float(grain_density) - (
        (float(grain_density) - float(fluid_density)) * float(phi0) * compaction_factor
    )
    return rho_avg


def _build_compaction_sublayers(
    thickness,
    phi0,
    z0,
    grain_density,
    fluid_density,
    host_density,
    layer_thickness,
):
    """MGM internal routine."""
    thickness = float(thickness)
    layer_thickness = float(layer_thickness)
    if thickness <= 0.0:
        return np.empty(0), np.empty(0), np.empty(0)
    if layer_thickness <= 0.0:
        raise ValueError("layer_thickness doit être strictement positif pour la compaction couche-par-couche.")

    n_layers = max(1, int(np.ceil(thickness / layer_thickness)))
    edges = np.linspace(0.0, thickness, n_layers + 1, dtype=float)
    z_top_rel = edges[:-1]
    z_bottom_rel = edges[1:]
    rho_avg = _average_density_between_depths_from_compaction(
        z_top_rel,
        z_bottom_rel,
        phi0=phi0,
        z0=z0,
        grain_density=grain_density,
        fluid_density=fluid_density,
    )
    delta_rho = rho_avg - float(host_density)
    return z_top_rel, z_bottom_rel, delta_rho


def _column_delta_rho(
    thickness,
    density_mode="decreasing",
    constant_delta_rho=-170.0,
    compaction_lithology="shaly_sand",
    host_density=2670.0,
    grain_density=2650.0,
    fluid_density=1000.0,
):
    """MGM internal routine."""
    delta = _generate_delta_rho_sediments(
        np.asarray([[float(thickness)]], dtype=float),
        density_mode=density_mode,
        constant_delta_rho=constant_delta_rho,
        compaction_lithology=compaction_lithology,
        host_density=host_density,
        grain_density=grain_density,
        fluid_density=fluid_density,
    )
    return float(delta[0, 0])


def _iter_sediment_subprisms(
    column_top,
    thickness,
    density_mode="decreasing",
    constant_delta_rho=-170.0,
    compaction_lithology="shaly_sand",
    host_density=2670.0,
    grain_density=2650.0,
    fluid_density=1000.0,
    compaction_layer_thickness=50.0,
):
    """MGM internal routine."""
    mode = _normalize_density_mode(density_mode)
    thickness = float(thickness)
    if thickness <= 0.0:
        return

    if mode != "compaction_layered":
        delta_rho = _column_delta_rho(
            thickness,
            density_mode=mode,
            constant_delta_rho=constant_delta_rho,
            compaction_lithology=compaction_lithology,
            host_density=host_density,
            grain_density=grain_density,
            fluid_density=fluid_density,
        )
        yield float(column_top), float(column_top) + thickness, delta_rho
        return

    params = _compaction_preset_params(compaction_lithology)
    z_top_rel, z_bottom_rel, delta_rho_layers = _build_compaction_sublayers(
        thickness,
        phi0=params["phi0"],
        z0=params["z0"],
        grain_density=grain_density,
        fluid_density=fluid_density,
        host_density=host_density,
        layer_thickness=compaction_layer_thickness,
    )
    for rel_top, rel_bottom, delta_rho in zip(z_top_rel, z_bottom_rel, delta_rho_layers):
        yield float(column_top) + float(rel_top), float(column_top) + float(rel_bottom), float(delta_rho)


def _generate_delta_rho_sediments(
    ep_sedo,
    density_mode="decreasing",
    constant_delta_rho=-170.0,
    compaction_lithology="shaly_sand",
    host_density=2670.0,
    grain_density=2650.0,
    fluid_density=1000.0,
):
    """MGM internal routine."""
    ep_sedo = np.asarray(ep_sedo, dtype=float)
    delta_rho = np.zeros_like(ep_sedo)
    active_mask = ep_sedo > 0.0
    mode = _normalize_density_mode(density_mode)

    if mode == "constant":
        delta_rho[active_mask] = float(constant_delta_rho)
        return delta_rho

    if mode in {"compaction", "compaction_layered"}:
        params = _compaction_preset_params(compaction_lithology)
        rho_avg = _average_density_from_compaction(
            ep_sedo,
            phi0=params["phi0"],
            z0=params["z0"],
            grain_density=grain_density,
            fluid_density=fluid_density,
        )
        delta_rho[active_mask] = rho_avg[active_mask] - float(host_density)
        return delta_rho

    delta_rho[(ep_sedo > 0.0) & (ep_sedo < 200.0)] = -170.0
    delta_rho[(ep_sedo >= 200.0) & (ep_sedo < 500.0)] = -120.0
    delta_rho[(ep_sedo >= 500.0) & (ep_sedo < 1000.0)] = -70.0
    delta_rho[ep_sedo >= 1000.0] = -30.0
    return delta_rho


def sort_grav(
    barringer,
    ep_sedo,
    topo,
    nbx,
    nby,
    nb,
    X,
    Y,
    spc,
    dxy,
    method="partition",
    density_mode="decreasing",
    constant_delta_rho=-170.0,
    compaction_lithology="shaly_sand",
    host_density=2670.0,
    grain_density=2650.0,
    fluid_density=1000.0,
):
    """MGM internal routine."""

    if method == "numba" and not _NUMBA_AVAILABLE:
        raise ImportError("numba n'est pas installé. Utilisez method='partition' ou installez numba.")

    gravi      = []
    gravi_prof = []

    for a in tqdm(range(nb), desc="Traitement en cours ", bar_format="{l_bar}{bar:10}{r_bar}", unit=" gravi iteration", unit_scale=True, leave=True):

        
        dz          = np.reshape(ep_sedo[a], barringer.shape)
        rho_sed     = _generate_delta_rho_sediments(
            dz,
            density_mode=density_mode,
            constant_delta_rho=constant_delta_rho,
            compaction_lithology=compaction_lithology,
            host_density=host_density,
            grain_density=grain_density,
            fluid_density=fluid_density,
        )
        z_topo      = topo[a]
        x_topo      = X
        y_topo      = Y

        Topographie = z_topo.reshape(dz.shape)

        
        Lx  = np.max(x_topo) - np.min(x_topo)
        Ly  = np.max(y_topo) - np.min(y_topo)

        
        dx  = dxy
        dy  = dxy

        
        d_Lx = np.arange(0, Lx, dx) 
        d_Ly = np.arange(0, Ly, dy) 

        
        z0    = np.zeros(dz.shape)   
        z1    = z0 - dz
        dz1   = dz

        if method == "numba":
            
            g_norm = _compute_gravity_grid_numba(d_Lx, d_Ly, dx, dy, z0, z1, dz1, rho_sed)
            
            if g_norm.shape != dz.shape:
                g_padded = np.zeros(dz.shape)
                g_padded[:g_norm.shape[0], :g_norm.shape[1]] = g_norm
                g_norm = g_padded
        else:
            
            centre_x = dx / 2
            centre_y = dy / 2
            space_calc = spc
            g_norm = np.zeros(dz.shape)

            for m in range(len(d_Ly)):

                dz_ep     = dz1[m,:]

                for i in range(len(d_Lx)):         

                    if dz_ep[i]==0:

                        g_norm += 0

                    else:

                        
                        a = max(i - space_calc, 0)
                        b = min(i + space_calc + 1, len(d_Lx))
                        c = max(m - space_calc, 0)
                        d = min(m + space_calc + 1, len(d_Ly))

                        for k in range(c, d):  
                            for j in range(a, b):
                                if k < g_norm.shape[0] and j < g_norm.shape[1]:

                                    g_norm[k,j] += anomaly(centre_x + d_Lx[j], centre_y + d_Ly[k], z0[k,j], d_Lx[i], d_Ly[m], z1[m,i], d_Lx[i]+dx, d_Ly[m]+dy, dz_ep[i], rho_sed[m,i])

        gravi.append(g_norm)
        gravi_prof.append(g_norm[int(nby/2),:])

    return gravi, gravi_prof

def sort_grav_corrected(
    barringer,
    ep_sedo,
    topo,
    nbx,
    nby,
    nb,
    X,
    Y,
    spc,
    dxy,
    delta_rho_topo=-230.0,
    density_mode="decreasing",
    constant_delta_rho=-170.0,
    compaction_lithology="shaly_sand",
    host_density=2670.0,
    grain_density=2650.0,
    fluid_density=1000.0,
):
    """MGM internal routine."""

    gravi = []
    gravi_prof = []

    for a in tqdm(range(nb), desc="Traitement en cours", unit=" itération"):
        
        dz = np.reshape(ep_sedo[a], barringer.shape)
        
        delta_rho_sed = _generate_delta_rho_sediments(
            dz,
            density_mode=density_mode,
            constant_delta_rho=constant_delta_rho,
            compaction_lithology=compaction_lithology,
            host_density=host_density,
            grain_density=grain_density,
            fluid_density=fluid_density,
        )

        z_topo = topo[a]
        x_topo = X
        y_topo = Y

        Topographie = z_topo.reshape(dz.shape)  

        
        Lx = np.max(x_topo) - np.min(x_topo)
        Ly = np.max(y_topo) - np.min(y_topo)
        dx = dxy
        dy = dxy

        
        d_Lx = np.arange(0, Lx, dx)
        d_Ly = np.arange(0, Ly, dy)

        centre_x = dx / 2.0
        centre_y = dy / 2.0
        space_calc = spc  

        
        g_norm = np.zeros(dz.shape)
        
        
        z0 = Topographie.copy()
        z1 = z0 - dz

        for m in range(len(d_Ly)):
            dz_ep = dz[m, :]
            for i in range(len(d_Lx)):
                if dz_ep[i] == 0:
                    continue  
                else:
                    a_idx = max(i - space_calc, 0)
                    b_idx = min(i + space_calc + 1, len(d_Lx))
                    c_idx = max(m - space_calc, 0)
                    d_idx = min(m + space_calc + 1, len(d_Ly))
                    for k in range(c_idx, d_idx):
                        for j in range(a_idx, b_idx):
                            g_norm[k, j] += anomaly(
                                centre_x + d_Lx[j],
                                centre_y + d_Ly[k],
                                z0[k, j],
                                d_Lx[i],
                                d_Ly[m],
                                z1[m, i],
                                d_Lx[i] + dx,
                                d_Ly[m] + dy,
                                dz_ep[i],
                                delta_rho_sed[m, i]
                            )

        
        
        
        g_topo = np.zeros(dz.shape)
        
        z_ref = - z1

        for m in range(len(d_Ly)):
            for i in range(len(d_Lx)):
                
                if (z0[m, i] - z_ref[m,i]) == 0:
                    continue
                else:
                    a_idx = max(i - space_calc, 0)
                    b_idx = min(i + space_calc + 1, len(d_Lx))
                    c_idx = max(m - space_calc, 0)
                    d_idx = min(m + space_calc + 1, len(d_Ly))
                    for k in range(c_idx, d_idx):
                        for j in range(a_idx, b_idx):
                            g_topo[k, j] += anomaly(
                                centre_x + d_Lx[j],
                                centre_y + d_Ly[k],
                                z0[k, j],
                                d_Lx[i],
                                d_Ly[m],
                                z_ref[m,i],
                                d_Lx[i] + dx,
                                d_Ly[m] + dy,
                                z0[m, i],
                                delta_rho_topo
                            )

        
        g_corr = g_norm - g_topo

        gravi.append(g_corr)
        gravi_prof.append(g_corr[int(nby/2), :])

    return gravi, gravi_prof

def correction_relief(topographies, dx, dy, delta_rho, spc, nby):
    """MGM internal routine."""
    
    G = 6.67430e-11
    si2mg = 1e5
    space_calc = spc

    def integrand(x_p, y_p, z_p, x, y, z):
        delta_x = x_p - x
        delta_y = y_p - y
        delta_z = z_p - z
        return delta_z / ((delta_x**2 + delta_y**2 + delta_z**2)**(3/2))

    corrections_list = []
    corrections_list_prof = []

    for altitudes in topographies:
        nrows, ncols = altitudes.shape
        corrections = np.zeros_like(altitudes, dtype=float)

        for i_p in range(nrows):
            print(i_p)
            for j_p in range(ncols):
                z_p = altitudes[i_p, j_p]
                x_p = j_p * dx
                y_p = i_p * dy
                correction_totale = 0

                
                a = max(j_p - space_calc, 0)
                b = min(j_p + space_calc + 1, len(ncols))
                c = max(i_p - space_calc, 0)
                d = min(i_p + space_calc + 1, len(nrows))

                for k in range(c, d):  
                    for j in range(a, b):
                        if k == i_p and j == j_p:
                            continue
                        z = altitudes[i_p, j_p]
                        x = j_p * dx
                        y = i_p * dy
                        correction_totale += G * delta_rho * integrand(x_p, y_p, z_p, x, y, z) * dx * dy * si2mg
                corrections[i_p, j_p] = correction_totale

        corrections_list.append(corrections)
        corrections_list_prof.append(corrections[int(nby/2),:])

    return corrections_list, corrections_list_prof


def _prepare_gravity_inputs(topo_snapshots, epais_snapshots):
    """MGM internal routine."""
    if len(topo_snapshots) != len(epais_snapshots):
        raise ValueError("TOPO et EPAIS doivent contenir le même nombre de snapshots.")

    topo_list = [np.asarray(topo_snap, dtype=float) for topo_snap in topo_snapshots]
    if not topo_list:
        raise ValueError("Aucun snapshot topographique disponible pour la gravimétrie.")

    expected_shape = topo_list[0].shape
    epais_list = [
        np.asarray(epais_snap, dtype=float).reshape(expected_shape)
        for epais_snap in epais_snapshots
    ]
    return topo_list, epais_list
if _NUMBA_AVAILABLE:
    @njit
    def _compute_gravity_grid_bouguer_numba(
        d_lx,
        d_ly,
        dx,
        dy,
        reduction_level,
        surface_z_down,
        thickness,
        delta_rho,
        space_calc,
    ):
        ny = surface_z_down.shape[0]
        nx = surface_z_down.shape[1]
        g_norm = np.zeros((ny, nx))
        centre_x = dx / 2.0
        centre_y = dy / 2.0

        for m in range(ny):
            for i in range(nx):
                th = thickness[m, i]
                if th == 0.0:
                    continue

                z_top = surface_z_down[m, i]
                z_bottom = z_top + th
                a_idx = max(i - space_calc, 0)
                b_idx = min(i + space_calc + 1, nx)
                c_idx = max(m - space_calc, 0)
                d_idx = min(m + space_calc + 1, ny)

                for k in range(c_idx, d_idx):
                    for j in range(a_idx, b_idx):
                        g_norm[k, j] += _anomaly_numba(
                            centre_x + d_lx[j],
                            centre_y + d_ly[k],
                            reduction_level,
                            d_lx[i],
                            d_ly[m],
                            z_top,
                            d_lx[i] + dx,
                            d_ly[m] + dy,
                            z_bottom,
                            delta_rho[m, i],
                        )

        return g_norm

    @njit
    def _compute_exact_terrain_effect_numba(
        d_lx,
        d_ly,
        dx,
        dy,
        reduction_level,
        topo_up,
        rho_terrain,
        space_calc,
    ):
        ny, nx = topo_up.shape
        g = np.zeros((ny, nx))
        centre_x = dx / 2.0
        centre_y = dy / 2.0

        for m in range(ny):
            for i in range(nx):
                h = topo_up[m, i]
                if h == 0.0:
                    continue

                if h > 0.0:
                    z_top = reduction_level - h
                    z_bottom = reduction_level
                    rho_cell = rho_terrain
                else:
                    z_top = reduction_level
                    z_bottom = reduction_level - h
                    rho_cell = -rho_terrain

                a_idx = max(i - space_calc, 0)
                b_idx = min(i + space_calc + 1, nx)
                c_idx = max(m - space_calc, 0)
                d_idx = min(m + space_calc + 1, ny)

                for k in range(c_idx, d_idx):
                    for j in range(a_idx, b_idx):
                        g[k, j] += _anomaly_numba(
                            centre_x + d_lx[j],
                            centre_y + d_ly[k],
                            reduction_level,
                            d_lx[i],
                            d_ly[m],
                            z_top,
                            d_lx[i] + dx,
                            d_ly[m] + dy,
                            z_bottom,
                            rho_cell,
                        )

        return g


def sort_grav_bouguer_reference(
    barringer,
    ep_sedo,
    topo,
    nbx,
    nby,
    nb,
    X,
    Y,
    spc,
    dxy,
    reduction_level=0.0,
    method="numba",
    density_mode="decreasing",
    constant_delta_rho=-170.0,
    compaction_lithology="shaly_sand",
    host_density=2670.0,
    grain_density=2650.0,
    fluid_density=1000.0,
    compaction_layer_thickness=50.0,
):
    """MGM internal routine."""
    ny, nx = barringer.shape
    dx = float(dxy)
    dy = float(dxy)
    d_lx = np.arange(0.0, nx * dx, dx)
    d_ly = np.arange(0.0, ny * dy, dy)
    space_calc = int(spc)
    density_mode = _normalize_density_mode(density_mode)

    gravi = []
    gravi_prof = []

    for step in range(int(nb)):
        dz = np.reshape(ep_sedo[step], barringer.shape).astype(float)
        topo_up = np.reshape(topo[step], barringer.shape).astype(float)
        delta_rho = None
        if density_mode != "compaction_layered":
            delta_rho = _generate_delta_rho_sediments(
                dz,
                density_mode=density_mode,
                constant_delta_rho=constant_delta_rho,
                compaction_lithology=compaction_lithology,
                host_density=host_density,
                grain_density=grain_density,
                fluid_density=fluid_density,
            )
        surface_z_down = reduction_level - topo_up

        if method == "numba":
            if density_mode == "compaction_layered":
                raise ImportError(
                    "Le mode 'compaction_layered' requiert le backend python pour sommer les sous-couches."
                )
            if not _NUMBA_AVAILABLE:
                raise ImportError("numba n'est pas installé. Utilisez method='python'.")
            g_norm = _compute_gravity_grid_bouguer_numba(
                d_lx=d_lx,
                d_ly=d_ly,
                dx=dx,
                dy=dy,
                reduction_level=float(reduction_level),
                surface_z_down=surface_z_down,
                thickness=dz,
                delta_rho=delta_rho,
                space_calc=space_calc,
            )
        else:
            centre_x = dx / 2.0
            centre_y = dy / 2.0
            g_norm = np.zeros_like(dz, dtype=float)

            for m in range(ny):
                for i in range(nx):
                    thickness = dz[m, i]
                    if thickness == 0.0:
                        continue

                    a_idx = max(i - space_calc, 0)
                    b_idx = min(i + space_calc + 1, nx)
                    c_idx = max(m - space_calc, 0)
                    d_idx = min(m + space_calc + 1, ny)

                    for z_top, z_bottom, delta_rho_cell in _iter_sediment_subprisms(
                        surface_z_down[m, i],
                        thickness,
                        density_mode=density_mode,
                        constant_delta_rho=constant_delta_rho,
                        compaction_lithology=compaction_lithology,
                        host_density=host_density,
                        grain_density=grain_density,
                        fluid_density=fluid_density,
                        compaction_layer_thickness=compaction_layer_thickness,
                    ):
                        for k in range(c_idx, d_idx):
                            for j in range(a_idx, b_idx):
                                g_norm[k, j] += anomaly(
                                    centre_x + d_lx[j],
                                    centre_y + d_ly[k],
                                    reduction_level,
                                    d_lx[i],
                                    d_ly[m],
                                    z_top,
                                    d_lx[i] + dx,
                                    d_ly[m] + dy,
                                    z_bottom,
                                    delta_rho_cell,
                                )

        gravi.append(g_norm)
        gravi_prof.append(g_norm[int(nby) // 2, :])

    return gravi, gravi_prof


def bouguer_slab_signed(topography_up, rho_terrain):
    """MGM internal routine."""
    gamma = 6.673e-11
    si2mg = 1e5
    return 2.0 * np.pi * gamma * float(rho_terrain) * np.asarray(topography_up, dtype=float) * si2mg


def compute_exact_terrain_effect(topography_up, X, Y, dxy, rho_terrain, spc=None, method="numba"):
    """MGM internal routine."""
    del X, Y
    topo_up = np.asarray(topography_up, dtype=float)
    ny, nx = topo_up.shape
    dx = float(dxy)
    dy = float(dxy)
    d_lx = np.arange(0.0, nx * dx, dx)
    d_ly = np.arange(0.0, ny * dy, dy)
    reduction_level = 0.0
    if spc is None:
        spc = max(nx, ny)
    space_calc = int(spc)

    if method == "numba":
        if not _NUMBA_AVAILABLE:
            raise ImportError("numba n'est pas installé. Utilisez method='python'.")
        return _compute_exact_terrain_effect_numba(
            d_lx=d_lx,
            d_ly=d_ly,
            dx=dx,
            dy=dy,
            reduction_level=reduction_level,
            topo_up=topo_up,
            rho_terrain=float(rho_terrain),
            space_calc=space_calc,
        )

    g = np.zeros_like(topo_up)
    centre_x = dx / 2.0
    centre_y = dy / 2.0

    for m in range(ny):
        for i in range(nx):
            h = topo_up[m, i]
            if h == 0.0:
                continue

            if h > 0.0:
                z_top = reduction_level - h
                z_bottom = reduction_level
                rho_cell = rho_terrain
            else:
                z_top = reduction_level
                z_bottom = reduction_level - h
                rho_cell = -rho_terrain

            a_idx = max(i - space_calc, 0)
            b_idx = min(i + space_calc + 1, nx)
            c_idx = max(m - space_calc, 0)
            d_idx = min(m + space_calc + 1, ny)

            for k in range(c_idx, d_idx):
                for j in range(a_idx, b_idx):
                    g[k, j] += anomaly(
                        centre_x + d_lx[j],
                        centre_y + d_ly[k],
                        reduction_level,
                        d_lx[i],
                        d_ly[m],
                        z_top,
                        d_lx[i] + dx,
                        d_ly[m] + dy,
                        z_bottom,
                        rho_cell,
                    )

    return g


def compute_terrain_correction(topography_up, X, Y, dxy, rho_terrain, spc=None, method="numba"):
    """MGM internal routine."""
    exact = compute_exact_terrain_effect(
        topography_up,
        X,
        Y,
        dxy=dxy,
        rho_terrain=rho_terrain,
        spc=spc,
        method=method,
    )
    slab = bouguer_slab_signed(topography_up, rho_terrain)
    terrain_correction = slab - exact
    return exact, slab, terrain_correction


def compute_gravity_anomalies(topo_snapshots, epais_snapshots, X, Y, params):
    """MGM internal routine."""
    topo_list, epais_list = _prepare_gravity_inputs(topo_snapshots, epais_snapshots)
    nby, nbx = topo_list[0].shape
    nb = len(topo_list)
    template = np.zeros((nby, nbx), dtype=float)
    gravity_method = str(params.get("gravity_method", "reference")).lower()
    gravity_kernel = str(params.get("gravity_kernel", "numba")).lower()
    gravity_density_mode = str(params.get("gravity_density_mode", "decreasing")).lower()
    gravity_constant_delta_rho = float(params.get("gravity_constant_delta_rho", -170.0))
    gravity_compaction_lithology = str(params.get("gravity_compaction_lithology", "shaly_sand"))
    gravity_host_density = float(params.get("gravity_host_density", 2670.0))
    gravity_grain_density = float(params.get("gravity_grain_density", 2650.0))
    gravity_fluid_density = float(params.get("gravity_fluid_density", 1000.0))
    gravity_compaction_layer_thickness = float(params.get("gravity_compaction_layer_thickness", 50.0))
    gravity_spc = int(params.get("gravity_spc", 1))
    gravity_terrain_spc = params.get("gravity_terrain_spc", max(nbx, nby))
    gravity_terrain_density = float(params.get("gravity_terrain_density", 2670.0))
    xy_space = float(params["xy_space"])
    _normalize_density_mode(gravity_density_mode)
    _normalize_compaction_lithology(gravity_compaction_lithology)
    if gravity_compaction_layer_thickness <= 0.0:
        raise ValueError("gravity_compaction_layer_thickness doit être strictement positif.")

    if gravity_method == "both":
        requested_methods = ["legacy", "reference"]
    elif gravity_method == "all":
        requested_methods = ["legacy", "reference", "complete"]
    else:
        requested_methods = [gravity_method]
    for method_name in requested_methods:
        if method_name not in {"legacy", "reference", "complete"}:
            raise ValueError(
                "gravity_method doit valoir 'legacy', 'reference', 'complete', 'both' ou 'all'."
            )
    if gravity_density_mode == "compaction_layered" and "legacy" in requested_methods:
        raise ValueError(
            "gravity_density_mode='compaction_layered' n'est supporte qu'avec "
            "gravity_method='reference' ou 'complete'."
        )

    background = None
    if params.get("gravity_add_initial_background", True):
        from Topo import generate_gravi

        background = np.asarray(
            generate_gravi(
                params["file_path_grav"],
                params["xy_space"],
                params["limit"],
            ),
            dtype=float,
        )
        background = background - background[-1]
        if background.shape != template.shape:
            raise ValueError(
                "La grille gravimétrique initiale n'a pas la même forme que les snapshots de simulation."
            )

    outputs = {}
    reference_raw_maps = None
    reference_backend = None

    if "legacy" in requested_methods:
        legacy_backend = "numba" if gravity_kernel == "numba" else "partition"
        try:
            legacy_maps, _ = sort_grav(
                template,
                epais_list,
                topo_list,
                nbx,
                nby,
                nb,
                X,
                Y,
                gravity_spc,
                xy_space,
                method=legacy_backend,
                density_mode=gravity_density_mode,
                constant_delta_rho=gravity_constant_delta_rho,
                compaction_lithology=gravity_compaction_lithology,
                host_density=gravity_host_density,
                grain_density=gravity_grain_density,
                fluid_density=gravity_fluid_density,
            )
        except ImportError:
            legacy_backend = "partition"
            legacy_maps, _ = sort_grav(
                template,
                epais_list,
                topo_list,
                nbx,
                nby,
                nb,
                X,
                Y,
                gravity_spc,
                xy_space,
                method=legacy_backend,
                density_mode=gravity_density_mode,
                constant_delta_rho=gravity_constant_delta_rho,
                compaction_lithology=gravity_compaction_lithology,
                host_density=gravity_host_density,
                grain_density=gravity_grain_density,
                fluid_density=gravity_fluid_density,
            )

        legacy_maps = np.asarray(legacy_maps, dtype=float)
        if background is not None:
            legacy_maps = legacy_maps + background[np.newaxis, :, :]
        outputs["legacy"] = {
            "maps": legacy_maps,
            "backend": legacy_backend,
            "title": "Anomalies gravimetriques - Legacy sort_grav",
        }

    if any(method_name in requested_methods for method_name in ("reference", "complete")):
        if gravity_density_mode == "compaction_layered":
            reference_backend = "python"
        else:
            reference_backend = "numba" if gravity_kernel == "numba" else "python"
        try:
            reference_raw_maps, _ = sort_grav_bouguer_reference(
                template,
                epais_list,
                topo_list,
                nbx,
                nby,
                nb,
                X,
                Y,
                gravity_spc,
                xy_space,
                method=reference_backend,
                density_mode=gravity_density_mode,
                constant_delta_rho=gravity_constant_delta_rho,
                compaction_lithology=gravity_compaction_lithology,
                host_density=gravity_host_density,
                grain_density=gravity_grain_density,
                fluid_density=gravity_fluid_density,
                compaction_layer_thickness=gravity_compaction_layer_thickness,
            )
        except ImportError:
            reference_backend = "python"
            reference_raw_maps, _ = sort_grav_bouguer_reference(
                template,
                epais_list,
                topo_list,
                nbx,
                nby,
                nb,
                X,
                Y,
                gravity_spc,
                xy_space,
                method=reference_backend,
                density_mode=gravity_density_mode,
                constant_delta_rho=gravity_constant_delta_rho,
                compaction_lithology=gravity_compaction_lithology,
                host_density=gravity_host_density,
                grain_density=gravity_grain_density,
                fluid_density=gravity_fluid_density,
                compaction_layer_thickness=gravity_compaction_layer_thickness,
            )

        reference_raw_maps = np.asarray(reference_raw_maps, dtype=float)

    if "reference" in requested_methods:
        reference_maps = np.asarray(reference_raw_maps, dtype=float)
        if background is not None:
            reference_maps = reference_maps + background[np.newaxis, :, :]
        outputs["reference"] = {
            "maps": reference_maps,
            "backend": reference_backend,
            "title": "Anomalies gravimetriques - Reference Bouguer",
        }

    if "complete" in requested_methods:
        terrain_backend = "numba" if gravity_kernel == "numba" else "python"
        exact_terrain_stack = []
        bouguer_slab_stack = []
        terrain_correction_stack = []

        for topo_snapshot in topo_list:
            try:
                terrain_exact, bouguer_slab, terrain_correction = compute_terrain_correction(
                    topo_snapshot,
                    X,
                    Y,
                    dxy=xy_space,
                    rho_terrain=gravity_terrain_density,
                    spc=gravity_terrain_spc,
                    method=terrain_backend,
                )
            except ImportError:
                terrain_backend = "python"
                terrain_exact, bouguer_slab, terrain_correction = compute_terrain_correction(
                    topo_snapshot,
                    X,
                    Y,
                    dxy=xy_space,
                    rho_terrain=gravity_terrain_density,
                    spc=gravity_terrain_spc,
                    method=terrain_backend,
                )

            exact_terrain_stack.append(terrain_exact)
            bouguer_slab_stack.append(bouguer_slab)
            terrain_correction_stack.append(terrain_correction)

        exact_terrain_stack = np.asarray(exact_terrain_stack, dtype=float)
        bouguer_slab_stack = np.asarray(bouguer_slab_stack, dtype=float)
        terrain_correction_stack = np.asarray(terrain_correction_stack, dtype=float)
        simple_bouguer_maps = reference_raw_maps + exact_terrain_stack - bouguer_slab_stack
        complete_maps = np.asarray(reference_raw_maps, dtype=float)

        if background is not None:
            simple_bouguer_maps = simple_bouguer_maps + background[np.newaxis, :, :]
            complete_maps = complete_maps + background[np.newaxis, :, :]

        outputs["complete"] = {
            "maps": complete_maps,
            "backend": f"sediments={reference_backend}, terrain={terrain_backend}",
            "title": "Anomalies gravimetriques - Bouguer complet (proxy)",
        }
        outputs["simple_bouguer"] = {
            "maps": simple_bouguer_maps,
            "backend": terrain_backend,
            "title": "Anomalies gravimetriques - Bouguer simple (proxy)",
        }
        outputs["terrain_correction"] = {
            "maps": terrain_correction_stack,
            "backend": terrain_backend,
            "title": "Correction de terrain explicite",
        }
        outputs["terrain_exact"] = {
            "maps": exact_terrain_stack,
            "backend": terrain_backend,
            "title": "Effet gravimetrique exact du relief",
        }
        outputs["bouguer_slab"] = {
            "maps": bouguer_slab_stack,
            "backend": terrain_backend,
            "title": "Lame de Bouguer signee",
        }

    if "legacy" in outputs and "reference" in outputs:
        outputs["comparison"] = {
            "maps": outputs["legacy"]["maps"] - outputs["reference"]["maps"],
            "title": "Difference gravimetrique - Legacy moins Reference",
            "first_label": "Legacy",
            "second_label": "Reference",
        }

    if "legacy" in outputs and "complete" in outputs:
        outputs["comparison_legacy_complete"] = {
            "maps": outputs["legacy"]["maps"] - outputs["complete"]["maps"],
            "title": "Difference gravimetrique - Legacy moins Bouguer complet",
            "first_label": "Legacy",
            "second_label": "Complete",
        }

    return outputs

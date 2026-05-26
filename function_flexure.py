#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flexural load and deflection routines used by MGM."""

import numpy as np
import scipy.special as sp
from scipy.special import kei
import math as mt

def calc_load(sed, rho, g, nb):
    """MGM internal routine."""
    
    if nb==0:
    
        qs = sed * rho * g
    
    else:
        
        qs = sed * g
    
    return qs

def calc_flex_extended(
    qs,
    nx,
    ny,
    dx,
    dy,
    E,
    Te,
    nu,
    rhom,
    rhoc,
    g,
    margin_km,
    crater_radius_km,
    outside_fill_mode="zero",
):
    """Compute flexural deflection on an extended domain."""

    margin_x = int(margin_km * 1000 / dx)
    margin_y = int(margin_km * 1000 / dy)
    crater_radius_nodes = int(crater_radius_km * 1000 / dx)

    nx_ext = nx + 2 * margin_x
    ny_ext = ny + 2 * margin_y

    if qs.shape != (ny, nx):
        raise ValueError(
            f"Incompatible qs shape: expected {(ny, nx)}, got {qs.shape}."
        )

    qs_ext = np.zeros((ny_ext, nx_ext))
    qs_ext[margin_y:margin_y + ny, margin_x:margin_x + nx] = qs

    center_x = nx // 2 + margin_x
    center_y = ny // 2 + margin_y
    crater_mask = np.fromfunction(
        lambda i, j: (i - center_y)**2 + (j - center_x)**2 < crater_radius_nodes**2,
        (ny_ext, nx_ext),
    )

    initial_domain_mask = np.zeros_like(qs_ext, dtype=bool)
    initial_domain_mask[margin_y:margin_y + ny, margin_x:margin_x + nx] = True
    initial_domain_mask[crater_mask] = False

    outside_domain_mask = ~initial_domain_mask
    outside_domain_mask[margin_y:margin_y + ny, margin_x:margin_x + nx] = False

    if outside_fill_mode == "zero":
        qs_ext[outside_domain_mask] = 0.0
    elif outside_fill_mode == "mean":
        qs_mean = float(np.mean(qs_ext[initial_domain_mask])) if np.any(initial_domain_mask) else 0.0
        qs_ext[outside_domain_mask] = qs_mean
    elif outside_fill_mode == "edge":
        padded = np.pad(qs, ((margin_y, margin_y), (margin_x, margin_x)), mode="edge")
        qs_ext[outside_domain_mask] = padded[outside_domain_mask]
    else:
        raise ValueError("outside_fill_mode must be 'zero', 'mean', or 'edge'.")

    D = (E * Te**3) / (12 * (1 - nu**2))
    alpha_2D = (D / ((rhom - rhoc) * g))**(1/4)

    bigshape = 2 * ny_ext + 1, 2 * nx_ext + 1
    dist_ny = np.arange(bigshape[0]) - ny_ext
    dist_nx = np.arange(bigshape[1]) - nx_ext
    dist_x, dist_y = np.meshgrid(dist_nx * dx, dist_ny * dy)

    bigdist = np.sqrt(dist_x**2 + dist_y**2)
    coef = alpha_2D**2 / (2 * np.pi * D)
    biggrid = coef * sp.kei(bigdist / alpha_2D)

    w_total_ext = np.zeros((ny_ext, nx_ext))
    for row_idx, col_idx in zip(*np.nonzero(qs_ext)):
        w_total_ext += (
            qs_ext[row_idx, col_idx]
            * dx
            * dy
            * biggrid[
                ny_ext - row_idx : 2 * ny_ext - row_idx,
                nx_ext - col_idx : 2 * nx_ext - col_idx,
            ]
        )

    return w_total_ext[margin_y:margin_y + ny, margin_x:margin_x + nx]

def calc_flex(qs, nx, ny, dx, dy, E, Te, nu, rhom, rhoc, g):
    """MGM internal routine."""
    
    if qs.shape != (ny, nx):
        raise ValueError(
            f"Shape de qs incompatible: attendu {(ny, nx)}, recu {qs.shape}."
        )

    
    D = (E*Te**3)/(12*(1-nu**2))
    
    
    alpha_2D = ((D)/((rhom-rhoc)*g))**(1/4)
    
    
    bigshape = 2*ny+1, 2*nx+1 
    
    dist_ny = np.arange(bigshape[0]) - ny
    dist_nx = np.arange(bigshape[1]) - nx
    
    dist_x, dist_y = np.meshgrid(dist_nx*dx, dist_ny*dy)
    
    
    bigdist = np.sqrt(dist_x**2 + dist_y**2)
    
    coef = alpha_2D**2 / (2*np.pi*D)
    
    biggrid = coef * kei(bigdist/alpha_2D)
    
    w_total  = np.zeros([ny, nx])

    for i in range(nx):
        for j in range(ny):
            if qs[j, i]:
                w_total += (
                    qs[j, i]
                    * dx
                    * dy
                    * biggrid[
                        ny - j : 2 * ny - j, nx - i : 2 * nx - i
                        ]
                    )
    
    return w_total


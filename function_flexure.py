#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flexural load and deflection routines used by MGM."""

import numpy as np
import scipy.special as sp
from scipy.special import kei
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve
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
    sign_convention="topographic",
):
    """Compute flexural deflection on an extended domain.

    Analytical (SAS) solver: Kelvin-function Green function (Turcotte &
    Schubert / gFlex "Superposition of Analytical Solutions").

    sign_convention : {"topographic", "mathematical"}
        - "topographic": a positive load gives a negative deflection
          (subsidence), directly addable to z-up topography (historical behaviour).
        - "mathematical": raw Green-function sign (opposite).
    """

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

    w_total = w_total_ext[margin_y:margin_y + ny, margin_x:margin_x + nx]

    # In the Kelvin convention used here, a positive load already gives a
    # negative deflection (subsidence) in z-up topographic convention.
    if str(sign_convention).lower() == "topographic":
        return w_total
    if str(sign_convention).lower() == "mathematical":
        return -w_total
    raise ValueError("sign_convention must be 'topographic' or 'mathematical'.")

def _laplacian_1d(n, spacing, boundary):
    """Sparse 1-D Laplacian used by the finite-difference (Wickert) solver."""
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    L = diags([off, main, off], offsets=[-1, 0, 1], shape=(n, n), format="lil")

    boundary = str(boundary).lower()
    if boundary in {"free_slope", "neumann"}:
        L[0, 0] = -1.0
        L[0, 1] = 1.0
        L[-1, -1] = -1.0
        L[-1, -2] = 1.0
    elif boundary in {"clamped_edge", "dirichlet"}:
        pass
    else:
        raise ValueError(
            "boundary must be 'free_slope'/'neumann' or 'clamped_edge'/'dirichlet'."
        )

    return L.tocsr() / spacing ** 2


def calc_flex_wickert_fd(
    qs,
    dx,
    dy,
    E,
    Te,
    nu,
    rhom,
    rho_f,
    g,
    boundary="free_slope",
    sign_convention="topographic",
    margin_km=0.0,
):
    """Finite-difference (sparse) flexure solver, gFlex/Wickert philosophy.

    Solves directly on the grid

        D nabla^4 w + (rho_m - rho_f) g w = q

    instead of convolving with the Kelvin Green function. Not a full gFlex
    reimplementation, but the same numerical approach.

    Parameters
    ----------
    qs : ndarray, shape (ny, nx)
        Surface load [N/m^2].
    dx, dy : float
        Grid spacing [m].
    E, Te, nu : float
        Elastic parameters.
    rhom : float
        Mantle density [kg/m^3].
    rho_f : float
        Density of the material filling the deflection [kg/m^3].
    g : float
        Gravity [m/s^2].
    boundary : {"free_slope", "neumann", "clamped_edge", "dirichlet"}
        Simplified boundary condition.
    sign_convention : {"topographic", "mathematical"}
        - "topographic": positive load -> subsidence (w < 0), addable to z-up topo.
        - "mathematical": q used as-is on the right-hand side.
    margin_km : float
        Width of a zero-load buffer added around the domain before solving,
        in km. The deflection is then cropped back to the original domain.
        Pushes the boundary conditions away from the region of interest
        (FD equivalent of the SAS solver margin).

    Returns
    -------
    w : ndarray, shape (ny, nx)
        Deflection [m] on the original domain.
    """
    qs = np.asarray(qs, dtype=float)
    ny, nx = qs.shape

    if nx < 5 or ny < 5:
        raise ValueError("Grid must be at least 5 x 5 nodes.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be strictly positive.")
    if rhom <= rho_f:
        raise ValueError("rhom must be strictly greater than rho_f.")

    margin_x = int(margin_km * 1000.0 / dx)
    margin_y = int(margin_km * 1000.0 / dy)
    if margin_x > 0 or margin_y > 0:
        qs_solve = np.pad(qs, ((margin_y, margin_y), (margin_x, margin_x)), mode="constant")
    else:
        qs_solve = qs
    ny_s, nx_s = qs_solve.shape

    D = (E * Te ** 3) / (12.0 * (1.0 - nu ** 2))
    restoring = (rhom - rho_f) * g

    Lx = _laplacian_1d(nx_s, dx, boundary)
    Ly = _laplacian_1d(ny_s, dy, boundary)
    Ix = eye(nx_s, format="csr")
    Iy = eye(ny_s, format="csr")

    L2 = kron(Iy, Lx, format="csr") + kron(Ly, Ix, format="csr")
    L4 = L2 @ L2
    A = D * L4 + restoring * eye(nx_s * ny_s, format="csr")

    rhs = qs_solve.reshape(-1)
    if str(sign_convention).lower() == "topographic":
        rhs = -rhs
    elif str(sign_convention).lower() != "mathematical":
        raise ValueError("sign_convention must be 'topographic' or 'mathematical'.")

    w = spsolve(A, rhs).reshape((ny_s, nx_s))

    if margin_x > 0 or margin_y > 0:
        w = w[margin_y:margin_y + ny, margin_x:margin_x + nx]
    return w

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


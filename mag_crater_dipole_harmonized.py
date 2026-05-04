#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Magnetic dipole and prism kernels used by MGM."""

import numpy as np
from numba import jit, prange

MU0 = 4e-7 * np.pi  
FOUR_PI = 4.0 * np.pi






def unit_vector_from_ID(I_deg, D_deg):
    """MGM internal routine."""
    I = np.deg2rad(I_deg)
    D = np.deg2rad(D_deg)
    ex = np.cos(I) * np.sin(D)  
    ey = np.cos(I) * np.cos(D)  
    ez = np.sin(I)              
    v = np.array([ex, ey, ez], dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Invalid I,D leading to zero vector")
    return v / n


def assign_magnetization_radial(centers, shape, dims, crater_center,
                                crater_radius, M_center, M_background,
                                r_scale_frac, z_scale, I_deg, D_deg):
    """MGM internal routine."""
    xc, yc, zc = crater_center
    u = unit_vector_from_ID(I_deg, D_deg)
    
    r2 = (centers[:,0]-xc)**2 + (centers[:,1]-yc)**2
    r = np.sqrt(r2)
    zd = np.abs(centers[:,2]-zc)
    
    sigma_r = max(1e-6, r_scale_frac * crater_radius)
    mag = M_background + (M_center - M_background) * \
          np.exp(-(r**2)/(2*sigma_r**2)) * np.exp(-zd/float(z_scale))
    
    M_vecs = mag[:,None] * u[None,:]
    return M_vecs


def total_field_anomaly_nT(Bx, By, Bz, I_deg, D_deg):
    """MGM internal routine."""
    Fhat = unit_vector_from_ID(I_deg, D_deg)
    dT_T = Bx*Fhat[0] + By*Fhat[1] + Bz*Fhat[2]
    return dT_T * 1e9  


def upward_continuation_2d(grid_2d, dx, dy, dz):
    """MGM internal routine."""
    if dz <= 0:
        return grid_2d.copy()

    nby, nbx = grid_2d.shape

    
    kx = np.fft.fftfreq(nbx, d=dx) * 2 * np.pi  
    ky = np.fft.fftfreq(nby, d=dy) * 2 * np.pi  
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    
    H = np.exp(-dz * K)

    
    F = np.fft.fft2(grid_2d)
    F_up = F * H
    grid_up = np.real(np.fft.ifft2(F_up))

    return grid_up


def compute_volumes_from_H(dx, dy, H_map):
    """MGM internal routine."""
    return (dx * dy * H_map.ravel()).astype(np.float64)






def compute_B_dipoles_vectorized(obsX, obsY, obsZ, centers, M_vecs, 
                                 volumes, eps=1e-6):
    """MGM internal routine."""
    const = MU0 / FOUR_PI
    
    Ny, Nx = obsX.shape
    
    obs_x = obsX.ravel()
    obs_y = obsY.ravel()
    obs_z = obsZ.ravel()
    
    x0 = centers[:, 0]
    y0 = centers[:, 1]
    z0 = centers[:, 2]
    
    
    m = M_vecs * volumes[:, None]
    mx = m[:, 0]
    my = m[:, 1]
    mz = m[:, 2]
    
    
    rx = obs_x[None, :] - x0[:, None]
    ry = obs_y[None, :] - y0[:, None]
    rz = obs_z[None, :] - z0[:, None]
    
    R2 = rx*rx + ry*ry + rz*rz + eps*eps
    invR = 1.0 / np.sqrt(R2)
    invR3 = invR**3
    invR5 = invR**5
    
    mdotr = mx[:, None]*rx + my[:, None]*ry + mz[:, None]*rz
    
    
    Bx_all = const * (3.0*rx*mdotr*invR5 - mx[:, None]*invR3)
    By_all = const * (3.0*ry*mdotr*invR5 - my[:, None]*invR3)
    Bz_all = const * (3.0*rz*mdotr*invR5 - mz[:, None]*invR3)
    
    
    Bx = np.sum(Bx_all, axis=0).reshape(Ny, Nx)
    By = np.sum(By_all, axis=0).reshape(Ny, Nx)
    Bz = np.sum(Bz_all, axis=0).reshape(Ny, Nx)
    
    return Bx, By, Bz






@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_B_dipoles_numba(obsX, obsY, obsZ, centers, M_vecs, 
                           volumes, eps=1e-6):
    """MGM internal routine."""
    const = MU0 / FOUR_PI
    
    Ny, Nx = obsX.shape
    N_voxels = centers.shape[0]
    
    Bx = np.zeros((Ny, Nx), dtype=np.float64)
    By = np.zeros((Ny, Nx), dtype=np.float64)
    Bz = np.zeros((Ny, Nx), dtype=np.float64)
    
    
    for iy in prange(Ny):
        for ix in range(Nx):
            x_obs = obsX[iy, ix]
            y_obs = obsY[iy, ix]
            z_obs = obsZ[iy, ix]
            
            bx_sum = 0.0
            by_sum = 0.0
            bz_sum = 0.0
            
            
            for i in range(N_voxels):
                x0 = centers[i, 0]
                y0 = centers[i, 1]
                z0 = centers[i, 2]
                
                
                V_i = volumes[i]
                mx = M_vecs[i, 0] * V_i
                my = M_vecs[i, 1] * V_i
                mz = M_vecs[i, 2] * V_i
                
                rx = x_obs - x0
                ry = y_obs - y0
                rz = z_obs - z0
                
                R2 = rx*rx + ry*ry + rz*rz + eps*eps
                invR = 1.0 / np.sqrt(R2)
                invR3 = invR * invR * invR
                invR5 = invR3 * invR * invR
                
                mdotr = mx*rx + my*ry + mz*rz
                
                
                bx_sum += const * (3.0*rx*mdotr*invR5 - mx*invR3)
                by_sum += const * (3.0*ry*mdotr*invR5 - my*invR3)
                bz_sum += const * (3.0*rz*mdotr*invR5 - mz*invR3)
            
            Bx[iy, ix] = bx_sum
            By[iy, ix] = by_sum
            Bz[iy, ix] = bz_sum
    
    return Bx, By, Bz






def compute_B_dipoles_auto(obsX, obsY, obsZ, centers, M_vecs,
                           volumes, eps=1e-6, use_numba=True):
    """MGM internal routine."""
    if use_numba:
        return compute_B_dipoles_numba(
            obsX.astype(np.float64),
            obsY.astype(np.float64),
            obsZ.astype(np.float64),
            centers.astype(np.float64),
            M_vecs.astype(np.float64),
            volumes.astype(np.float64),
            eps
        )
    else:
        return compute_B_dipoles_vectorized(
            obsX, obsY, obsZ, centers, M_vecs, volumes, eps
        )













@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_B_prism_numba(obsX, obsY, obsZ, centers, M_vecs,
                          volumes, dx, dy):
    """MGM internal routine."""
    const = MU0 / FOUR_PI

    Ny, Nx = obsX.shape
    N_prisms = centers.shape[0]

    Bx = np.zeros((Ny, Nx), dtype=np.float64)
    By = np.zeros((Ny, Nx), dtype=np.float64)
    Bz = np.zeros((Ny, Nx), dtype=np.float64)

    hdx = dx * 0.5
    hdy = dy * 0.5
    dxdy = dx * dy

    for iy in prange(Ny):
        for ix in range(Nx):
            xp = obsX[iy, ix]
            yp = obsY[iy, ix]
            zp = obsZ[iy, ix]

            bx_sum = 0.0
            by_sum = 0.0
            bz_sum = 0.0

            for ip in range(N_prisms):
                V_i = volumes[ip]
                if V_i <= 0.0:
                    continue

                cx = centers[ip, 0]
                cy = centers[ip, 1]
                cz = centers[ip, 2]

                Mx = M_vecs[ip, 0]
                My = M_vecs[ip, 1]
                Mz = M_vecs[ip, 2]

                hdz = V_i / (dxdy * 2.0)

                
                u1 = (cx - hdx) - xp
                u2 = (cx + hdx) - xp
                v1 = (cy - hdy) - yp
                v2 = (cy + hdy) - yp
                w1 = (cz - hdz) - zp
                w2 = (cz + hdz) - zp

                u = (u1, u2)
                v = (v1, v2)
                w = (w1, w2)

                
                sum_fxx = 0.0
                sum_fyy = 0.0
                sum_fzz = 0.0
                sum_fxy = 0.0
                sum_fxz = 0.0
                sum_fyz = 0.0

                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            ui = u[ii]
                            vj = v[jj]
                            wk = w[kk]

                            R = np.sqrt(ui * ui + vj * vj + wk * wk)

                            
                            sign = 1.0
                            if (ii + jj + kk) % 2 == 1:
                                sign = -1.0

                            
                            
                            denom_xx = ui * R
                            if abs(denom_xx) > 1e-30:
                                fxx = np.arctan2(vj * wk, denom_xx)
                            else:
                                fxx = 0.0

                            denom_yy = vj * R
                            if abs(denom_yy) > 1e-30:
                                fyy = np.arctan2(ui * wk, denom_yy)
                            else:
                                fyy = 0.0

                            denom_zz = wk * R
                            if abs(denom_zz) > 1e-30:
                                fzz = np.arctan2(ui * vj, denom_zz)
                            else:
                                fzz = 0.0

                            arg_xy = R + wk
                            if arg_xy > 1e-30:
                                fxy = -np.log(arg_xy)
                            else:
                                fxy = 0.0

                            arg_xz = R + vj
                            if arg_xz > 1e-30:
                                fxz = -np.log(arg_xz)
                            else:
                                fxz = 0.0

                            arg_yz = R + ui
                            if arg_yz > 1e-30:
                                fyz = -np.log(arg_yz)
                            else:
                                fyz = 0.0

                            sum_fxx += sign * fxx
                            sum_fyy += sign * fyy
                            sum_fzz += sign * fzz
                            sum_fxy += sign * fxy
                            sum_fxz += sign * fxz
                            sum_fyz += sign * fyz

                
                bx_sum += const * (Mx * sum_fxx + My * sum_fxy + Mz * sum_fxz)
                by_sum += const * (Mx * sum_fxy + My * sum_fyy + Mz * sum_fyz)
                bz_sum += const * (Mx * sum_fxz + My * sum_fyz + Mz * sum_fzz)

            Bx[iy, ix] = bx_sum
            By[iy, ix] = by_sum
            Bz[iy, ix] = bz_sum

    return Bx, By, Bz


def compute_B_prism_vectorized(obsX, obsY, obsZ, centers, M_vecs,
                               volumes, dx, dy):
    """MGM internal routine."""
    const = MU0 / FOUR_PI

    Ny, Nx = obsX.shape
    N_obs = Ny * Nx

    obs_x = obsX.ravel()
    obs_y = obsY.ravel()
    obs_z = obsZ.ravel()

    cx = centers[:, 0]
    cy = centers[:, 1]
    cz = centers[:, 2]

    dxdy = dx * dy
    hdx = dx * 0.5
    hdy = dy * 0.5
    hdz = volumes / (dxdy * 2.0)

    
    x1 = cx - hdx
    x2 = cx + hdx
    y1 = cy - hdy
    y2 = cy + hdy
    z1 = cz - hdz
    z2 = cz + hdz

    
    Bx_out = np.zeros(N_obs, dtype=np.float64)
    By_out = np.zeros(N_obs, dtype=np.float64)
    Bz_out = np.zeros(N_obs, dtype=np.float64)

    faces_x = np.stack([x1, x2], axis=0)
    faces_y = np.stack([y1, y2], axis=0)
    faces_z = np.stack([z1, z2], axis=0)

    alive = volumes > 0.0

    for ip in np.where(alive)[0]:
        Mx_i = M_vecs[ip, 0]
        My_i = M_vecs[ip, 1]
        Mz_i = M_vecs[ip, 2]

        sum_fxx = np.zeros(N_obs, dtype=np.float64)
        sum_fyy = np.zeros(N_obs, dtype=np.float64)
        sum_fzz = np.zeros(N_obs, dtype=np.float64)
        sum_fxy = np.zeros(N_obs, dtype=np.float64)
        sum_fxz = np.zeros(N_obs, dtype=np.float64)
        sum_fyz = np.zeros(N_obs, dtype=np.float64)

        for ii in range(2):
            ui = faces_x[ii, ip] - obs_x
            for jj in range(2):
                vj = faces_y[jj, ip] - obs_y
                for kk in range(2):
                    wk = faces_z[kk, ip] - obs_z

                    R = np.sqrt(ui * ui + vj * vj + wk * wk)
                    sign = 1.0 if (ii + jj + kk) % 2 == 0 else -1.0

                    denom_xx = ui * R
                    fxx = np.where(np.abs(denom_xx) > 1e-30,
                                   np.arctan2(vj * wk, denom_xx), 0.0)

                    denom_yy = vj * R
                    fyy = np.where(np.abs(denom_yy) > 1e-30,
                                   np.arctan2(ui * wk, denom_yy), 0.0)

                    denom_zz = wk * R
                    fzz = np.where(np.abs(denom_zz) > 1e-30,
                                   np.arctan2(ui * vj, denom_zz), 0.0)

                    arg_xy = R + wk
                    fxy = np.where(arg_xy > 1e-30, -np.log(arg_xy), 0.0)

                    arg_xz = R + vj
                    fxz = np.where(arg_xz > 1e-30, -np.log(arg_xz), 0.0)

                    arg_yz = R + ui
                    fyz = np.where(arg_yz > 1e-30, -np.log(arg_yz), 0.0)

                    sum_fxx += sign * fxx
                    sum_fyy += sign * fyy
                    sum_fzz += sign * fzz
                    sum_fxy += sign * fxy
                    sum_fxz += sign * fxz
                    sum_fyz += sign * fyz

        Bx_out += const * (Mx_i * sum_fxx + My_i * sum_fxy + Mz_i * sum_fxz)
        By_out += const * (Mx_i * sum_fxy + My_i * sum_fyy + Mz_i * sum_fyz)
        Bz_out += const * (Mx_i * sum_fxz + My_i * sum_fyz + Mz_i * sum_fzz)

    return Bx_out.reshape(Ny, Nx), By_out.reshape(Ny, Nx), Bz_out.reshape(Ny, Nx)


def compute_B_prism_auto(obsX, obsY, obsZ, centers, M_vecs,
                         volumes, dx, dy, use_numba=True):
    """MGM internal routine."""
    if use_numba:
        return compute_B_prism_numba(
            obsX.astype(np.float64),
            obsY.astype(np.float64),
            obsZ.astype(np.float64),
            centers.astype(np.float64),
            M_vecs.astype(np.float64),
            volumes.astype(np.float64),
            float(dx), float(dy),
        )
    else:
        return compute_B_prism_vectorized(
            obsX, obsY, obsZ, centers, M_vecs, volumes,
            float(dx), float(dy),
        )

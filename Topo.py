#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Topography generation and crater-geometry utilities used by MGM."""

import numpy as np
from scipy.signal import savgol_filter, find_peaks
import ruptures as rpt
from scipy.ndimage import gaussian_filter1d

def generate_topography(file_path, dx, max_distance):

    crater_profile = np.loadtxt(file_path, delimiter=',')

    distances = crater_profile[:, 1]*1000
    heights   = crater_profile[:, 0]

    
    window_length = 21  
    polyorder     = 2   
    
    heights = savgol_filter(heights, window_length=window_length, polyorder=polyorder)

    
    grid_size = int(2 * max_distance / dx) + 1

    
    x_coords = np.linspace(-max_distance, max_distance, grid_size)
    y_coords = np.linspace(-max_distance, max_distance, grid_size)
    x_coords, y_coords = np.meshgrid(x_coords, y_coords)

    
    radial_distances = np.sqrt(x_coords**2 + y_coords**2)

    topography_heights = np.zeros_like(radial_distances)

    
    for i in range(len(distances)):
        topography_heights[radial_distances >= distances[i]] = heights[i]
    
    topography_heights = topography_heights*1000

    return topography_heights, x_coords, y_coords

def generate_gravi(file_path, dx, max_distance):

    crater_profile = np.loadtxt(file_path, delimiter=',')

    distances = crater_profile[:, 1]*1000
    heights   = crater_profile[:, 0]

    
    grid_size = int(2 * max_distance / dx) + 1

    
    x_coords = np.linspace(-max_distance, max_distance, grid_size)
    y_coords = np.linspace(-max_distance, max_distance, grid_size)
    x_coords, y_coords = np.meshgrid(x_coords, y_coords)

    
    radial_distances = np.sqrt(x_coords**2 + y_coords**2)

    topography_heights = np.zeros_like(radial_distances)

    
    for i in range(len(distances)):
        topography_heights[radial_distances >= distances[i]] = heights[i]

    return topography_heights

def detect_crater_ridges(profile, height, distance, prominence, xy_space):
    
    peaks, properties = find_peaks(profile, height=height, distance=distance, prominence=prominence)
    
    if len(peaks) >= 2:
        
        return (peaks[-1] - peaks[0]) * xy_space
    return 0


def detect_crater_diameter_hybrid(profile, height, distance, prominence, xy_space,
                                    sigma=3, curvature_thresh=0.1, peak_distance=10):
  
    
    peaks, _ = find_peaks(profile, height=height, distance=distance, prominence=prominence)
    if len(peaks) >= 2:
        return (peaks[-1] - peaks[0]) * xy_space

    
    
    profile_smooth = gaussian_filter1d(profile, sigma=sigma)
    
    
    grad = np.gradient(profile_smooth)
    
    curvature = np.gradient(grad)
    
    
    curvature_abs = np.abs(curvature)
    peaks_curv, _ = find_peaks(curvature_abs, height=curvature_thresh, distance=peak_distance)
    
    if len(peaks_curv) < 2:
        return 0  
    
    left_boundary = peaks_curv[0]
    right_boundary = peaks_curv[-1]
    
    return (right_boundary - left_boundary) * xy_space

def detect_crater_diameter_curvature(profile, xy_space, sigma=3, curvature_thresh=0.1, peak_distance=10):

    
    profile_smooth = gaussian_filter1d(profile, sigma=sigma)
    
    
    grad = np.gradient(profile_smooth)
    curvature = np.gradient(grad)
    
    
    peaks, properties = find_peaks(np.abs(curvature), height=curvature_thresh, distance=peak_distance)
    
    if len(peaks) < 2:
        return 0  
    
    
    left_boundary = peaks[0]
    right_boundary = peaks[-1]
    
    crater_diameter = (right_boundary - left_boundary) * xy_space
    return crater_diameter

def detect_crater_diameter_sign_change(profile, d_thresh, xy_space):

    
    grad = np.gradient(profile)
    
    sign_grad = np.sign(grad)
    
    
    diff_sign = np.diff(sign_grad)
    
    
    sign_change_indices = np.where(diff_sign != 0)[0]
    
    
    
    significant_changes = []
    for idx in sign_change_indices:
        if abs(grad[idx]) >= d_thresh or abs(grad[idx+1]) >= d_thresh:
            significant_changes.append(idx)
    
    
    if len(significant_changes) < 2:
        return 0
    
    
    
    left_boundary = significant_changes[0]
    right_boundary = significant_changes[-1] + 1  
    
    diameter = (right_boundary - left_boundary) * xy_space
    return diameter

def detect_crater_diameter(profile, height, distance, prominence, xy_space, plateau=0, tol=0.05):

    
    peaks, properties = find_peaks(profile, height=height, distance=distance, prominence=prominence)
    
    if len(peaks) >= 2:
        
        return (peaks[-1] - peaks[0]) * xy_space
    
    
    
    plateau_indices = np.where(np.abs(profile - plateau) < tol)[0]
    
    if len(plateau_indices) == 0:
        return 0  
    
    
    min_idx = np.argmin(profile)
    
    
    left_candidates = plateau_indices[plateau_indices < min_idx]
    right_candidates = plateau_indices[plateau_indices > min_idx]
    
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return 0  
    
    left_boundary = left_candidates[-1]
    right_boundary = right_candidates[0]
    
    return (right_boundary - left_boundary) * xy_space


def creer_masque_circulaire(nx, ny, dx, dy, centre=None, rayon_m=13000):
    """MGM internal routine."""
    if centre is None:
        centre = (ny // 2, nx // 2)  
    
    
    y_indices, x_indices = np.ogrid[:ny, :nx]
    
    
    distances = np.sqrt(((x_indices - centre[1]) * dx)**2 + ((y_indices - centre[0]) * dy)**2)
    
    
    masque = distances <= rayon_m
    return masque

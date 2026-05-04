#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-level gravity helper routines used by MGM."""

import numpy as np 
import math


def anomaly(x0,y0,z0,x1,y1,z1,x2,y2,z2,rho):
    
    isign = [-1.0,1.0]
    gamma = 6.673e-11
    twopi = 2 * np.pi
    si2mg = 1e5
    
    x = np.zeros(len(isign))
    y = np.zeros(len(isign))
    z = np.zeros(len(isign))
    
    x[0] = x0 - x1 
    y[0] = y0 - y1
    z[0] = z0 - z1
    x[1] = x0 - x2
    y[1] = y0 - y2
    z[1] = z0 - z2

    som = 0
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                rijk = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                ijk  = isign[i] * isign[j] * isign[k]
                arg1 = math.atan2((x[i]*y[j]) , (z[k]*rijk))
                
                if arg1 < 0:
                    arg1 = arg1 + twopi
                else:
                    arg1 = arg1
                
                arg2 = np.log(rijk + y[j])
                arg3 = np.log(rijk + x[i])
                
                som += (z[k]*arg1 - x[i]*arg2 - y[j]*arg3) * ijk
                
                if np.isnan(som)==True:
                    som = 0
                
    g = rho*gamma*som*si2mg
    
    return g

    
    
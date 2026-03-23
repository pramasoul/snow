# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:29:38 2020

@author: Luis Ledezma
"""

import numpy as np

def sech(x):
    return 1/np.cosh(x)

def check(value, ll, ul):
    if ll <= value <= ul:
        return True
    return False

def absorption_coeff(Alpha):
    # convert from dB/cm to 1/m
    alpha = np.log((10**(Alpha * 0.1))) * 100
    return alpha

def fwhm_interp(t, x):
    """Interpolated FWHM of |x|² using linear interpolation at half-max crossings.

    Returns the full width at half maximum in the same units as t.
    Uses linear interpolation between grid points to achieve sub-dt
    resolution, avoiding the staircase quantization of bin-counting.

    Returns np.nan if the pulse has no clear peak above the noise floor.
    """
    I = np.abs(x)**2
    peak = np.max(I)
    if peak <= 0:
        return np.nan
    half = peak / 2
    above = I >= half
    if np.sum(above) < 2:
        return np.nan

    # Find first and last crossings by linear interpolation
    idxs = np.where(above)[0]
    i_lo, i_hi = idxs[0], idxs[-1]

    # Interpolate left crossing (between i_lo-1 and i_lo)
    if i_lo > 0 and I[i_lo - 1] < half:
        frac = (half - I[i_lo - 1]) / (I[i_lo] - I[i_lo - 1])
        t_left = t[i_lo - 1] + frac * (t[i_lo] - t[i_lo - 1])
    else:
        t_left = t[i_lo]

    # Interpolate right crossing (between i_hi and i_hi+1)
    if i_hi < len(I) - 1 and I[i_hi + 1] < half:
        frac = (half - I[i_hi + 1]) / (I[i_hi] - I[i_hi + 1])
        t_right = t[i_hi + 1] - frac * (t[i_hi + 1] - t[i_hi])
    else:
        t_right = t[i_hi]

    return t_right - t_left


def derivative( f, x, n, h ):
    """Richardson's Extrapolation to approximate  f'(x) at a particular x.

    USAGE:
	d = richardson( f, x, n, h )

    INPUT:
	f	- function to find derivative of
	x	- value of x to find derivative at
	n	- number of levels of extrapolation
	h	- initial stepsize

    OUTPUT:
        numpy float array -  two-dimensional array of extrapolation values.
                             The [n,n] value of this array should be the
                             most accurate estimate of f'(x).

    NOTES:                             
        Based on an algorithm in "Numerical Mathematics and Computing"
        4th Edition, by Cheney and Kincaid, Brooks-Cole, 1999.

    AUTHOR:
        Jonathan R. Senning <jonathan.senning@gordon.edu>
        Gordon College
        February 9, 1999
        Converted ty Python August 2008
    """

    # d[n,n] will contain the most accurate approximation to f'(x).

    d = np.array( [[0] * (n + 1)] * (n + 1), float )

    for i in range( n + 1 ):
        val = 0.5 * ( f( x + h ) - f( x - h ) ) / h
        d[i,0] = val.item() if hasattr(val, 'item') else val

        powerOf4 = 1  # values of 4^j
        for j in range( 1, i + 1 ):
            powerOf4 = 4 * powerOf4
            d[i,j] = d[i,j-1] + ( d[i,j-1] - d[i-1,j-1] ) / ( powerOf4 - 1 )

        h = 0.5 * h

    return d[n,n]
"""
fitscripts.py:
Script that contains fitting routines for two and three-point functions

***TODO***
"""


import sys
import functools as ft
import itertools as it
import numpy as np
import scipy.optimize as opt
import lstats
import fitfuncs as fit
import setup_lat as sl


T = sl.T/2.0


#-------------------- C2pt --------------------#
# Effective Mass Plots
def eff_func(E, data, t):
    """ returns data to be minimized for effective mass plot """
    dat = data.mean(0)
    return abs(dat[t+1]/dat[t]-np.cosh(E*(T-t+1))/np.cosh(E*(T-t)))


def get_eff(data, trange, lims):
    """ computes effective mass at given time slice """
    fargs = [(data, t) for t in trange]
    tres, results = [], []
    for arg in fargs:
        try:
            results.append(opt.brentq(eff_func, lims[0], lims[1], arg))
            tres.append(arg[-1])
        except RuntimeError:
            results.append(None)
            tres.append(None)
    return tres, results


def get_meff(data, trange, nbins, lims):
    """ computes effective mass at given time slice over all configs """
    coeff = (nbins-1.0)/nbins
    jk_dat = lstats.jk_blocks(data, nbins)
    tres, results = get_eff(data, trange, lims)
    results_blocks = []
    for jkdat in jk_dat:
        jktres, jkresults = get_eff(jkdat, trange, lims)
        results_blocks.append(jkresults)
        tres = np.vstack((tres, jktres))
    jkresults = np.asarray(jkresults)

    ntrange = tuple(t for t in trange if None not in tres[:, t])
    results = results[ntrange]
    jkresults = jkresults[:, ntrange]
    ntrange = np.asarray(ntrange)
    err = np.sqrt(coeff*np.sum((jkresults-results)**2, 0))

    return ntrange, results, err


# Fit two-point function
def fit_2pt(data, lims, nbins, p0=None, fit_p0=None, **kwargs):
    """ Multi-Exponential Fit for Two-Point Function """
    trange, dof = np.arange(lims[0], lims[1]), lims[1]-lims[0]
    data = data[:, lims[0], lims[1]]
    mean, cov = lstats.jk_mean_cov(data, nbins)
    func = fit.c2pt if fit_p0 is None else fit.fix_param(fit.c2pt, fit_p0)

    try:
        popt, pcov = opt.curve_fit(func, trange, mean, p0=p0, sigma=cov, **kwargs)
    except Exception as err:
        print(err)
        return None

    res = mean - func(trange, *popt)
    chisq = res.T.dot(np.linalg.inv(cov).dot(res))/dof
    return popt, chisq


def fit_c2pt(data, lims, nbins, p0=None, fit_p0=None, **kwargs):
    """ Multi-Exponential Fit of c2pt for full data set """
    coeff = (nbins-1.0)/nbins
    dblocks = lstats.jk_blocks(data, nbins)
    popt, chisq = fit_2pt(data, lims, nbins, p0=p0, fit_p0=fit_p0, **kwargs)

    popt_blocks, chisq_blocks = [], []
    try:
        for dblk in dblocks:
            pjk, chijk = fit_2pt(dblk, lims, nbins, p0=popt, fit_p0=fit_p0, **kwargs)
            popt_blocks.append(pjk)
            chisq_blocks.append(chisq)

    except Exception as err:
        print(err)
        return None

    popt_err = np.sqrt(coeff*np.sum((popt_blocks-popt)**2, 0))
    chisq_err = np.sqrt(coeff*np.sum((chisq_blocks-chisq)**2))

    return (popt, popt_err), (chisq, chisq_err)


#-------------------- C3pt --------------------#
def fit_3pt(data, dts, taus_dt, fix_pars, struct, p0=None, **kwargs):
    xs = []
    for dt, taus in zip(dts, taus_dt):
        xs.append([[dt, tau] for tau in taus])
    xs = np.asarray(xs)

"""
lstats.py:
--------------------
Statistics module for (auto-correlated) lattice data.
Mainly deals with computing means, variance via regular
way and also via jackknife resampling.

***TODO***
--------------------
Add bootstrap resampling
Add numba jit to make it faster?

"""


import numpy as np


# All for 2d data sets
def mean_cov(data, axis=0):
    """ Computes Mean and Covariance Matrix of 2D data """
    assert data.ndim == 2
    Ncf, Nt = data.shape
    mean = data.mean(axis)
    cov = np.zeros((Ncf, Nt, Nt), dtype=data.dtype)
    for j in range(Ncf):
        cov[j] = np.outer((mean-data[j]), (mean-data[j].conj()))
    cov = cov.sum(0)/(Ncf-1.0)
    return mean, cov


def mean_var(data, axis=0):
    """ Computes Mean and Variance of 2D data """
    mean, cov = mean_cov(data, axis)
    var = cov.diagonal()
    return mean, var


# Now looking at Jackknife Samples
def jk_blocks(data, nbins):
    """ Creates Jackknife Blocks of data across axis=0 """
    Ncf = data.shape[0]
    assert Ncf/nbins % 0 == 0
    take_out = int(Ncf/nbins)
    newshape = tuple([nbins, Ncf - take_out] + list(data.shape[1:]))
    jk_dat = np.zeros(newshape, dtype=data.dtype)

    delinds = np.arange(0, Ncf, nbins)
    ct = 0
    while delinds[-1] < Ncf:
        jk_dat[ct] = np.delete(data, delinds, axis=0)
        ct += 1
        delinds += 1

    return jk_dat


def jk_mean_cov(data, nbins):
    """ Computes jk-bias-corrected mean and jackknife covariance matrix """
    mean = data.mean(0)
    jkdat = jk_blocks(data, nbins)
    jkdat = jkdat.mean(1)
    jkdat_mean = jkdat.mean(0)

    jkcov = np.array([
        np.outer((jkd-jkdat_mean).conj(), (jkd-jkdat_mean)) for jkd in jkdat
    ])
    jkmean = nbins*mean - (nbins-1)*jkdat_mean(0)

    return jkmean, jkcov


def jk_mean_var(data, nbins):
    """ Computes jk-bias-corrected mean and jackknife variance """
    jkmean, jkcov = jk_mean_cov(data, nbins)
    return jkmean, jkcov.diagonal()


def jk_mean_err(datA, datB, nbins, tinsert=None):
    """
    Computes jk-bias corrected mean and jackknife var of ratio of two data sets
    """
    if tinsert is not None:
        datB = datB[:, tinsert]

    ratio = datA.mean(0)/datB.mean(0)

    jkA, jkB = jk_blocks(datA, nbins), jk_blocks(datB, nbins)
    rblocks = np.array([jA.mean(0)/jB.mean(0) for jA, jB in zip(jkA, jkB)])
    jk_corrected_mean = nbins*ratio - (nbins-1.0)*rblocks
    coeff = (nbins-1.0)/nbins
    rerr_real = np.sqrt(coeff*np.sum((jkA.real-ratio.real)**2, 0))
    rerr_imag = np.sqrt(coeff*np.sum((jkA.imag-ratio.imag)**2, 0))
    rerr = rerr_real + 1j*rerr_imag
    return jk_corrected_mean, rerr

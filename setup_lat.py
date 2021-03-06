"""
setup_lat.py:
lattice info used for all analysis

"""


import numpy as np


L = np.array([48.0, 48.0, 48.0, 64.0])
a = 0.06  # fm
hbarc = 0.197/a  # GeV
fct = 2*np.pi/L[0]
T = L[-1]
mpi = 0.3/hbarc  # lattice units


#-------------------- Global Functions --------------------#
def dispers(pvec):
    """ Dispersion Relation for Hadron at momentum pvec = [px, py, pz]"""
    pvec = pvec*fct
    return np.sqrt(np.dot(pvec, pvec) + mpi**2)


#-------------------- Global Tables --------------------#
gamma_table = {
    '1': 'g0',
    'gx': 'g1', 'gy': 'g2', 'gz': 'g4', 'gt': 'g8',
    'g5': 'g15',
    'gxgt': 'g9', 'gygt': 'g10', 'gzgt': 'g12',
    'gxgz': 'g5', 'gygz': 'g6',
    'g5gt': 'g7', 'g5gz': 'g11'
}

gph_table = {
    '1': 1, 'gx': -1, 'gy': -1, 'gz': -1, 'gt': 1, 'g5': 1,
    'gxgt': 1, 'gygt': 1, 'gzgt': 1, 'gxgz': -1, 'gygz': -1,
}

import numpy as np
import pandas as pd
from scipy.stats import gamma

def status_to_state_(status):
    return ((status > 0) and (status < 6)) + 2 * (status >=6)

status_to_state = np.vectorize(status_to_state_)

def listofhouses(houses):
    housedict = {house_no : [] for house_no in np.unique(houses)}
    for i in range(len(houses)):
        housedict[houses[i]].append(i)
    return housedict


class dummy_logger():
    def __init__(self):
        self.description = "This shape has not been described yet"
    def info(self, s):
        return True


def quarantine_households(idx,quarantine,houses,housedict,verbose = True):
    if not quarantine:
        return []
    out = []
    for i in idx:
        out += list(filter(lambda x: x not in idx, housedict[houses[i]]))
    return list(set(out))


def gamma_params(mn, sd):
    scale = (sd**2)/mn
    shape = mn/scale

    return(shape, scale)

def gamma_pdf_array(T, mu, sigma):
    """
    discrete gamma function:
    T: len(array) = T+1
    mu: mu of gamma
    sigma: std of gammas
    """
    k, scale = gamma_params(mu, sigma)
    gamma_array = gamma.pdf(range(T+1), k, scale=scale)
    #def sym_delay(delta_t):
    #    return gamma_delay[delta_t]
    return gamma_array

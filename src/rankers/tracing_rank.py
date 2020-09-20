import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from .template_rank import AbstractRanker
from .mean_field_rank import records_to_csr, get_rank, check_inputs


def csr_to_list(x):
    x_coo = x.tocoo()
    return zip(x_coo.row, x_coo.col, x_coo.data)


def ranking_tracing(t, transmissions, observations, tau):
    """Naive contact tracing.

    Search for all individuals that have been in contact during [t-tau, t]
    with the individuals last tested positive (observations s=I at
    t-tau <= t_test < t) and count the number of encounters.

    Returns: scores = encounters. If t < delta returns random scores.
    """
    N = transmissions[0].shape[0]
    if (t < tau):
        scores = np.random.rand(N)
        return scores
    # last_tested : observations s=I for t-tau <= t_test < t
    last_tested = set(
        obs["i"] for obs in observations
        if obs["s"] == 1 and (t - tau <= obs["t_test"]) and (obs["t_test"] < t)
    )
    # contacts with last_tested people during [t - tau, t]
    contacts = pd.DataFrame(
        dict(i=i, j=j, t=t_contact)
        for t_contact in range(t - tau, t)
        for i, j, lamb in csr_to_list(transmissions[t_contact])
        if j in last_tested and lamb # lamb = 0 does not count
    )
    encounters = pd.DataFrame({"i": range(N)})
    # no encounters -> count = 0
    if (contacts.shape[0] == 0):
        encounters["count"] = 0
    else:
        counts = contacts.groupby("i").size() # number of encounters for all i
        encounters["count"] = encounters["i"].map(counts).fillna(0)
    scores = encounters["count"].values
    return scores


class TracingRanker(AbstractRanker):

    def __init__(self, tau, lamb):
        self.description = "class for naive contact tracing of openABM loop."
        self.author = "https://github.com/sphinxteam"
        self.tau = tau
        self.lamb = lamb

    def init(self, N, T):
        self.transmissions = []
        self.observations = []
        self.T = T
        self.N = N
        return True

    def rank(self, t_day, daily_contacts, daily_obs, data):
        '''
        computing rank of infected individuals
        return: list -- [(index, value), ...]
        '''
        # check that t=t_day in daily_contacts and t=t_day-1 in daily_obs
        check_inputs(t_day, daily_contacts, daily_obs)
        # append daily_contacts and daily_obs
        daily_transmissions = records_to_csr(self.N, daily_contacts, self.lamb)
        self.transmissions.append(daily_transmissions)
        self.observations += [
            dict(i=i, s=s, t_test=t_test) for i, s, t_test in daily_obs
        ]
        # scores given by mean field run from t-delta to t
        scores = ranking_tracing(
            t_day, self.transmissions, self.observations, self.tau
        )
        # convert to list [(index, value), ...]
        rank = get_rank(scores)
        return rank

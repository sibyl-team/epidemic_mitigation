## TEMPLATE for the class of the algorithm that rank indivuals to be tested

class AbstractRanker():

    def __init__(self):
        self.description = "This shape has not been described yet"
        self.author = "Nobody has claimed to make this shape yet"
 
    def init(self, N, T):
        raise NotImplementedError
        
    def rank(self, t, daily_contacts, daily_obs, data):
        '''
        Order the individuals by the probability to be infected
        
        input
        ------
        t: int - 
            day of rank
        
        daily_contacts: list (i, j, t, value)
            list of daily contacts
        daily_obs: list (i, state, t)
            list of daily observations

        return
        ------
        list -- [(index, value), ...]
        '''
        raise NotImplementedError

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import covid19
from COVID19.model import AgeGroupEnum, EVENT_TYPES, TransmissionTypeEnum
from COVID19.model import Model, Parameters, ModelParameterException
import COVID19.simulation as simulation

from analysis_utils import ranker_I, check_fn_I, ranker_IR, check_fn_IR, roc_curve, events_list
from abm_utils import status_to_state, listofhouses, dummy_logger, quarantine_households
#import sib
#import greedy_Rank

def loop_abm(params,
             inference_algo,
             logger = dummy_logger(),
             input_parameter_file = "./abm_params/baseline_parameters.csv",
             household_demographics_file = "./abm_params/baseline_household_demographics.csv",
             parameter_line_number = 1,
             seed=1,
             initial_steps = 0,
             num_test_random = 50,
             num_test_algo = 50,
             fraction_SM_obs = 0.2,
             fraction_SS_obs = 1,
             quarantine_HH = False,             
             test_HH = False,
             name_file_res = "res",
             output_dir = "./output/",
             save_every_iter = 5,
             stop_zero_I = True,
             adoption_fraction = 1.0,
             fp_rate = 0.0,
             fn_rate = 0.0,
             smartphone_users_abm = False, # if True use app users fraction from OpenABM model
             callback = lambda x : None,
             data = {}
            ):
    '''
    Simulate interventions strategy on the openABM epidemic simulation.

    input
    -----
    params: Dict
            Dictonary with openABM to set
    inference_algo: Class (rank_template)
            Class for order the nodes according to the prob to be infected
            logger = logger for printing intermediate steps
    results:
        print on file true configurations and transmission
    '''

    params_model = Parameters(input_parameter_file,
                          parameter_line_number,
                          output_dir,
                          household_demographics_file)
    
    ### create output_dir if missing
    fold_out = Path(output_dir)
    if not fold_out.exists():
        fold_out.mkdir(parents=True)

    ### initialize a separate random stream
    rng = np.random.RandomState()
    rng.seed(seed)
    
    ### initialize ABM model
    for k, val in params.items():
        params_model.set_param(k, val)
    model = Model(params_model)
    model = simulation.COVID19IBM(model=model)

    
    T = params_model.get_param("end_time")
    N = params_model.get_param("n_total")
    sim = simulation.Simulation(env=model, end_time=T, verbose=False)
    house = covid19.get_house(model.model.c_model)
    housedict = listofhouses(house)
    has_app = covid19.get_app_users(model.model.c_model) if smartphone_users_abm else np.ones(N,dtype = int)
    has_app &= (rng.random(N) <= adoption_fraction)    

    ### init data and data_states
    data_states = {}
    data_states["true_conf"] = np.zeros((T,N))
    data_states["statuses"] = np.zeros((T,N))
    data_states["tested_algo"] = []
    data_states["tested_random"] = []
    data_states["tested_SS"] = []
    data_states["tested_SM"] = []
    for name in ["num_quarantined", "q_SS", "q_SM", "q_algo", "q_random", "q_all", "infected_free", "S", "I", "R", "IR", "aurI", "prec1%", "prec5%", "test_+", "test_-", "test_f+", "test_f-"]:
        data[name] = np.full(T,np.nan)
    data["logger"] = logger

    
    ### init inference algo
    inference_algo.init(N, T)
    
    ### running variables
    indices = np.arange(N, dtype=int)
    excluded = np.zeros(N, dtype=bool)
    daily_obs = []
    all_obs = []
    all_quarantined = []
    freebirds = 0
    num_quarantined = 0
    fp_num = 0
    fn_num = 0
    p_num = 0
    n_num = 0
    
    noise_SM = rng.random(N)
    nfree = params_model.get_param("n_seed_infection")
    for t in range(T):
        ### advance one time step
        sim.steps(1)
        status = np.array(covid19.get_state(model.model.c_model))
        state = status_to_state(status)
        data_states["true_conf"][t] = state
        nS, nI, nR = (state == 0).sum(), (state == 1).sum(), (state == 2).sum()
        if nI == 0 and stop_zero_I:
            logger.info("stopping simulation as there are no more infected individuals")
            break
        if t == initial_steps:
            logger.info("\nobservation-based inference algorithm starts now\n")
        logger.info(f'time:{t}')

        ### extract contacts
        daily_contacts = covid19.get_contacts_daily(model.model.c_model, t)
        logger.info(f"number of unique contacts: {len(daily_contacts)}")

        
        
        ### compute potential test results for all
        if fp_rate or fn_rate:
            noise = rng.random(N)
            f_state = (state==1)*(noise > fn_rate) + (state==0)*(noise < fp_rate) + 2*(state==2)
        else:
            f_state = state
        
        to_quarantine = []
        all_test = []
        excluded_now = excluded.copy()
        fp_num_today = 0
        fn_num_today = 0
        p_num_today = 0
        n_num_today = 0
        def test_and_quarantine(rank, num):
            nonlocal to_quarantine, excluded_now, all_test, fp_num_today, fn_num_today, p_num_today, n_num_today
            test_rank = []
            for i in rank:
                if len(test_rank) == num:
                    break;
                if excluded_now[i]:
                    continue
                test_rank += [i]
                if f_state[i] == 1:
                    p_num_today += 1
                    if state[i] != 1:
                        fp_num_today += 1
                    q = housedict[house[i]] if quarantine_HH else [i]
                    excluded_now[q] = True
                    to_quarantine += q
                    excluded[q] = True
                    if test_HH:
                        all_test += q
                    else:
                        all_test += [i]
                else:
                    n_num_today += 1
                    if state[i] == 1:
                        fn_num_today += 1
                    excluded_now[i] = True
                    all_test += [i]
            return test_rank
        
        ### compute rank from algorithm
        num_test_algo_today = num_test_algo
        if t < initial_steps:
            daily_obs = []
            num_test_algo_today = 0            
        
        weighted_contacts = [(c[0], c[1], c[2], 2.0 if c[3] == 0 else 1.0) for c in daily_contacts if (has_app[c[0]] and has_app[c[1]])]
        if nfree == 0 and quarantine_HH:
            print("faster end")
            rank_algo = np.zeros((N,2))
            rank_algo[:, 0]=np.arange(N)
            rank_algo[:, 1]=np.random.rand(N)
        else:
            rank_algo = inference_algo.rank(t, weighted_contacts, daily_obs, data)
        rank = np.array(sorted(rank_algo, key= lambda tup: tup[1], reverse=True))
        rank = [int(tup[0]) for tup in rank]
        
        ### test num_test_algo_today individuals
        test_algo = test_and_quarantine(rank, num_test_algo_today)

        ### compute roc now, only excluding past tests
        eventsI = events_list(t, [(i,1,t) for (i,tf) in enumerate(excluded) if tf], data_states["true_conf"], check_fn = check_fn_I)
        xI, yI, aurI, sortlI = roc_curve(dict(rank_algo), eventsI, lambda x: x)
        
        ### test all SS
        SS = test_and_quarantine(indices[status == 4], N)
        
        ### test a fraction of SM
        SM = indices[(status == 5) & (noise_SM < fraction_SM_obs)]
        SM = test_and_quarantine(SM, len(SM))

        ### do num_test_random extra random tests
        test_random = test_and_quarantine(rng.permutation(N), num_test_random)

        ### quarantine infected individuals
        num_quarantined += len(to_quarantine)
        covid19.intervention_quarantine_list(model.model.c_model, to_quarantine, T+1)
            
        ### update observations
        daily_obs = [(int(i), int(f_state[i]), int(t)) for i in all_test]
        all_obs += daily_obs

        ### exclude forever nodes that are observed recovered
        rec = [i[0] for i in daily_obs if f_state[i[0]] == 2]
        excluded[rec] = True

        ### update data 
        data_states["tested_algo"].append(test_algo)
        data_states["tested_random"].append(test_random)
        data_states["tested_SS"].append(SS)
        data_states["tested_SM"].append(SM)
        data_states["statuses"][t] = status
        data["S"][t] = nS
        data["I"][t] = nI
        data["R"][t] = nR
        data["IR"][t] = nR+nI
        data["aurI"][t] = aurI
        prec = lambda f: yI[int(f/100*len(yI))]/int(f/100*len(yI)) if len(yI) else np.nan
        ninfq = sum(state[to_quarantine]>0)
        nfree = int(nI - sum(excluded[state == 1]))
        data["aurI"][t] = aurI
        data["prec1%"][t] = prec(1)
        data["prec5%"][t] = prec(5)
        data["num_quarantined"][t] = num_quarantined
        data["test_+"][t] = p_num
        data["test_-"][t] = n_num
        data["test_f+"][t] = fp_num
        data["test_f-"][t] = fn_num
        data["q_SS"][t] = len(SS)
        data["q_SM"][t] = len(SM)
        sus_test_algo = sum(state[test_algo]==0)
        inf_test_algo = sum(state[test_algo]==1)
        rec_test_algo = sum(state[test_algo]==2)
        inf_test_random = sum(state[test_random]==1)
        data["q_algo"][t] = inf_test_algo
        data["q_random"][t] = sum(state[test_random]==1)
        data["infected_free"][t] = nfree
        asbirds = 'a bird' if nfree == 1 else 'birds'

        fp_num += fp_num_today
        fn_num += fn_num_today
        n_num += n_num_today
        p_num += p_num_today
        
        ### show output
        logger.info(f"True  : (S,I,R): ({nS:.1f}, {nI:.1f}, {nR:.1f})")
        logger.info(f"AUR_I : {aurI:.3f}, prec(1% of {len(yI)}): {prec(1):.2f}, prec5%: {prec(5):.2f}")
        logger.info(f"SS: {len(SS)}, SM: {len(SM)}, results test algo (S,I,R): ({sus_test_algo},{inf_test_algo},{rec_test_algo}), infected test random: {inf_test_random}/{num_test_random}")
        logger.info(f"false+: {fp_num} (+{fp_num_today}), false-: {fn_num} (+{fn_num_today})")
        logger.info(f"...quarantining {len(to_quarantine)} guys -> got {ninfq} infected, {nfree} free as {asbirds} ({nfree-freebirds:+d})")
        freebirds = nfree

        ### callback
        callback(data)

        if t % save_every_iter == 0:
            df_save = pd.DataFrame.from_records(data, exclude=["logger"])
            df_save.to_csv(output_dir + name_file_res + "_res.gz")

    # save files
    df_save = pd.DataFrame.from_records(data, exclude=["logger"])
    df_save.to_csv(output_dir + name_file_res + "_res.gz")
    with open(output_dir + name_file_res + "_states.pkl", mode="wb") as f_states:
        pickle.dump(data_states, f_states)
    sim.env.model.write_individual_file()
    df_indiv = pd.read_csv(output_dir+"individual_file_Run1.csv", skipinitialspace = True)
    df_indiv.to_csv(output_dir+name_file_res+"_individuals.gz")
    sim.env.model.write_transmissions()
    df_trans = pd.read_csv(output_dir+"transmission_Run1.csv")
    df_trans.to_csv(output_dir + name_file_res+"_transmissions.gz")
    return df_save





def free_abm(params,
             logger = dummy_logger(),
             input_parameter_file = "./abm_params/baseline_parameters.csv",
             household_demographics_file = "./abm_params/baseline_household_demographics.csv",
             parameter_line_number = 1,
             name_file_res = "res",
             output_dir = "./output/",
             save_every_iter = 5,
             stop_zero_I = True,
             data = {},
             callback = lambda data : None
            ):    
    '''
    Simulate the openABM epidemic simulation with no intervention strategy

    input
    -----
    params: Dict
            Dictonary with openABM to set
    results:
        print on file true configurations and transmission
    '''
    
    params_model = Parameters(input_parameter_file,
                          parameter_line_number,
                          output_dir,
                          household_demographics_file)
    
    fold_out = Path(output_dir)
    if not fold_out.exists():
        fold_out.mkdir(parents=True)

    params_model = Parameters()
    for k, val in params.items():
        params_model.set_param(k, val)
    model = simulation.COVID19IBM(model = Model(params_model))
    
    T = params_model.get_param("end_time")
    N = params_model.get_param("n_total")
    sim = simulation.Simulation(env=model, end_time=T, verbose=False)
    for col_name in ["I","IR"]:
        data[col_name] = np.full(T,np.nan)
    for t in range(T):
        sim.steps(1)
        status = np.array(covid19.get_state(model.model.c_model))
        state = status_to_state(status)
        nS, nI, nR = (state == 0).sum(), (state == 1).sum(), (state == 2).sum()
        if nI == 0 and stop_zero_I:
            logger.info("stopping simulation as there are no more infected individuals")
            break
        logger.info(f'time:{t}')

        data["I"][t] = nI
        data["IR"][t] = nI + nR

        callback(data)

    # print and import files
    sim.env.model.write_individual_file()
    df_indiv = pd.read_csv(output_dir+"individual_file_Run1.csv", skipinitialspace = True)
    df_indiv.to_csv(output_dir+name_file_res+"_individuals.gz")
    sim.env.model.write_transmissions()
    df_trans = pd.read_csv(output_dir+"transmission_Run1.csv")
    df_trans.to_csv(output_dir + name_file_res+"_transmissions.gz")
    print("End of Simulation")
    return 

import SIP
import numpy as np
import model

if __name__ == "__main__":

    # todo: get this input from a file

    # initialize model
    # interaction matrix
    A = [[-3.0, -1.0], [-1.0, -2.0]]
    # growth rates
    r = [5.0, 3.0]
    # species concentrations at t=0
    initial_conc = [0.25, 0.5]
    # number of species to model in partial LV
    reduced_size = 1
    max_time = 1.1
    t_span = (0.0, max_time)
    timestep = 0.1
    induction_time = 0.0
    times = np.arange(induction_time, max_time, timestep)
    obs_error = 0.001
    glv_obj = model.glv(A, r, initial_conc, reduced_size, t_span, times, obs_error)
    obs = glv_obj.get_concentrations()[:,0:reduced_size].flatten()
    print(obs)

    #detailed vals
    # y = get_concentrations(r,A, initial_conc)
    yobs = glv_obj.get_detailed_obs()
    np.savetxt("inputs/datafile.txt", obs)

    # reduced vals
    red_y = glv_obj.get_reduced_obs()
    np.savetxt("inputs/datafile-reduced.txt", red_y)

    # SIP: calibration
    # number of MCMC steps
    steps = 10000
    initial_thetas = [-0.96]
    initial_phis = np.array([-1.0, 0.01])
    # "mass matrix" aka initial proposal covariance matirx
    # M = [[1, 0], [0, 1]]
    proposal_sigma = 0.001
    cov = np.identity(len(initial_thetas))*(proposal_sigma)
    hyper_cov = np.identity(len(initial_phis))*proposal_sigma
    obs_error= obs_error*np.identity(len(obs))
    # obs_error = np.linalg.inv(obs_error)

    sip = SIP.MCMC_with_Gibbs(steps, initial_thetas, initial_phis, cov, hyper_cov, obs_error, glv_obj)
    accept_ratio = sip.MCMC()
    # print(sip.thetas)
    print(accept_ratio)
    np.savetxt("outputs/sip_raw_chain.dat", sip.thetas)
    np.savetxt("outputs/sip_hyper_raw_chain.dat", sip.phis)

    # SFP
    # sampled_thetas = np.loadtxt("sip_raw_chain.dat")
    sampled_thetas = sip.thetas
    chain_samples = len(sampled_thetas)
    num_times = len(times)
    model_output = np.zeros((chain_samples, num_times))
    for i in range(chain_samples):
        thetas = sampled_thetas[i]
        # conc = integrate.odeint(enriched_lv_derivatives, t=times, y0=[0.25], args=tuple([thetas],)).T.flatten()
        conc = glv_obj.get_enriched_conc(thetas).flatten()
        # print(conc)
        model_output[i] = conc

    # print(conc)
    np.savetxt("outputs/sfp_qoi_seq.dat", model_output)
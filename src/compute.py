import SIP
import numpy as np
import model

if __name__ == "__main__":

    # initialize model
    dataFile = "inputs/info.txt"
    info = np.loadtxt(dataFile,comments="%")
    n_S = int(info[0])
    reduced_size = int(info[1])
    n_phis_cal = int(info[2])
    n_phis_val = int(info[3])
    n_times = int(info[4])
    obs_error = float(info[5])
    inad_type = int(info[6])

    # get calibration/validation A and r
    # interaction matrix
    # A = [[-3.0, -1.0], [-1.0, -2.0]]
    A = np.loadtxt("inputs/matrix.txt")
    # A = np.loadtxt("inputs/matrix2.txt")
    # # growth rates
    # r = [5.0, 3.0]
    r = np.loadtxt("inputs/growthrates.txt")
    # r = np.loadtxt("inputs/growthrates2.txt")

    # species concentrations at t=0
    # initial_conc = [0.25, 0.5]
    initial_conc = np.loadtxt("inputs/init_concentrations.txt")[0]
    max_time = 1.1
    t_span = (0.0, max_time)
    timestep = 0.1
    induction_time = 0.0
    times = np.arange(induction_time, max_time, timestep)
    glv_obj = model.glv(A, r, initial_conc, reduced_size, t_span, times, obs_error)
    # obs = glv_obj.get_concentrations()[:,0:reduced_size].flatten()
    # print(obs)

    #detailed vals
    yobs = glv_obj.get_detailed_obs()
    # print(yobs)
    # np.savetxt("inputs/datafile2.txt", yobs)
    np.savetxt("inputs/datafile.txt", yobs)

    # reduced vals
    red_y = glv_obj.get_reduced_obs()
    # np.savetxt("inputs/datafile-reduced2.txt", red_y)
    np.savetxt("inputs/datafile-reduced.txt", red_y)

    # SIP: calibration
    # number of MCMC steps
    steps = 100
    initial_thetas = [-10.0]*reduced_size
    initial_phis = np.array([[-10.0]*reduced_size, [0.1]*reduced_size]).flatten()

    proposal_sigma = 0.001
    cov = np.identity(len(initial_thetas))*(proposal_sigma)
    proposal_sigma = 0.001
    hyper_cov = np.identity(len(initial_phis))*proposal_sigma
    obs_error= obs_error*np.identity(len(yobs))

    sip = SIP.MCMC_with_Gibbs(steps, initial_thetas, initial_phis, cov, hyper_cov, obs_error, glv_obj)
    accept_ratio = sip.MCMC()
    # print(sip.thetas)
    print(accept_ratio)
    # np.savetxt("outputs/sip_raw_chain2.dat", sip.thetas)
    # np.savetxt("outputs/sip_hyper_raw_chain2.dat", sip.phis)
    np.savetxt("outputs/sip_raw_chain_gibbs1.dat", sip.thetas)
    np.savetxt("outputs/sip_hyper_raw_chain_gibbs1.dat", sip.phis)

    # SFP
    # sampled_thetas = np.loadtxt("outputs/sip_raw_chain.dat")
    sampled_thetas = sip.thetas
    chain_samples = len(sampled_thetas)
    num_times = len(times)
    model_output = np.zeros((chain_samples, num_times*reduced_size*(n_phis_cal+n_phis_val)))
    for i in range(chain_samples):
        thetas = sampled_thetas[i]
        # conc = integrate.odeint(enriched_lv_derivatives, t=times, y0=[0.25], args=tuple([thetas],)).T.flatten()
        conc = glv_obj.get_enriched_conc(thetas).flatten()
        # print(conc)
        model_output[i] = conc

    # print(conc)
    # np.savetxt("outputs/sfp_qoi_seq2.dat", model_output)
    np.savetxt("outputs/sfp_qoi_seq_gibbs1.dat", model_output)
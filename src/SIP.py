import numpy as np
import scipy.stats as stats
from scipy import integrate
import model
# import math
from scipy.linalg import fractional_matrix_power

def adapt_covariance(eps, X, old_cov, step, non_adapt_interval=50):
    """
    recompute the proposal covariance
    # X  is thetas over time
    """
    if step > non_adapt_interval:
        s_d = (2.4**2)/len(X[0])
        # m^0 is the initial guess for thetas
        # C^1 = s_d*Cov(m_0,...,m^(step-1)) + s_d*eps*I_d
        xstep_bars = np.zeros((len(X[0]), 1))
        for d in range(len(X[0])):
            xi = X[0:step, d]
            xstep_bars[d, 0] = np.sum(xi)/(step)
        avg_xstep = (step) * np.matmul(xstep_bars, xstep_bars.T)

        summation = np.zeros((len(X[0]), len(X[0])))
        for k in range(1,step+1):
            xi = X[k-1:k]
            temp = np.matmul(xi.T, xi)
            summation += temp
        new_cov = (summation - avg_xstep)/(step-1)

        # sanity check
        if new_cov.diagonal().any() < 0.0:
            print("negative sd!?!") 
        return (s_d * new_cov) + s_d * eps * np.identity(len(X[0]))
    else:
        return old_cov

class MCMC_with_Gibbs:
    """ Markov Chain Monte Carlo with Gibbs sampling"""

    def __init__(self, MC_steps, initial_thetas, initial_phis, cov, hyper_cov, obs_error, glv_obj):
        self.MC_steps = MC_steps
        self.cov = cov
        self.hyper_cov = hyper_cov
        self.num_parameters = len(initial_thetas)
        self.thetas = np.zeros((MC_steps, self.num_parameters))
        self.thetas[0] = initial_thetas
        # self.num_hyperparameters = len(initial_phis)
        self.phis = np.zeros((MC_steps, len(initial_phis)))
        self.phis[0] = initial_phis
        # self.obs = glv_obj.get_concentrations()[:,0:glv_obj.reduced_size].flatten()
        self.obs = glv_obj.get_detailed_obs()
        self.obs_error = obs_error
        self.glv_obj = glv_obj
        self.lowerbound = np.diag(self.glv_obj.A[0:self.glv_obj.reduced_size, 0:self.glv_obj.reduced_size])
        # competitive/cooperative
        self.upperbound = -2*self.lowerbound

        # competitive
        # self.upperbound = -1*self.lowerbound

    def loglikelihood(self, theta):
        """
        p(D|theta) = N(D, mu=theta, sigma=obs_error)
        """
        y_theta = self.glv_obj.get_enriched_conc(theta).flatten()
        return stats.multivariate_normal.logpdf(self.obs, mean=y_theta, cov=self.obs_error)
        # misfitValue = 0.0
        # var = self.obs_error[0][0]
        # for i in range(len(y_theta)):
        #     diff = self.obs[i] - y_theta[i]
        #     misfitValue += diff * diff / var
        # return (-0.5 * misfitValue)

    def logprior(self, theta, phi):
        """
        p_prior = p(theta|phi)p(mu)p(Sigma)
        p(theta|phi) = N(theta, mu=phi[0], sigma=phi[1])
        p(mu_i) ~ U(-a_ii, a_ii)
        p(Sigma_ii) ~ logN(0, 1)
        """
        # print(self.lowerbound, self.upperbound)
        uni = np.sum(stats.uniform.logpdf(phi[0:self.num_parameters], loc=self.lowerbound, scale=self.upperbound))
        if uni <= np.NINF:
            # print("abs mean too big")
            return uni
        # log_norm = np.sum(stats.lognorm.logpdf(phi[self.num_parameters:], [0.5]*self.num_parameters))
        log_norm = np.sum(stats.expon.logpdf(phi[self.num_parameters:], scale = 0.1))
        if log_norm <= np.NINF:
            # print("not PSD", log_norm)
            return log_norm
        norm = stats.multivariate_normal.logpdf(theta, mean=phi[0:self.num_parameters], cov=np.diag(phi[self.num_parameters:]))
        # print("mu_0", phi[0:self.num_parameters])
        # print("sigma_0", phi[self.num_parameters:])
        # print("uni", uni)
        # print("log_norm", log_norm)
        # print("norm", norm)
        return norm + uni + log_norm

    def gibbs_proposal(self, step):
        # return self.thetas[step -1], self.phis[step -1]
        """
        Generate theta_prime, phi_prime given theta and phi and gaussian dist
        theta_prime = mu + cov^(1/2)*N(0,I)
        phi_prime = mu + cov^(1/2)*N(0,I)
        """
        mu = self.thetas[step - 1]
        default_mu = np.zeros(len(mu))
        default_cov =  np.identity(len(mu))
        norm_dist = np.random.multivariate_normal(default_mu, default_cov)
        half_cov = fractional_matrix_power(self.cov, 0.5)
        new_theta = mu + np.matmul(half_cov, norm_dist)

        mu = self.phis[step - 1]
        default_mu = np.zeros(len(mu))
        default_cov =  np.identity(len(mu))
        norm_dist = np.random.multivariate_normal(default_mu, default_cov)
        half_cov = fractional_matrix_power(self.hyper_cov, 0.5)
        new_phi = mu + np.matmul(half_cov, norm_dist)
        return new_theta, new_phi

    def logtarget(self, theta, phi):
        """
        log-prob_posterior(theta, phi|D) propto loglikelihood + logprior
        """
        prior = self.logprior(theta, phi)
        # print("prior", prior)
        if prior <= np.NINF:
            return prior
        like = self.loglikelihood(theta)
        # print("like", like)
        return prior + like

    def _MH_acceptance(self, curr_theta, curr_phi, step):
        """
        Metroplis-Hastings Acceptance Criterion
        todo: do not assume propsal is Gaussian
        """
        old_theta = self.thetas[step-1]
        old_phi = self.phis[step-1]
        new_post = self.logtarget(curr_theta, curr_phi)
        if new_post <= np.NINF:
            return False
        temp = new_post - self.logtarget(old_theta, old_phi)
        posterior_div = np.exp(temp)
        # print("posterior div", posterior_div)
        alpha = min(1, posterior_div)
        r = np.random.uniform()
        if r < alpha:
            return True
        return False

    def MCMC(self):
        """ Markov Chain Monte Carlo loop"""
        accept_count = 1
        # newCov = 0.0
        for i in range(1, self.MC_steps):
            # if i <= burnin:
            #     eps = 0.00001
            #     newCov = adapt_covariance(eps, self.thetas, self.M, i)
            #     # print("new propsal cov", newCov)
            #     temp_M = np.linalg.inv(newCov)
            #     self.set_M_and_G(temp_M, newCov)

            curr_theta, curr_phi = self.gibbs_proposal(i)
            # print(curr_theta, curr_phi)
            accept = self._MH_acceptance(curr_theta, curr_phi, i)
            # print(i, accept)
            if accept:
                self.thetas[i] = curr_theta
                self.phis[i] = curr_phi
                accept_count += 1
            else:
                self.thetas[i] = self.thetas[i-1]
                self.phis[i] = self.phis[i-1]
        print("accepted steps:", accept_count)
        # print("final adapted covariance", newCov)
        return accept_count/(self.MC_steps)
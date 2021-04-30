import numpy as np
from scipy import integrate
from numpy import linalg as LA

np.random.seed(seed=1)

class glv:
    """ Generalized Lotka-Volterra Equations """

    def __init__(self, A, r, initial_conc, reduced_size, t_span, times, obs_error):
        self.A = A
        self.r = r
        self.initial_conc = initial_conc
        self.reduced_size = reduced_size
        self.t_span = t_span
        self.times = times
        self.obs_error = np.sqrt(obs_error)
        self.obs = self.get_detailed_obs()

    def get_concentrations(self):
        """
        return Lotka-Volterra species concentrations of the GLV model for input
        growth factors (r) and interaction matrix (A)
        uses global initial and time
        """
        return np.array(integrate.solve_ivp(lv_derivatives,
                                            self.t_span, self.initial_conc, method='RK45',
                                            t_eval=self.times, atol=1e-4, rtol=1e-8,
                                            args=(self.r, self.A)).y).T

    def get_detailed_obs(self):
        obs = self.get_concentrations()[:,0:self.reduced_size].flatten()
        obs += np.random.normal(loc=0, scale=self.obs_error, size=len(obs))
        return obs

    def get_reduced_obs(self):
        reduced_r = self.r[0:self.reduced_size]
        A = np.array(self.A)
        reduced_A = A[0:self.reduced_size, 0:self.reduced_size]
        reduced_initial = self.initial_conc[:self.reduced_size]
        return np.array(integrate.solve_ivp(lv_derivatives,
                                            self.t_span, reduced_initial, method='RK45',
                                            t_eval=self.times, atol=1e-4, rtol=1e-8,
                                            args=(reduced_r, reduced_A)).y).T.flatten()

    def get_enriched_conc(self, thetas):
        # print("HERE!")
        # print(thetas)
        reduced_r = self.r[0:self.reduced_size]
        # # print(reduced_r)
        reduced_A = self.A[0:self.reduced_size, 0:self.reduced_size]
        # # print(reduced_A)
        reduced_initial = self.initial_conc[:self.reduced_size]
        return np.array(integrate.solve_ivp(enriched_lv_derivatives,
                                            self.t_span, reduced_initial, method='RK45',
                                            t_eval=self.times, atol=1e-4, rtol=1e-8,
                                            args=(reduced_r, reduced_A, thetas)).y).T

    #     #calibrate theta: dx_1/dt = 5x_1 -3x_1^2 - theta*x_1*x_2
    #     A = np.array(self.A)
    #     A[0,1] = thetas[0]
    #     return integrate.odeint(enriched_lv_derivatives, y0=self.initial_conc, t=self.times, args=(self.r, A, thetas))[:,0:self.reduced_size]
        # return np.array(integrate.solve_ivp(lv_derivatives,
        #                                         self.t_span, self.initial_conc, method='RK45',
        #                                         t_eval=self.times, atol=1e-4, rtol=1e-8,
        #                                         args=(self.r, tempA)).y).T[:,0:self.reduced_size]

def lv_derivatives(t, conc, r, A):
    """
    dx_i/dt = r_i*x_i + sum(aij*x_j)*x_i
    """
    # print("time", t)
    S = len(conc)
    derivs = np.zeros(S)
    for i in range(S):
        interactions = sum([A[i][j]*conc[j] for j in range(S)])
        derivs[i] = (r[i] + interactions)*conc[i]  
    return derivs

def enriched_lv_derivatives(t, conc, r, A, thetas, inad_type=6):
    """
    dx_i/dt = r_i*x_i + sum(aij*x_j)*x_i
    """
    s = len(conc)
    derivs = np.zeros(s)
    for i in range(s):
        interactions = sum([A[i][j]*conc[j] for j in range(s)])
        temp = (r[i] + interactions)*conc[i]
        if inad_type == 1:
            derivs[i] = temp + thetas[i]*conc[i] + thetas[i+s]*abs(temp)
        elif inad_type == 6:
            derivs[i] = temp + thetas[i]*conc[i]
        else:
            derivs[i] = temp
    return derivs
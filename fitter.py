from scipy.optimize import minimize
from parameters import FitParams
import numpy as np
from numpy.linalg import inv
from iminuit import Minuit

class Fitter:
    def __init__(self, datasets):
        self.datasets = datasets
        self.param_names = self._collect_params()

    def _collect_params(self):
        names = []
        for ds in self.datasets:
            for p in ds.param_names:
                if p not in names:
                    names.append(p)
        return names

    def chi2(self, p_array):
        p = FitParams(self.param_names, p_array)
        return sum(ds.chi2(p) for ds in self.datasets)

    def fit(self, p0, bounds):
        result = minimize(
            self.chi2,
            p0,
            method="L-BFGS-B",
            bounds=bounds
        )
        return result
        
    def numerical_hessian(self, func, x0, eps=1e-5):
        n = len(x0)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                dx_i = np.zeros(n); dx_i[i] = eps
                dx_j = np.zeros(n); dx_j[j] = eps
                fpp = func(x0 + dx_i + dx_j)
                fpm = func(x0 + dx_i - dx_j)
                fmp = func(x0 - dx_i + dx_j)
                fmm = func(x0 - dx_i - dx_j)
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps**2)
        return H
        
    def covariance(self, result):
        H = self.numerical_hessian(self.chi2, result.x)
        cov = inv(H) * 2.0
        return cov
        
    def make_minuit_func(self):
        def f(*args):
            return self.chi2(args)
        return f
        
    def fit_minuit(self, p0_dict, bounds_dict=None):
        f = self.make_minuit_func()


        m = Minuit(f, name=self.param_names, **p0_dict)

        # Important: tells Minuit this is a chi2 (not likelihood)
        m.errordef = Minuit.LEAST_SQUARES

        # Apply bounds if provided
        if bounds_dict:
            for name, (low, high) in bounds_dict.items():
                if name in m.parameters:
                    m.limits[name] = (low, high)

        m.migrad()
        m.hesse()

        return m

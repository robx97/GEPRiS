import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid, trapezoid
import pandas as pd
import uproot
import glob
import os
import re
from scipy.stats import norm

def _extract_energy(filename):
    # extract "1080" from run_gamma_1080keV.root
    match = re.search(r'run_gamma_(\d+)keV', filename)
    return float(match.group(1)) / 1000.0  if match else None
    
def _cache_key(*vals, precision=8):
    return tuple(round(v, precision) for v in vals)
    
def load_gamma_electron_distributions(path):
    files = glob.glob(os.path.join(path, "run_gamma_*keV.root"))

    gamma_data = []

    for f in files:
        energy = _extract_energy(f)
        if energy is None:
            continue

        with uproot.open(f) as root:
            hist = root["8"]   # assuming always "8"

            centers = hist.axis(0).centers()
            values = hist.values()
            gamma_data.append((energy, centers, values))

    # sort by energy (miss this and it breaks)
    gamma_data.sort(key=lambda x: x[0])

    return gamma_data

class ScintillatorModel:
    """
    Nonlinearity model and fit for JUNO liquid scintillator
    """

    def __init__(self, rho, path):
        self.rho = rho

        self._birks_cache = {}

        #get cosmos
        self._b12 = pd.read_csv('./inputs/B12_betashape.csv', sep=',')
        self._n12 = pd.read_csv('./inputs/N12_betashape.csv', sep=',')
        self._c11 = pd.read_csv('./inputs/C11_betashape.csv', sep=',')
        
        self.gamma_electron = load_gamma_electron_distributions(path+"gammas/")

        self.gamma_hists = {}

        for E, centers, values in self.gamma_electron:
            self.gamma_hists[E] = (centers, values)
        
        self.electron = [
            [values, centers]
            for (_, centers, values) in self.gamma_electron] # (2,n) array of secondary histos for n simulated gamma points. [0] is bin content and [1] is bin centers

        ## Get dE/dX
        df = pd.read_csv(path+'JUNO_stopping_sim.txt', sep='\\s+', header=None, engine='python')
        mass_stopping = df[1].to_numpy()
        E_vals = df[0].to_numpy()
        self.mass_stopping = mass_stopping
        self.E_vals = E_vals

        ## get cherenkov curve and interpolate
        cherenkov_energies = np.array([0., 0.19985692, 0.240201179, 0.25962539, 0.275677909, 0.294745057, 0.218886576, 0.320458404, 0.341831426, 0.394354839, 0.361602497, 0.439569688, 0.47528616, 0.510828564, 0.558324662, 0.61300078, 0.672312284, 0.744194644, 0.839337088, 0.94066077, 1.050728408, 1.176118626, 1.296305931, 1.416493236, 1.536680541, 1.656867846, 1.777055151, 1.897242456, 2.017429761, 2.137617066, 2.25780437, 2.377991675, 2.49817898, 2.618366285, 2.73855359, 2.858740895, 2.9789282, 3.099115505, 3.21930281, 3.339490114, 3.459677419, 3.579864724, 3.700052029, 3.820239334, 3.940426639, 4.060613944, 4.180801249, 4.300988554, 4.421175858, 4.541363163, 4.661550468, 4.781737773, 4.901925078, 5.022112383, 5.142299688, 5.262486993, 5.382674298, 5.502861602, 5.623048907, 5.743236212, 5.863423517, 5.983610822, 6.103798127, 6.223985432, 6.344172737, 6.464360042, 6.584547347, 6.704734651, 6.824921956, 6.945109261, 7.065296566, 7.185483871, 7.305671176, 7.425858481, 7.546045786, 7.666233091, 7.786420395, 7.9066077, 8.026795005, 8.14698231, 8.267169615, 8.38735692, 8.507544225, 8.62773153, 8.747918835, 8.868106139, 8.988293444, 9.108480749, 9.228668054, 9.348855359, 9.469042664, 9.589229969, 9.709417274, 9.829604579, 9.938865765])
        cherenkov_yield = np.array([0., 8.13249E-05, 0.000766873, 0.001266486, 0.001860128, 0.002543045, 0.000367906, 0.003304245, 0.00409492, 0.006595886, 0.005074881, 0.008956401, 0.011021791, 0.013341224, 0.016187772, 0.020080121, 0.024146027, 0.02964739, 0.037093932, 0.045053896, 0.054738379, 0.06479271, 0.075101867, 0.086904814, 0.096754722, 0.107409243, 0.118162999, 0.128981544, 0.140313682, 0.152043227, 0.16145597, 0.171169609, 0.183260589, 0.194522115, 0.207152101, 0.218709268, 0.229491038, 0.240804321, 0.251948907, 0.262635302, 0.272987892, 0.28339867, 0.292763746, 0.306307051, 0.318381099, 0.329166401, 0.339199155, 0.350978267, 0.361973458, 0.373006148, 0.383085462, 0.393357009, 0.404043213, 0.416813696, 0.428221838, 0.43862389, 0.452728171, 0.461944607, 0.474655834, 0.485116096, 0.491957144, 0.503295238, 0.513456617, 0.524556166, 0.535895658, 0.545906342, 0.552467362, 0.562093636, 0.576758076, 0.588822268, 0.599574988, 0.605285505, 0.614819785, 0.627336009, 0.642732548, 0.648320597, 0.658261963, 0.674972496, 0.686450931, 0.69755054, 0.701592383, 0.712057699, 0.722679121, 0.739198546, 0.750842186, 0.760789493, 0.772137824, 0.782367261, 0.793384569, 0.805881731, 0.814545656, 0.823980232, 0.8301192, 0.840770916, 0.850634188])

        self.cherenkov = interp1d(cherenkov_energies, cherenkov_yield, kind='linear', fill_value='extrapolate')

    ### Physics ### 
    
    def juno_resolution(self, E, a=0.033, b=0.009, c=0.0, fit=False):
        # Energy resolution (absolute sigma)
        frac = np.sqrt((a / np.sqrt(E))**2 + (b*b) + (c/E)**2)
        if fit:
            return frac
        else:
            return frac * E

    def instrumental_nl(self, E_vis, alpha, E_anchor = 2.22):
        #return np.array(1 + alpha*(E_vis-E_anchor))
        return np.array(1 - alpha*E_vis)

    def juno_reso_error(self, E, a, b, c, sigma_a, sigma_b, sigma_c, cov=None):
        frac = np.sqrt(a**2 / E + b**2 + c**2 / E**2)
        
        da = a / (E * frac)
        db = b / frac
        dc = c / (E**2 * frac)
        
        derivs = np.array([da, db, dc])
        
        if cov is None:
            var = (da*sigma_a)**2 + (db*sigma_b)**2 + (dc*sigma_c)**2
        else:
            derivs = np.stack([da, db, dc], axis=1)  # shape (len(E),3)
            var = np.einsum('ij,jk,ik->i', derivs, cov, derivs)  # shape (len(E),)
        
        return np.sqrt(var)

    # gammas and their deltas

    def electron_nl(self, T, A, kB_gcm2, fC):

        T = np.asarray(T)
        E_grid, quench_factor, _ = self.birks_integral(kB_gcm2)
        Q = np.interp(T, E_grid, quench_factor)
        cher = np.zeros_like(T)
        mask = T > 0
        cher[mask] = fC * self.cherenkov(T[mask]) / T[mask]
        nl = A * (Q + cher)
        return nl 
    
    def gamma_visible_nl(
        self,
        centers,
        weights,
        A,
        kB_gcm2,
        fC,
        E_gamma
    ):

        electron_nl = self.electron_nl(
            centers,
            A,
            kB_gcm2,
            fC
        )

        visible_energy = centers * electron_nl
        numerator = trapezoid(visible_energy * weights, centers)
        denominator = trapezoid(centers * weights, centers)

        return numerator / denominator
        #return numerator / E_gamma

    def gamma_peak_visible_energy(
        self,
        E_gamma,
        A,
        kB_gcm2,
        fC
    ):
    
        centers, weights = self.gamma_hists[E_gamma]
    
        gamma_nl = self.gamma_visible_nl(
            centers,
            weights,
            A,
            kB_gcm2,
            fC,
            E_gamma
        )
    
        return gamma_nl * E_gamma
        
        
    def gamma_nl_curve(self, A, kB_gcm2, fC, kI=0):
        E_true = []
        NL = []

        for E_gamma, centers, weights in self.gamma_electron:

            gamma_nl = self.gamma_visible_nl(
                centers,
                weights,
                A,
                kB_gcm2,
                fC,
                E_gamma
            )

            gamma_vis = gamma_nl * E_gamma
            gamma_vis *= self.instrumental_nl(gamma_vis, kI)
            E_true.append(E_gamma)
            NL.append(gamma_vis/E_gamma)

        E_true = np.array(E_true)
        NL = np.array(NL)

        return E_true, NL
    
    def scint_model(self, E_eval, A, kB_gcm2, fC, kI=0):

        E_sim, NL_sim = self.gamma_nl_curve(
            A,
            kB_gcm2,
            fC,
            kI
        )

        interp = interp1d(
            E_sim,
            NL_sim,
            kind='cubic',
            bounds_error=False,
            fill_value='extrapolate'
        )

        return interp(E_eval)

    ## electrons and positrons ##

    def beta_scint(self, T, A, kB_gcm2, fC, kI=0.0, is_pos=False):
        ############print(T[0:20])
        E_grid, quench_factor, _ = self.birks_integral(kB_gcm2)
        Q_interp = np.interp(T, E_grid, quench_factor)
        cher = fC * self.cherenkov(T) / T
        #print('cher: '+str(cher))
        #print('Q_interp: '+str(Q_interp))
        #print('A: '+str(A))
        #print('kB_gcm2: '+str(kB_gcm2))
        #print('fC: '+str(fC))
        #print('kI: '+str(kI))
        scint = A * (Q_interp + cher)
        E_vis_beta = T * scint
    
        if is_pos:

            # get actual 511 keV gamma histogram
            centers_511, weights_511 = self.gamma_hists[0.511]
    
            E_vis_511 = self.gamma_peak_visible_energy(
                0.511,
                A,
                kB_gcm2,
                fC
            )
             # electron visible energy
            E_vis_total = E_vis_beta + 2 * E_vis_511
            E_vis_total *= self.instrumental_nl(E_vis_total, kI)
            E_dep_total = T + 1.022
            nl = E_vis_total / E_dep_total
    
        else:
            E_vis_beta *= self.instrumental_nl(E_vis_beta, kI)
            nl = E_vis_beta / T
        return nl 

    def birks_integral(self, kB_gcm2):
        key = _cache_key(kB_gcm2)
        if key in self._birks_cache:
            return self._birks_cache[key]
        
        rho = self.rho
        mass_stopping = self.mass_stopping  # array in MeV cm^2/g
        E_vals = self.E_vals                # MeV
        # Convert dE/dx to MeV/cm and interpolate
        dEdx_vals = mass_stopping * rho
        dEdx = interp1d(E_vals, dEdx_vals, kind='linear', fill_value='extrapolate')
    
        # Convert kB to cm/MeV
        kB = kB_gcm2 / rho
    
        # Energy grid (avoid including exactly 0)
        Emin=1e-8
        #E_grid = np.linspace(Emin, E_vals[-1], int(1e6))
        E_grid = np.unique(np.concatenate([
        np.logspace(np.log10(Emin), np.log10(1.0), int(1e5)),
        np.linspace(1.0, E_vals[-1], 10000)]))
        E_grid.sort()
    
        # Integrand: dL/dE
        dL = 1.0 / (1 + kB * dEdx(E_grid))
        # cumulative integral
        L_cum = cumulative_trapezoid(dL, E_grid, initial=0.0)
        
        quench_factor = L_cum / E_grid
        self._birks_cache[key] = (E_grid, quench_factor, dEdx)
        
        return E_grid, quench_factor, dEdx

    def build_visible_spectrum(
        self,
        true_energy,
        dnde,
        visible_energy,
        target_centers,
        a,
        b,
        c
    ):
        smeared = self.smear_spectrum(
            visible_energy,
            dnde,
            a=a,
            b=b,
            c=c
        )
    
        spectrum, bins, _ = self.rebin_to_centers(
            visible_energy,
            smeared,
            target_centers
        )
    
        return spectrum, bins

    def B12_prediction(
        self,
        target_centers,
        A,
        kB_gcm2,
        fC,
        kI=0.0,
        a=0.033,
        b=0.009,
        bp=0.0,
        c=0.0,
        perturb=False,
        random_seed=None
    ):
    
        beta_E = self._b12['E_keV'].to_numpy() / 1000.0
        beta_dnde = self._b12['dNdE'].to_numpy()
        beta_unc = self._b12['unc'].to_numpy() / 1000.0
    
        n12_E = self._n12['E_keV'].to_numpy() / 1000.0
        n12_dnde = self._n12['dNdE'].to_numpy()
        n12_unc = self._n12['unc'].to_numpy() / 1000.0
    
        if perturb:
    
            rng = np.random.default_rng(random_seed)
    
            beta_dnde = rng.normal(beta_dnde, beta_unc)
            n12_dnde = rng.normal(n12_dnde, n12_unc)
    
        beta_nl = self.beta_scint(
            beta_E,
            A,
            kB_gcm2,
            fC,
            kI,
            is_pos=False
        )
    
        n12_nl = self.beta_scint(
            n12_E,
            A,
            kB_gcm2,
            fC,
            kI,
            is_pos=True
        )
    
        beta_vis = beta_E * beta_nl
        n12_vis = (n12_E + 1.022) * n12_nl

        #print("beta_E finite:", np.all(np.isfinite(beta_E)))
        #print("beta_nl finite:", np.all(np.isfinite(beta_nl)))
        #print("beta_vis finite:", np.all(np.isfinite(beta_vis)))

        #bad = ~np.isfinite(beta_vis)
        #print("bad beta_vis:", beta_vis[bad])
        #print("corresponding E:", beta_E[bad])
        #print(self.cherenkov(beta_E[:20]))
    
        # detector response 
        b12_spec, bins = self.build_visible_spectrum(
            beta_E,
            beta_dnde,
            beta_vis,
            target_centers,
            a,
            b + bp,
            c
        )
        
        n12_spec, _ = self.build_visible_spectrum(
            n12_E,
            n12_dnde,
            n12_vis,
            target_centers,
            a,
            b + bp,
            c
        )
        #print("b12 ", b12_spec[0:10])
        #print("c12 ",n12_spec[0:10])
        #print("Are there NaNs in 12N?", np.isnan(n12_spec).any())
        #print("Does model hit zero?", (b12_spec + n12_spec == 0).any())
    
        return b12_spec, n12_spec, bins    
    
    def C11_prediction(
        self,
        target_centers,
        A,
        kB_gcm2,
        fC,
        kI=0.0,
        a=0.033,
        b=0.009,
        bp=0.0,
        c=0.0,
        perturb=False,
        random_seed=None
    ):
    
        E = self._c11['E_keV'].to_numpy() / 1000.0
        dnde = self._c11['dNdE'].to_numpy()
        unc = self._c11['unc'].to_numpy() / 1000.0
    
        if perturb:
    
            rng = np.random.default_rng(random_seed)
    
            dnde = rng.normal(dnde, unc)
    
        nl = self.beta_scint(
            E,
            A,
            kB_gcm2,
            fC,
            kI,
            is_pos=True
        )
    
        visible = (E + 1.022) * nl
    
        spectrum, bins = self.build_visible_spectrum(
            E,
            dnde,
            visible,
            target_centers,
            a,
            b + bp,
            c
        )
    
        return spectrum, bins

    def nH_prediction(
        self,
        target_centers,
        A,
        kB_gcm2,
        fC,
        kI=0.0,
        a=0.033,
        b=0.009,
        bp=0.0,
        c=0.0
    ):
    
        Evis = self.gamma_peak_visible_energy(
            2.22,
            A,
            kB_gcm2,
            fC,
            kI
        )
    
        sigma = self.juno_resolution(
            Evis,
            a,
            b + bp,
            c
        )
    
        spectrum = norm.pdf(
            target_centers,
            loc=Evis,
            scale=sigma
        )
    
        spectrum /= np.sum(spectrum)

        return spectrum

    def beta_mc_uncertainty(self, T, A, kB_gcm2, fC, alpha, sigma_A, sigma_kB, sigma_fC, sigma_alpha,
                      is_pos=False, cov=None, n_samples=1000, random_seed=None):
        rng = np.random.default_rng(random_seed)
    
        if cov is None:
            cov = np.diag([sigma_A**2, sigma_kB**2, sigma_fC**2, sigma_alpha**2])
        # multivariate normal draws
        draws = rng.multivariate_normal(mean=[A, kB_gcm2, fC, alpha], cov=cov, size=n_samples)
    
        nls = []
        for a, k, f,af in draws:
            try:
                nl = self.beta_scint(T, a, k, f, af, is_pos=is_pos)
            except Exception:
                # if occasional draws produce bad values (e.g. negative kB) skip
                continue
            nls.append(np.asarray(nl))
    
        nls = np.stack(nls, axis=0)  # shape (n_valid, ...) ; second axes match nl shape
        mean_nl = np.mean(nls, axis=0)
        sigma_nl = np.std(nls, axis=0, ddof=1)
    
        return mean_nl, sigma_nl

    def gamma_mc_uncertainty(self, E, A, kB_gcm2, fC, alpha, sigma_A, sigma_kB, sigma_fC, sigma_alpha,
                             cov=None, n_samples=500, random_seed=None):
        rng = np.random.default_rng(random_seed)
    
        if cov is None:
            cov = np.diag([sigma_A**2, sigma_kB**2, sigma_fC**2, sigma_alpha**2])
    
        draws = rng.multivariate_normal(mean=[A, kB_gcm2, fC, alpha], cov=cov, size=n_samples)
    
        results = []
        for A_i, kB_i, fC_i, alpha_i in draws:
            try:
                res = self.scint_model(E, A_i, kB_i, fC_i, alpha_i)
            except Exception:
                continue  # skip unphysical parameter sets
            results.append(res)
    
        results = np.stack(results, axis=0)  # (n_valid, len(E))
        mean_scint = np.mean(results, axis=0)
        sigma_scint = np.std(results, axis=0, ddof=1)
        return mean_scint, sigma_scint

    def cosmo_with_uncertainty(self, predict_func, A, kB_gcm2, fC, a, b, bp, c, alpha, n_ibd, n_b12, n_n12, n_c11, target_centers, 
                           cov, n_samples=200, perturb_beta=True, random_seed=None):

        rng = np.random.default_rng(random_seed)
    
        # Nominal prediction (no perturbation)
        if predict_func == self.C11_prediction:
            spec_nom, spec_bins = predict_func(target_centers, A, kB_gcm2, fC, a, b, bp, c, alpha, perturb=False)
        if predict_func == self.B12_prediction:
            spec_b12, spec_n12, spec_bins = predict_func(target_centers, A, kB_gcm2, fC, a, b, bp, c, alpha, perturb=False)
        if predict_func == self.nH_prediction:
            spec_nom = predict_func(target_centers, A, kB_gcm2, fC, a, b, bp, c, alpha)
    
        # Sample correlated parameter variations
        if(len(cov) > 8):
            params = rng.multivariate_normal([A, kB_gcm2, fC, a, b, bp, c, alpha, n_ibd, n_b12, n_n12, n_c11], cov, size=n_samples)
        else:
            params = rng.multivariate_normal([A, kB_gcm2, fC, n_b12, n_n12, n_c11], cov, size=n_samples)
    
        spectra = []
        if(len(cov) > 8):
            for Ai, kBi, fCi, ai, bi, bpi, ci, alpha_i, ibd_i, b12_i, n12_i, c_11_i in params:
                if predict_func == self.C11_prediction:
                    spec_i, *_ = predict_func(
                        target_centers, Ai, kBi, fCi, ai, bi, bpi, ci, alpha_i ,
                        perturb=perturb_beta,      
                        random_seed=rng.integers(1e9)         # unique perturbation each draw!!!
                    )
                    spectra.append(spec_i*c_11_i)
                if predict_func == self.B12_prediction:
                    spec_bi, spec_ni, *_ = predict_func(
                        target_centers, Ai, kBi, fCi, ai, bi, bpi, ci, alpha_i ,
                        perturb=perturb_beta,      
                        random_seed=rng.integers(1e9)         # unique perturbation each draw!!!
                    )
                    spectra.append(spec_bi*b12_i + spec_ni*n12_i)
                if predict_func == self.nH_prediction:
                    spec_nhi = predict_func(
                        target_centers, Ai, kBi, fCi, ai, bi, bpi, ci, alpha_i        # unique perturbation each draw!!!
                    )
                    spectra.append(spec_nhi*ibd_i)
        else:
            for Ai, kBi, fCi, b12_i, n12_i, c_11_i in params:
                if predict_func == self.C11_prediction:
                    spec_i, *_ = predict_func(
                        target_centers, Ai, kBi, fCi, 
                        perturb=perturb_beta,      
                        random_seed=rng.integers(1e9)         # unique perturbation each draw!!!
                    )
                    spectra.append(spec_i*c_11_i)
                if predict_func == self.B12_prediction:
                    spec_bi, spec_ni, *_ = predict_func(
                        target_centers, Ai, kBi, fCi, 
                        perturb=perturb_beta,      
                        random_seed=rng.integers(1e9)         # unique perturbation each draw!!!
                    )
                    spectra.append(spec_bi*b12_i + spec_ni*n12_i)
    
        spectra = np.vstack(spectra)
        spec_mean = np.mean(spectra, axis=0)
        spec_std  = np.std(spectra, axis=0)

        if predict_func == self.C11_prediction:
            return spec_nom, spec_bins, spec_std
        if predict_func == self.B12_prediction:
            return spec_b12, spec_n12, spec_bins, spec_std
        if predict_func == self.nH_prediction:
            return spec_nom, spec_std

    def make_pulls(self, popt, p_err, cov=None, n_draws=1000, seed=None):
        # Convert to proper numpy arrays
        p0 = np.asarray(popt, dtype=float)
        p_err = np.asarray(p_err, dtype=float)
        rng = np.random.default_rng(seed)

        if cov is None:
            cov = np.diag(np.array(p_err)**2)
        draws = rng.multivariate_normal(mean=popt, cov=cov, size=n_draws)
        sigmas = p_err #sqrt of diagonal of the cov matrix, technically
        pulls = (draws - p0[None, :]) / sigmas[None, :]   # shape (n_draws, 3)
        return pulls

    def rebin_to_centers(self, old_centers, old_contents, target_centers, old_errors=None):
        old_width = np.mean(np.diff(old_centers))
        new_width = np.mean(np.diff(target_centers))
    
        old_edges = np.concatenate([[old_centers[0] - old_width/2], old_centers + old_width/2])
        new_edges = np.concatenate([[target_centers[0] - new_width/2], target_centers + new_width/2])
    
        new_contents = np.zeros(len(target_centers))
        new_vars = np.zeros(len(target_centers))  # store variances
    
        for i in range(len(old_contents)):
            x0, x1 = old_edges[i], old_edges[i+1]
            height = old_contents[i] / old_width  # density
            var_density = (old_errors[i] / old_width)**2 if old_errors is not None else 0.0
    
            # find overlapping new bins
            mask = (new_edges[:-1] < x1) & (new_edges[1:] > x0)
            for j in np.where(mask)[0]:
                overlap = min(x1, new_edges[j+1]) - max(x0, new_edges[j])
                if overlap > 0:
                    frac = overlap  # absolute length, since height already in per-width units
                    new_contents[j] += height * frac
                    new_vars[j] += var_density * frac**2  # variance adds in quadrature
    
        new_errors = np.sqrt(new_vars)
        return new_contents, target_centers, new_errors

    def smear_spectrum(self, E_bins, spectrum, n_sigma=3, a=None, b=None, c=None):
        smeared = np.zeros_like(spectrum, dtype=float)

        dE = np.gradient(E_bins)

        for i, E in enumerate(E_bins):
            if spectrum[i] <= 0:
                continue

            sigma = self.juno_resolution(E, a, b, c) if a is not None else self.juno_resolution(E)
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 1e-6

            half_width = int(n_sigma * sigma / np.mean(dE))
            low = max(i - half_width, 0)
            high = min(i + half_width + 1, len(E_bins))

            E_window = E_bins[low:high]
            dE_window = dE[low:high]

            kernel = np.exp(-0.5 * ((E_window - E) / sigma)**2)

            # normalization
            kernel /= np.sum(kernel * dE_window)

            # weighting
            smeared[low:high] += spectrum[i] * kernel * dE[i]

        return smeared

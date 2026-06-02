import numpy as np
from parameters import get_param, FitParams, n_smooth_points

def _get(p, name):
    return p[name] if isinstance(p, dict) else getattr(p, name)
    
def _cache_key(*vals, precision=8):
    return tuple(round(v, precision) for v in vals)


class Dataset:
    def __init__(self, name, data, err):
        self.name = name
        self.data = data
        self.err = err

    def prediction(self, params):
        raise NotImplementedError

    def chi2(self, params):
        pred = self.prediction(params)
        return np.sum(((pred - self.data) / self.err)**2)

    def uncertainty(self, params, errors, n_samples=300, random_seed=None, cov=None):
        """
        Monte Carlo uncertainty on prediction using the master covariance matrix structure.
        """
        rng = np.random.default_rng(random_seed)
        names  = self.param_names
        means  = np.array([get_param(params, n) for n in names])

        if cov is not None:
            # 1. Fallback to guessing order based on params, but protect against bounds
            all_names = list(params.keys())
            
            # Extract valid indices that sit securely inside the covariance dimensions
            idx = []
            valid_names = []
            for n in names:
                if n in all_names:
                    i = all_names.index(n)
                    if i < cov.shape[0]: 
                        idx.append(i)
                        valid_names.append(n)
            
            # Project the sliced covariance matrix into this dataset's specific space
            sub_cov = np.zeros((len(names), len(names)))
            if len(idx) > 0:
                matrix_slice = cov[np.ix_(idx, idx)]
                for map_i, name_i in enumerate(valid_names):
                    for map_j, name_j in enumerate(valid_names):
                        pos_i = names.index(name_i)
                        pos_j = names.index(name_j)
                        sub_cov[pos_i, pos_j] = matrix_slice[map_i, map_j]
        else:
            # Diagonal uncertainty fallback when no covariance matrix is supplied
            sigmas  = np.array([errors.get(n, 0.0) for n in names])
            sub_cov = np.diag(sigmas ** 2)
    
        if not np.any(sub_cov > 0):
            return np.zeros_like(self.data, dtype=float)

        # Draw correlated parameter variations safely
        draws = rng.multivariate_normal(means, sub_cov, size=n_samples)
        preds = []
        for draw in draws:
            p = FitParams(names, draw)
            try:
                preds.append(np.asarray(self.prediction(p, perturb=True, smooth=True), dtype=float))
            except TypeError:
                preds.append(np.asarray(self.prediction(p, perturb=True), dtype=float))
            except Exception:
                continue

        if len(preds) < 2:
            return np.zeros_like(self.data, dtype=float)

        return np.std(np.stack(preds), axis=0, ddof=1)


class GammaDataset(Dataset):

    def __init__(self, gepris, E_data, y_data, yerr):
        super().__init__("gamma", y_data, yerr)
        self.param_names = ["A", "kB", "fC", "kI"]
        self.gepris = gepris
        self.E = E_data
        self._cache = {}
        self.ndf = len(y_data) - len(self.param_names)

    def prediction(self, p, perturb=False, smooth=False):
        A  = get_param(p, "A")
        kB = get_param(p, "kB")
        fC = get_param(p, "fC")
        kI = get_param(p, "kI")
        key = _cache_key(A, kB, fC, kI)
        if smooth:
            energies = np.linspace(np.min(self.E), np.max(self.E), n_smooth_points)
            return self.gepris.scint_model(energies, A, kB, fC, kI)
        else:
            if key not in self._cache:
                self._cache[key] = self.gepris.scint_model(self.E, A, kB, fC, kI)
            return self._cache[key]

class B12Dataset(Dataset):

    def __init__(self, gepris, centers, data, err):
        super().__init__("b12", data, err)
        self.param_names = ["A", "kB", "fC", "kI",
                   "resol_a", "resol_b", "resol_bp", "resol_c",
                   "N_b12", "N_n12"]
        self.gepris  = gepris
        self.centers = centers
        self._cache  = {}
        self.ndf = len(data) - len(self.param_names)

    def prediction(self, p, perturb = False):
        A    = get_param(p, "A");    kB = get_param(p, "kB")
        fC   = get_param(p, "fC");   kI = get_param(p, "kI")
        a    = get_param(p, "resol_a"); b  = get_param(p, "resol_b")
        bp   = get_param(p, "resol_bp"); c = get_param(p, "resol_c")
        N_b12 = get_param(p, "N_b12"); N_n12 = get_param(p, "N_n12")
        key = _cache_key(A, kB, fC, kI, a, b, bp, c)
        if perturb:
            b12, n12, _ = self.gepris.B12_prediction(
                self.centers, A, kB, fC, kI, a, b, bp, c, perturb = perturb)
            return N_b12 * b12 + N_n12 * n12
        else:
            if key not in self._cache:
                self._cache[key] = self.gepris.B12_prediction(
                    self.centers, A, kB, fC, kI, a, b, bp, c, perturb = perturb)
            b12, n12, _ = self._cache[key]
            return N_b12 * b12 + N_n12 * n12


class C11Dataset(Dataset):

    def __init__(self, gepris, centers, data, err):
        super().__init__("c11", data, err)
        self.param_names = ["A", "kB", "fC", "kI",
                   "resol_a", "resol_b", "resol_bp", "resol_c",
                   "N_c11"]
        self.gepris  = gepris
        self.centers = centers
        self._cache  = {}
        self.ndf = len(data) - len(self.param_names)

    def prediction(self, p, perturb = False):
        A  = get_param(p, "A");  kB = get_param(p, "kB")
        fC = get_param(p, "fC"); kI = get_param(p, "kI")
        a  = get_param(p, "resol_a"); b  = get_param(p, "resol_b")
        bp = get_param(p, "resol_bp"); c = get_param(p, "resol_c")
        N_c11 = get_param(p, "N_c11")
        key = _cache_key(A, kB, fC, kI, a, b, bp, c)
        if perturb:
            spec, _ = self.gepris.C11_prediction(
                self.centers, A, kB, fC, kI, a, b, bp, c, perturb = perturb)
            return N_c11 * spec
        else:
            if key not in self._cache:
                self._cache[key] = self.gepris.C11_prediction(
                self.centers, A, kB, fC, kI, a, b, bp, c, perturb = perturb)
            spec, _ = self._cache[key]
            return N_c11 * spec

class C10Dataset(Dataset):

    def __init__(self, gepris, centers, data, err):
        super().__init__("c10", data, err)
        self.param_names = ["A", "kB", "fC", "kI",
                   "resol_a", "resol_b", "resol_bp", "resol_c",
                   "N_c10", "N_c11_bkg"]
        self.gepris  = gepris
        self.centers = centers
        self._cache  = {}
        self.ndf = len(data) - len(self.param_names)

    def prediction(self, p, perturb = False):
        A    = get_param(p, "A");    kB = get_param(p, "kB")
        fC   = get_param(p, "fC");   kI = get_param(p, "kI")
        a    = get_param(p, "resol_a"); b  = get_param(p, "resol_b")
        bp   = get_param(p, "resol_bp"); c = get_param(p, "resol_c")
        N_c10 = get_param(p, "N_c10"); N_c11 = get_param(p, "N_c11_bkg")
        key = _cache_key(A, kB, fC, kI, a, b, bp, c)
        if perturb:
            c10, c11, _ = self.gepris.C10_prediction(
                self.centers, A, kB, fC, kI, a, b, bp, c, perturb = perturb)
            return N_c10 * c10 + N_c11 * c11
        else:
            if key not in self._cache:
                self._cache[key] = self.gepris.C10_prediction(
                    self.centers, A, kB, fC, kI, a, b, bp, c, perturb = perturb)
            c10, c11, _ = self._cache[key]
            return N_c10 * c10 + N_c11 * c11

class HeBLiDataset(Dataset):

    def __init__(self, gepris, centers, data, err):
        super().__init__("hebli", data, err)
        self.param_names = ["A", "kB", "fC", "kI",
                   "resol_a", "resol_b", "resol_bp", "resol_c",
                   "N_li8", "N_b8", "N_he6"]
        self.gepris  = gepris
        self.centers = centers
        self._cache  = {}
        self.ndf = len(data) - len(self.param_names)

    def prediction(self, p, perturb = False):
        A  = get_param(p, "A");  kB = get_param(p, "kB")
        fC = get_param(p, "fC"); kI = get_param(p, "kI")
        a  = get_param(p, "resol_a"); b  = get_param(p, "resol_b")
        bp = get_param(p, "resol_bp"); c = get_param(p, "resol_c")
        N_li8 = get_param(p, "N_li8"); N_b8 = get_param(p, "N_b8")
        N_he6 = get_param(p, "N_he6")
        key = _cache_key(A, kB, fC, kI, a, b, bp, c)
        if perturb:
            he6, b8, li8, _ = self.gepris.HeBLi_prediction(
                self.centers, A, kB, fC, kI, a, b, bp, c, perturb = perturb)
            return N_he6 * he6 + N_b8 * b8 + N_li8 * li8
        else:
            if key not in self._cache:
                self._cache[key] = self.gepris.HeBLi_prediction(
                self.centers, A, kB, fC, kI, a, b, bp, c, perturb = perturb)
            he6, b8, li8, _ = self._cache[key]
            return N_he6 * he6 + N_b8 * b8 + N_li8 * li8

class nHDataset(Dataset):

    def __init__(self, gepris, centers, data, err):
        super().__init__("nH", data, err)
        self.param_names = ["A", "kB", "fC", "kI",
                   "resol_a", "resol_b", "resol_bp", "resol_c",
                   "N_nH"]
        self.gepris  = gepris
        self.centers = centers
        self._cache  = {}
        self.ndf = len(data) - len(self.param_names)

    def prediction(self, p, perturb = False):
        A  = get_param(p, "A");  kB = get_param(p, "kB")
        fC = get_param(p, "fC"); kI = get_param(p, "kI")
        a  = get_param(p, "resol_a"); b  = get_param(p, "resol_b")
        bp = get_param(p, "resol_bp"); c = get_param(p, "resol_c")
        N_nH = get_param(p, "N_nH")
        key = _cache_key(A, kB, fC, kI, a, b, bp, c)
        if key not in self._cache:
            self._cache[key] = self.gepris.nH_prediction(
            self.centers, A, kB, fC, kI, a, b, bp, c)
        spec = self._cache[key]
        return N_nH * spec


class ResolutionDataset(Dataset):
    
    def __init__(self, gepris, centers, data, err):
        sort = np.argsort(centers)
        super().__init__("resol.", data[sort], err[sort])
        self.gepris = gepris
        self.centers = centers[sort]
        self.param_names = ["resol_a", "resol_b", "resol_bp", "resol_c"]
        self.ndf = len(data) - len(self.param_names)

    def prediction(self, p, perturb = False, smooth=False):
        a  = get_param(p, "resol_a")
        b  = get_param(p, "resol_b")
        c  = get_param(p, "resol_c")
        if smooth:
            energies = np.linspace(np.min(self.centers), np.max(self.centers), n_smooth_points)
            return self.gepris.juno_resolution(energies, a, b, c, fit=True)
        else:
            return self.gepris.juno_resolution(self.centers, a, b, c, fit=True)

    def uncertainty(self, params, errors, n_samples=None, random_seed=None):
        # resolution function is differentiable so no MC needed, neat
        a  = get_param(params, "resol_a"); b = get_param(params, "resol_b")
        c  = get_param(params, "resol_c")
        sa = errors.get("resol_a", 0.0);   sb = errors.get("resol_b", 0.0)
        sc = errors.get("resol_c", 0.0)
        energies = np.linspace(np.min(self.centers), np.max(self.centers), n_smooth_points)
        return self.gepris.juno_reso_error(energies, a, b, c, sa, sb, sc)


class InstNLDataset(Dataset):

    def __init__(self, gepris, centers):
        super().__init__("instNL", np.array([]), np.array([]))
        self.param_names = ["kI"]
        self.gepris  = gepris
        self.centers = centers

    def prediction(self, p, perturb = False):
        kI = get_param(p, "kI")
        return self.gepris.instrumental_nl(self.centers, kI)

    def chi2(self, p):
        kI = get_param(p, "kI")
        return ((kI - 0) / 0.6e-3) ** 2
import numpy as np
from parameters import get_param

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
        
class GammaDataset(Dataset):
    def __init__(self, gepris, E_data, y_data, yerr):
        super().__init__("gamma", y_data, yerr)
        self.param_names = ["A", "kB", "fC", "kI"]
        self.gepris = gepris
        self.E = E_data
        self._cache = {}
        self.ndf = len(y_data) - len(self.param_names)

    def prediction(self, p):
        A = get_param(p, "A")
        kB = get_param(p, "kB")
        fC = get_param(p, "fC")
        kI = get_param(p, "kI")
        key = _cache_key(A, kB, fC, kI)  # cache only physics params
        if key not in self._cache:
            self._cache[key] = self.gepris.scint_model(
                self.E,
                A, kB, fC, kI
            )
        return self._cache[key]

        
class B12Dataset(Dataset):
    def __init__(self, gepris, centers, data, err):
        super().__init__("b12", data, err)
        self.param_names = ["A", "kB", "fC", "kI", "resol_a",  "resol_b",  "resol_bp",  "resol_c", "N_b12", "N_n12"]
        self.gepris = gepris
        self.centers = centers
        self._cache = {}
        self.ndf = len(data) - len(self.param_names)

    def prediction(self, p):
        A = get_param(p, "A")
        kB = get_param(p, "kB")
        fC = get_param(p, "fC")
        kI = get_param(p, "kI")
        a = get_param(p, "resol_a")
        b = get_param(p, "resol_b")
        bp = get_param(p, "resol_bp")
        c = get_param(p, "resol_c")
        N_b12 = get_param(p, "N_b12")
        N_n12 = get_param(p, "N_n12")
        key = _cache_key(A, kB, fC, kI, a, b, bp, c)  # cache only physics params

        if key not in self._cache:
            self._cache[key] = self.gepris.B12_prediction(
                self.centers,
                A, kB, fC, kI, a, b, bp, c
            )

        b12, n12, _ = self._cache[key]
        return N_b12*b12 + N_n12*n12

class C11Dataset(Dataset):
    def __init__(self, gepris, centers, data, err):
        super().__init__("c11", data, err)
        self.param_names = ["A", "kB", "fC", "kI", "resol_a",  "resol_b",  "resol_bp",  "resol_c", "N_c11"]
        self.gepris = gepris
        self.centers = centers
        self._cache = {}
        self.ndf = len(data) - len(self.param_names)

    def prediction(self, p):
        A = get_param(p, "A")
        kB = get_param(p, "kB")
        fC = get_param(p, "fC")
        kI = get_param(p, "kI")
        a = get_param(p, "resol_a")
        b = get_param(p, "resol_b")
        bp = get_param(p, "resol_bp")
        c = get_param(p, "resol_c")
        N_c11 = get_param(p, "N_c11")
        
        key = _cache_key(A, kB, fC, kI, a, b, bp, c)

        if key not in self._cache:
            self._cache[key] = self.gepris.C11_prediction(
                self.centers,
                A, kB, fC, kI, a, b, bp, c
            )

        spec, _ = self._cache[key]
        return N_c11*spec
        
class InstNLDataset(Dataset):
    param_names = ["kI"]
    def __init__(self, gepris, centers):
        self.gepris = gepris
    
    def prediction(self, p):
        kI = get_param(p, "kI")
        return self.gepris.instrumental_nl(self.centers,kI)

    def chi2(self, p):
        kI = get_param(p, "kI")
        return ((kI - 0)/0.6e-3)**2
        
class ResolutionDataset(Dataset):
    def __init__(self, gepris, centers, data, err):
        self.param_names = ["resol_a","resol_b","resol_bp","resol_c"]
        sort = np.argsort(centers)
        super().__init__("resol.", data[sort], err[sort])
        self.gepris = gepris
        self.centers = centers[sort]
        self.ndf = len(data) - len(self.param_names)
        
    def prediction(self, p):
        a = get_param(p, "resol_a")
        b = get_param(p, "resol_b")
        c = get_param(p, "resol_c")
        return self.gepris.juno_resolution(
        self.centers,
        a,
        b,
        c,
        fit=True
    )

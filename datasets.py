import numpy as np

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
    param_names = ["A", "kB", "fC", "kI"]
    def __init__(self, gepris, E_data, y_data, yerr):
        super().__init__("gamma", y_data, yerr)
        self.gepris = gepris
        self.E = E_data
        self._cache = {}

    def prediction(self, p):
        key = (p.A, p.kB, p.fC, p.kI)  # cache only physics params
        if key not in self._cache:
            self._cache[key] = self.gepris.scint_model(
                self.E,
                p.A, p.kB, p.fC, p.kI
            )
        return self._cache[key]

        
class B12Dataset(Dataset):
    param_names = ["A", "kB", "fC", "kI", "N_b12", "N_n12"]
    def __init__(self, gepris, centers, data, err):
        super().__init__("b12", data, err)
        self.gepris = gepris
        self.centers = centers
        self._cache = {}

    def prediction(self, p):
        key = (p.A, p.kB, p.fC, p.kI)  # cache only physics params

        if key not in self._cache:
            self._cache[key] = self.gepris.B12_prediction(
                self.centers,
                p.A, p.kB, p.fC, p.kI
            )

        b12, n12, _ = self._cache[key]
        return p.N_b12*b12 + p.N_n12*n12

class C11Dataset(Dataset):
    param_names = ["A", "kB", "fC", "kI", "N_c11"]
    def __init__(self, gepris, centers, data, err):
        super().__init__("c11", data, err)
        self.gepris = gepris
        self.centers = centers
        self._cache = {}

    def prediction(self, p):
        key = (p.A, p.kB, p.fC, p.kI)

        if key not in self._cache:
            self._cache[key] = self.gepris.C11_prediction(
                self.centers,
                p.A, p.kB, p.fC, p.kI
            )

        spec, _ = self._cache[key]
        return p.N_c11*spec
        
class InstNLDataset(Dataset):
    param_names = ["kI"]
    def __init__(self, gepris, centers):
        self.gepris = gepris
    
    def prediction(self, p):
        return self.gepris.intrumental_nl(p.kI)

    def chi2(self, p):
        return ((p.kI - 0)/0.6e-3)**2
        
class ResolutionDataset(Dataset):
    param_names = ["a","b","c"]
    def __init__(self, gepris, centers, data, err):
        super().__init__("resol.", data, err)
        self.gepris = gepris
        self.centers = centers
    
    def prediction(self, p):
        return self.gepris.juno_resolution(centers, p.a, p.b, p.c)

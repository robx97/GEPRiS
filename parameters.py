DEFAULT_PARAMS = {
    # physics
    "A": 1.0,
    "kB": 6.5e-3,
    "fC": 0.0,

    # detector
    "resol_a": 0.033,
    "resol_b": 0.009,
    "resol_bp": 0.0,
    "resol_c": 0.0,
    "kI": 0.0,

    # normalizations
    "N_b12": 1.0,
    "N_n12": 1.0,
    "N_c11": 1.0,
    "N_ibd": 1.0,
}

def get_param(p, name):
    # dict
    if isinstance(p, dict):
        if name in p:
            return p[name]

    # FitParams
    if hasattr(p, name):
        return getattr(p, name)

    # defaults
    if name in DEFAULT_PARAMS:
        return DEFAULT_PARAMS[name]

    raise KeyError(f"Parameter '{name}' not found and no default defined. Check paramters.py!")

class FitParams:
    def __init__(self, names, values):
        self._names = names
        self._values = dict(zip(names, values))

    def __getattr__(self, name):
        return self._values[name]

    def to_array(self):
        return [self._values[n] for n in self._names]

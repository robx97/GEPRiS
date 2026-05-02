class FitParams:
    def __init__(self, names, values):
        self._names = names
        self._values = dict(zip(names, values))

    def __getattr__(self, name):
        return self._values[name]

    def to_array(self):
        return [self._values[n] for n in self._names]

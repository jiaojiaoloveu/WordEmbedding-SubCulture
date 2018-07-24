class Norm:
    epa = None
    mean = None
    std = None

    @classmethod
    def norm(cls, epa):
        import numpy as np
        if cls.epa is None:
            cls.mean = np.mean(epa, axis=0)
            cls.std = np.std(epa, axis=0)
            cls.epa = epa
            epa = (epa - cls.mean) / cls.std
        return epa


    @classmethod
    def denorm(cls, epa):
        if cls.std is not None and cls.mean is not None:
            epa = (epa * cls.std) + cls.mean
        return epa

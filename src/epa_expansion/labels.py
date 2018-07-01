class LabelSpace:
    V = 'V'
    A = 'A'
    D = 'D'
    E = 'E'
    P = 'P'
    A = 'A'
    Dimension = 3
    Max = 4.3
    Min = -4.3
    mapping_epa = {V: E,
                   A: P,
                   D: A
                   }

    @classmethod
    def get_epa(cls, axis):
        return cls.mapping_epa[axis]


class WarrinerColumn:
    V = 'V.Mean.Sum'
    A = 'A.Mean.Sum'
    D = 'D.Mean.Sum'
    Word = 'Word'
    Min = 'Min'
    Max = 'Max'

    @classmethod
    def get_min_max_dic(cls):
        return {
            cls.Min: 10,
            cls.Max: 0
        }


class Configs:
    alpha = 1
    iterations = 100
    enn = 0.6
    exp = 1

    seed = 5000
    eval = 8000
    epa = 2
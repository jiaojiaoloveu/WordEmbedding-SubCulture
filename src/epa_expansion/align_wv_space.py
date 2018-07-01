from align_models import align_space
import numpy as np
import copy


def align_models(source, target):
    w = align_space(source, target)
    new_source = copy.deepcopy(source)
    new_source.wv.vectors = np.matmul(new_source.wv.vectors, w)
    return new_source, target

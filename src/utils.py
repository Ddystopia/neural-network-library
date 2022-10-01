import numpy as _np


class Utils:
    @staticmethod
    def unison_shuffled_copies(rng, *copies: _np.ndarray) -> list[_np.ndarray]:
        p = _np.arange(len(copies[0]))
        rng.shuffle(p)
        return ([a[p] for a in copies])


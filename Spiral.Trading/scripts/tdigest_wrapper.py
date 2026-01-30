"""Picklable wrapper for pytdigest.TDigest."""

import numpy as np
from pytdigest import TDigest


class PicklableTDigest:
    """A picklable wrapper around pytdigest.TDigest."""
    
    def __init__(self, td: TDigest):
        self._compression = td.compression
        self._tdigest = td
    
    def cdf(self, values: np.ndarray) -> np.ndarray:
        """Compute CDF for given values (returns values in [0, 1])."""
        return self._tdigest.cdf(values) # type: ignore
    
    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize values to [-1, 1] using CDF transform."""
        return self.cdf(values) * 2 - 1

    def __getstate__(self):
        """For pickling: only save centroids and compression."""
        return {
            'centroids': self._tdigest.get_centroids(),
            'compression': self._compression,
        }
    
    def __setstate__(self, state):
        """For unpickling: restore from centroids."""
        self.__init__(TDigest.of_centroids(state['centroids'], state['compression']))

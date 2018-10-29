"""
created by qiushye on 2018.10.22
python version >= 3
"""

import pca
from .Imputation import imputation
from sktensor import tucker, cp
import numpy as np
import time
import os
import sys
sys.path.append("..")


class BPCA_CPT(imputation):
    def __init__(self, miss_data, mult_components, threshold, max_iter=100):
        super(BPCA_CPT, self).__init__(miss_data, threshold, max_iter)
        self.multi_components = mult_components

    def impute(self):
        time_s = time.time()
        est_BPCA = np.zeros_like(self.miss_data)

        for i in range(self.shape[0]):
            data = self.miss_data[i]
            bppca = pca.bppca.BPPCA(data, q=self.multi_components)
            est_BPCA[i] = bppca.transform_infers()
        time_e = time.time()
        self.exec_time = time_e - time_s
        self.est_data = est_BPCA

        return est_BPCA

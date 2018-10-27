import pca
"""
created by qiushye on 2018.10.22
python version >= 3
"""

from Imputation import imputation
from sktensor import tucker, cp
import numpy as np
import time
import os
os.sys.path.append('../ppca-master/src/')


class bpca_cpt(imputation):
    def __init__(self, miss_data, mult_components):
        super(bpca_cpt, self).__init__(miss_data)
        self.multi_components = mult_components

    def impute(self):
        time_s = time.time()
        est_BPCA = np.zeros_like(self.miss_data)

        for i in range(self.shape):
            data = self.miss_data[0]
            bppca = pca.bppca.BPPCA(data, q=self.multi_components[i])
            est_BPCA[i] = bppca.transform_infers()
        time_e = time.time()
        self.exec_time = time_e - time_s
        self.est_data = est_BPCA

        return est_BPCA

"""
created by qiushye on 2018.10.23
python version >= 3
sktensor install: https://github.com/mnick/scikit-tensor/tree/master/sktensor
"""

import scipy
from sktensor.dtensor import dtensor


class truncator:

    def __init__(self, data, p):
        self.data = data
        self.truncate_rate = p

    def truncated_svd(self):
        SD = dtensor(self.data.copy())
        N = len(SD.shape)
        U_list = []     # left singluar matrix list
        r_list = []     # rank list
        SG = []         # singular value list
        for i in range(N):
            B = SD.unfold(i)
            U, sigma, VT = scipy.linalg.svd(B, 0)
            row_s = len(sigma)
            mat_sig = scipy.zeros((row_s, row_s))
            for j in range(row_s):
                mat_sig[j, j] = sigma[j]
                if sum(sigma[:j])/sum(sigma) > p:
                    SG.append(sigma[j])
                    break

            U_list.append(U[:, :j])
            r_list.append(j)

        self.SV_list = SG
        self.trun_index = j
        self.LSM_list = U_list
        self.rank_list = r_list
        return SG, j, U_list, r_list

"""
created by qiushye on 2018.10.28
python version >= 3
"""

import scipy.io as scio

if __name__ == "__main__":
    ori_path = 'D:/GZ_data/60days_tensor.mat'
    data_dir = 'D:/ITS_217/data/'
    img_dir = 'D:/ITS_217/img_test/'

    ori_data = scio.loadmat(ori_path)['tensor']

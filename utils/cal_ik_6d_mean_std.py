import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm
import torch
from motion_process import IK_6D, FK_6D, IK_6D_np
from plot_pose import plot_3d_motion
from paramUtil import t2m_kinematic_chain
import torch
#################################################################################
#                                Calculate Mean Std                             #
#################################################################################

def mean_variance(data_dir, save_dir, joints_num):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in tqdm(file_list):
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data[:, :4+(joints_num-1)*3])

    data = np.concatenate(data_list, axis=0)
    data = IK_6D_np(data)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    # Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
    # std 에 0 있으면 안전하게
    np.save(pjoin(save_dir, 'Mean_6d.npy'), Mean)
    np.save(pjoin(save_dir, 'Std_6d.npy'), Std)

    return Mean, Std


if __name__ == '__main__':
    data_dir1 = 'datasets/HumanML3D/new_joint_vecs/'
    save_dir1 = 'datasets/HumanML3D/'
    mean, std = mean_variance(data_dir1, save_dir1, 22)

    # data_dir2 = 'datasets/KIT-ML/new_joint_vecs/'
    # save_dir2 = 'datasets/KIT-ML/'
    # mean2, std2 = mean_variance(data_dir2, save_dir2, 21)
import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm
import torch
from motion_process import IK_np , IK, FK
from plot_pose import plot_3d_motion
from paramUtil import t2m_kinematic_chain
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
    # data = IK_np(data)
    data_torch = torch.from_numpy(data).float().cuda()[None,:]
    with torch.no_grad():
        data_ik = IK(data_torch)
        data_recovered = FK(data_ik)
        print('reconstruction error:', torch.max(torch.abs(data_torch - data_recovered)).item())
        exit()
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    # Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
    # std 에 0 있으면 안전하게
    np.save(pjoin(save_dir, 'Mean_ik.npy'), Mean)
    np.save(pjoin(save_dir, 'Std_ik.npy'), Std)

    return Mean, Std


if __name__ == '__main__':
    data_dir1 = 'datasets/HumanML3D/new_joint_vecs/'
    save_dir1 = 'datasets/HumanML3D/'
    mean, std = mean_variance(data_dir1, save_dir1, 22)

    # data_dir2 = 'datasets/KIT-ML/new_joint_vecs/'
    # save_dir2 = 'datasets/KIT-ML/'
    # mean2, std2 = mean_variance(data_dir2, save_dir2, 21)
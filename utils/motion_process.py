import torch
import torch.nn.functional as F
from typing import Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3

#################################################################################
#                                   Data Params                                 #
#################################################################################
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]
t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]

kit_raw_offsets = np.array(
    [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
     [1, 0, 0], [0, -1, 0], [0, -1, 0], [-1, 0, 0], [0, -1, 0],
     [0, -1, 0], [1, 0, 0], [0, -1, 0], [0, -1, 0], [0, 0, 1],
     [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, -1, 0], [0, 0, 1],
     [0, 0, 1]])
t2m_raw_offsets = np.array([[0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0],
                            [0,-1,0], [0,1,0], [0,-1,0], [0,-1,0], [0,1,0],
                            [0,0,1], [0,0,1], [0,1,0], [1,0,0], [-1,0,0],
                            [0,0,1], [0,-1,0], [0,-1,0], [0,-1,0], [0,-1,0],
                            [0,-1,0], [0,-1,0]])

#################################################################################
#                                  Joints Revert                                #
#################################################################################
def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

#################################################################################
#                                 Motion Plotting                               #
#################################################################################
def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    ani.save(save_path, fps=fps)
    plt.close()

# t2m
bone = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21]]
bone_len = torch.tensor([103.07396697998047, 109.88336944580078, 131.5682373046875, 393.6232604980469, 390.1882019042969, 143.19021606445312, 432.4330749511719, 425.6433410644531, 57.3647346496582, 143.3818359375, 149.41905212402344, 219.36004638671875, 137.4867706298828, 143.38282775878906, 103.03923034667969, 131.61404418945312, 122.9843978881836, 256.8398742675781, 263.0918884277344, 266.0118103027344, 269.8764953613281]).float() / 1000
root_mean = torch.tensor(0.938)
root_std = torch.tensor(0.153)

# # kit
# bone = [[0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [5, 6], [6, 7], [3, 8],
#  [8, 9], [9, 10], [0, 11], [11, 12], [12, 13], [13, 14], [14, 15],
#  [0, 16], [16, 17], [17, 18], [18, 19], [19, 20]]
# bone_len = torch.tensor([80000.0, 120000.0, 44000.015625, 30000.015625, 110000.0, 188000.0, 144999.984375, 110000.0, 188000.0, 144999.984375, 52000.0, 245000.0, 246000.03125, 30273.560546875, 56289.19921875, 52000.0, 245000.0, 246000.03125, 30273.591796875, 56287.2890625]).float() / 1000




def IK(input, offset=None, chain=None):
    offset = t2m_raw_offsets if offset is None else offset
    offset = torch.tensor(offset).float().to(input.device)
    chain = bone if chain is None else chain
    # t2m
    x = input[..., :67].clone()
    #kit
    # x = input[..., :64].clone()

    so3_vecs = []
    batch_size, frames = x.shape[0], x.shape[1]
    root = x[..., :4] # (B, T, 4)
    pose = x[..., 4:].reshape(batch_size, frames,-1, 3) # (B, T, 21, 3)
    zero = torch.zeros_like(pose[..., :1, :])
    zero[..., 1] = root[..., 3:4]

    pose = torch.cat([zero, pose], dim=-2)  # (B, T, 22, 3)
    bones = []
    so3_vecs.append(pose[..., 0, :])  # root
    for i, (p1, p2) in enumerate(chain):
        bone_vec = pose[..., p2, :] - pose[..., p1, :]
        so3_vector = get_so3_from_vectors(offset[p2].to(input.device).expand_as(bone_vec), bone_vec)
        so3_vecs.append(so3_vector)
    so3_vecs = torch.stack(so3_vecs, dim=-2)[..., 1:, :].reshape(batch_size, frames, -1)  # (B, T, 21, 3)
    out = torch.cat([root, so3_vecs], dim=-1)
    return out

def IK_np(input, offset=None, chain=None):
    input = torch.tensor(input).float()[None, ...]
    output = IK(input, offset, chain)[0]
    return output.numpy()

def FK(so3_vecs, offset=None, chain=None):
    offset = t2m_raw_offsets if offset is None else offset
    offset = torch.tensor(offset).float()
    chain = bone if chain is None else chain

    b, f = so3_vecs.shape[0], so3_vecs.shape[1]
    root = so3_vecs[..., :4]
    so3_vecs = so3_vecs[..., 4:].reshape(so3_vecs.shape[0], so3_vecs.shape[1], -1, 3)  # (B, T, 21, 3)
    zero = torch.zeros_like(so3_vecs[..., :1, :])
    zero[..., 1] = root[..., 3:4]
    so3_vecs = torch.cat([zero, so3_vecs], dim=-2)  # (B, T, 22, 3)
    joints = []
    joints.append(so3_vecs[..., 0, :])  # root
    for i in range(so3_vecs.shape[-2] - 1):
        p1, p2 = bone[i]
        joint_pos = joints[p1] + so3_rotate_vector(so3_vecs[..., p2, :], offset[p2].to(so3_vecs.device).expand_as(joints[p1])) * bone_len[i]

        joints.append(joint_pos)
    joints = torch.stack(joints, dim=-2)
    joints = joints[..., 1:, :].reshape(b, f, -1)
    out = torch.cat([root, joints], dim=-1)
    return out


def get_so3_from_vectors(vec_a: torch.Tensor, vec_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_unit = F.normalize(vec_a, p=2, dim=-1)
    b_unit = F.normalize(vec_b, p=2, dim=-1)
    
    cos_theta = torch.sum(a_unit * b_unit, dim=-1)
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

    so3_vector = torch.zeros_like(vec_a)


    antiparallel_mask = cos_theta < (-1.0 + eps)
    if torch.any(antiparallel_mask):
        a_anti = a_unit[antiparallel_mask]
        
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=vec_a.device).expand(a_anti.shape)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=vec_a.device).expand(a_anti.shape)
        is_parallel_to_z = torch.all(torch.isclose(a_anti, z_axis, atol=eps), dim=-1) | \
                           torch.all(torch.isclose(a_anti, -z_axis, atol=eps), dim=-1)
        helper_vec = torch.where(is_parallel_to_z.unsqueeze(-1), y_axis, z_axis)
        perp_axis = F.normalize(torch.cross(a_anti, helper_vec, dim=-1), p=2, dim=-1)
        
        so3_vector[antiparallel_mask] = perp_axis * torch.pi

    general_mask = (cos_theta < (1.0 - eps)) & (cos_theta > (-1.0 + eps))
    if torch.any(general_mask):
        a_gen = a_unit[general_mask]
        b_gen = b_unit[general_mask]
        theta_gen = theta[general_mask].unsqueeze(-1)
        
        axis_gen = F.normalize(torch.cross(a_gen, b_gen, dim=-1), p=2, dim=-1)
        so3_vector[general_mask] = axis_gen * theta_gen

    return so3_vector

def so3_rotate_vector(log_rot: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    original_shape = list(v.shape)
    log_rot = log_rot.contiguous().view(-1, 3)
    v = v.contiguous().view(-1, 3)
    theta = torch.linalg.norm(log_rot, dim=-1, keepdim=True)
    
    mask = theta.squeeze() < eps
    if torch.all(mask):
        return v.view(original_shape)
        
    axis = log_rot / (theta + eps)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cross_product = torch.cross(axis, v, dim=-1)
    dot_product = torch.sum(axis * v, dim=-1, keepdim=True)
    dot_product_term = axis * dot_product
    
    v_rot = v * cos_theta + cross_product * sin_theta + dot_product_term * (1 - cos_theta)
    v_rot[mask] = v[mask]
    return v_rot.view(original_shape)

# ----------------------------------------------------------------------------------
# 6D <-> rotation matrix helpers (Zhou et al. 2019)
# ----------------------------------------------------------------------------------

def _skew_symmetric_matrix_batch(v: torch.Tensor) -> torch.Tensor:
    batch_shape = v.shape[:-1]
    O = torch.zeros(batch_shape, device=v.device, dtype=v.dtype)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    K = torch.stack([
        torch.stack([ O, -vz,  vy], dim=-1),
        torch.stack([ vz,  O, -vx], dim=-1),
        torch.stack([-vy,  vx,  O], dim=-1),
    ], dim=-2)
    return K

def get_rotation_matrix(vec1: torch.Tensor, vec2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    was_single_vector = vec1.dim() == 1
    if was_single_vector:
        vec1, vec2 = vec1.unsqueeze(0), vec2.unsqueeze(0)

    a = F.normalize(vec1, p=2, dim=-1)
    b = F.normalize(vec2, p=2, dim=-1)
    cos_angle = torch.sum(a * b, dim=-1, keepdim=True)
    
    parallel_mask = (1.0 - cos_angle).abs() < eps
    antiparallel_mask = (1.0 + cos_angle).abs() < eps

    v = torch.cross(a, b, dim=-1)
    K = _skew_symmetric_matrix_batch(v)
    sin_sq_angle = 1.0 - cos_angle**2
    denom = torch.clamp(sin_sq_angle, min=eps)
    
    I = torch.eye(3, device=a.device, dtype=a.dtype).expand_as(K)
    factor = ((1 - cos_angle) / denom).unsqueeze(-1)
    R_general = I + K + (K @ K) * factor

    z = torch.tensor([0.0, 0.0, 1.0], device=a.device, dtype=a.dtype).expand_as(a)
    y = torch.tensor([0.0, 1.0, 0.0], device=a.device, dtype=a.dtype).expand_as(a)
    near_z_mask = (a[..., 2].abs() > 0.99).unsqueeze(-1)
    helper = torch.where(near_z_mask, y, z)
    axis_ap = F.normalize(torch.cross(a, helper, dim=-1), p=2, dim=-1)
    K_ap = _skew_symmetric_matrix_batch(axis_ap)
    R_ap = I + 2.0 * (K_ap @ K_ap)

    R = torch.where(antiparallel_mask.unsqueeze(-1), R_ap, R_general)
    R = torch.where(parallel_mask.unsqueeze(-1), I, R)

    if was_single_vector:
        R = R.squeeze(0)
    return R


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    c1 = R[..., :, 0]
    c2 = R[..., :, 1]
    return torch.cat([c1, c2], dim=-1)

def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]

    r1 = F.normalize(a1, dim=-1)
    a2_ortho = a2 - torch.sum(r1 * a2, dim=-1, keepdim=True) * r1
    r2 = F.normalize(a2_ortho, dim=-1)
    r3 = torch.cross(r1, r2, dim=-1)

    return torch.stack([r1, r2, r3], dim=-1)

def rot6d_rotation(d6: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    R = rot6d_to_matrix(d6)
    
    v_rot = torch.matmul(R, v.unsqueeze(-1)).squeeze(-1)
    return v_rot




def IK_6D(input, offset=None, chain=None):
    offset = t2m_raw_offsets if offset is None else offset
    offset = torch.tensor(offset).float().to(input.device)
    chain = bone if chain is None else chain
    # t2m
    x = input[..., :67].clone()
    #kit
    # x = input[..., :64].clone()

    d3_vecs = []
    batch_size, frames = x.shape[0], x.shape[1]
    root = x[..., :4] # (B, T, 4)
    pose = x[..., 4:].reshape(batch_size, frames,-1, 3) # (B, T, 21, 3)
    zero = torch.zeros_like(pose[..., :1, :])
    zero[..., 1] = root[..., 3:4]

    pose = torch.cat([zero, pose], dim=-2)  # (B, T, 22, 3)
    bones = []
    for i, (p1, p2) in enumerate(chain):
        bone_vec = pose[..., p2, :] - pose[..., p1, :]
        d3_vector = matrix_to_rot6d(get_rotation_matrix(offset[p2].to(input.device).expand_as(bone_vec), bone_vec))
        d3_vecs.append(d3_vector)
    d3_vecs = torch.stack(d3_vecs, dim=-2).reshape(batch_size, frames, -1)  # (B, T, 21, 3)
    out = torch.cat([root, d3_vecs], dim=-1)
    return out


def FK_6D(d3_vecs, offset=None, chain=None):
    offset = t2m_raw_offsets if offset is None else offset
    offset = torch.tensor(offset).float()
    chain = bone if chain is None else chain

    b, f = d3_vecs.shape[0], d3_vecs.shape[1]
    root = d3_vecs[..., :4]
    d3_vecs = d3_vecs[..., 4:].reshape(d3_vecs.shape[0], d3_vecs.shape[1], -1, 6)  # (B, T, 21, 6)
    zero = torch.zeros_like(d3_vecs[..., :1, :3])
    zero[..., 1] = root[..., 3:4]
    joints = []
    joints.append(zero[..., 0, :])  # root
    for i in range(d3_vecs.shape[-2]):
        p1, p2 = bone[i]
        joint_pos = joints[p1] + rot6d_rotation(d3_vecs[..., p2-1, :], offset[p2].to(d3_vecs.device).expand_as(joints[p1])) * bone_len[i]
        joints.append(joint_pos)
    
    joints = torch.stack(joints, dim=-2)
    joints = joints[..., 1:, :].reshape(b, f, -1)
    out = torch.cat([root, joints], dim=-1)
    return out

# ----------------------------------------------------------------------------------
# Convenience NumPy wrappers (parity with your helpers)
# ----------------------------------------------------------------------------------

def IK_6D_np(x_in, offset=None, chain=None):
    x = torch.tensor(x_in).float().unsqueeze(0)
    y = IK_6D(x, offset, chain)[0]
    return y.detach().cpu().numpy()


def FK_6D_np(rots6d, offset=None, chain=None):
    x = torch.tensor(rots6d).float().unsqueeze(0)
    y = FK_6D(x, offset, chain)[0]
    return y.detach().cpu().numpy()



if __name__ == "__main__":
    a = torch.randn(10,3)
    a = F.normalize(a, dim=-1, p=2)
    b = torch.randn(10,3)
    b = F.normalize(b, dim=-1, p=2)
    R = get_rotation_matrix(a, b)
    d6 = matrix_to_rot6d(R)
    
    v_rot = rot6d_rotation(d6, a)
    print(b)
    print(v_rot)  # b와 동일해야 함
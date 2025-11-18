import os
import numpy as np
from scipy import linalg
import torch
from utils.motion_process import recover_from_ric, FK, IK, FK_6D
from tqdm import tqdm

#################################################################################
#                               Eval Function Loops                             #
#################################################################################
@torch.no_grad()
def evaluation(out_dir, val_loader, ema_mardm, ae, writer, ep, best_fid, best_div,
                        best_top1, best_top2, best_top3, best_matching, eval_wrapper, device, clip_score_old, time_steps=None,
                        cond_scale=None, temperature=1, cal_mm=False, train_mean=None, train_std=None, plot_func=None,
                        draw=True, hard_pseudo_reorder=False, save_anim=False, space='ik'):

    ema_mardm.eval()
    if ae is not None:
        ae.eval()

    save=False
    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    if time_steps is None: time_steps = 18
    if cond_scale is None:
        if "kit" in out_dir:
            cond_scale = 2.5
        else:
            cond_scale = 4.5
    clip_score_real = 0
    clip_score_gt = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(tqdm(val_loader)):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.to(device)

        bs, seq = pose.shape[:2]
        if i < num_mm_batch:
            motion_multimodality_batch = []
            batch_clip_score_pred = 0
            for _ in tqdm(range(30)):

                
                if ae is None:
                    pred_latents = ema_mardm.generate(clip_text, m_length, cond_scale)
                    pred_motions = pred_latents
                else:
                    pred_latents = ema_mardm.generate(clip_text, m_length//4, cond_scale)
                    pred_latents = pred_latents * ae.std.unsqueeze(1) + ae.mean.unsqueeze(1)
                    pred_motions = ae.decode(pred_latents.permute(0, 2, 1))
            
                pred_motions = pred_motions * torch.tensor(train_std).cuda() + torch.tensor(train_mean).cuda()
                if space == 'ik':
                    pred_motions = FK(pred_motions)
                elif space == '6d':
                    pred_motions = FK_6D(pred_motions)
                pred_motions = val_loader.dataset.transform(pred_motions.detach().cpu().numpy())
                (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                                            clip_text,
                                                                            torch.from_numpy(pred_motions).to(device),
                                                                            m_length)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
            for j in range(32):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred

        else:
            if ae is None:
                pred_latents = ema_mardm.generate(clip_text, m_length, cond_scale)
                pred_motions = pred_latents
            else:
                pred_latents = ema_mardm.generate(clip_text, m_length//4, cond_scale)
                pred_latents = pred_latents * ae.std.unsqueeze(1) + ae.mean.unsqueeze(1)
                pred_motions = ae.decode(pred_latents.permute(0, 2, 1))
                
            pred_motions = pred_motions * torch.tensor(train_std).cuda() + torch.tensor(train_mean).cuda()
            if space == 'ik':
                pred_motions = FK(pred_motions)
            elif space == '6d':
                pred_motions = FK_6D(pred_motions)
            pred_motions = val_loader.dataset.transform(pred_motions.detach().cpu().numpy())
            (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings,
                                                                              pos_one_hots, sent_len,
                                                                              clip_text,
                                                                              torch.from_numpy(pred_motions).to(device),
                                                                              m_length)
            batch_clip_score_pred = 0
            for j in range(32):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred

        pose = pose.cuda().float()
        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text,
                                                          pose.clone(), m_length)
        batch_clip_score = 0
        for j in range(32):
            single_em = em_clip[j]
            single_et = et_clip[j]
            clip_score = (single_em @ single_et.T).item()
            batch_clip_score += clip_score
        clip_score_gt += batch_clip_score
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)
        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    clip_score_real = clip_score_real / nb_sample
    clip_score_gt = clip_score_gt / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep/Re {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred} multimodality. {multimodality:.4f} clip score real. {clip_score_gt} clip score. {clip_score_real}"
    print(msg)

    if draw:
        writer.log({'Test/FID': fid, 'epoch':ep})
        writer.log({'Test/Diversity': diversity, 'epoch':ep})
        writer.log({'Test/top1': R_precision[0], 'epoch':ep})
        writer.log({'Test/top2': R_precision[1], 'epoch':ep})
        writer.log({'Test/top3': R_precision[2], 'epoch':ep})
        writer.log({'Test/matching_score': matching_score_pred, 'epoch':ep})
        writer.log({'Test/clip_score': clip_score_real, 'epoch':ep})


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        if draw: print(msg)
        best_fid, best_ep = fid, ep
        save=True


    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        if draw: print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        if draw: print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        if draw: print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        if draw: print(msg)
        best_top3 = R_precision[2]

    if clip_score_real > clip_score_old:
        msg = f"--> --> \t CLIP-score Improved from {clip_score_old:.4f} to {clip_score_real:.4f} !!!"
        if draw: print(msg)
        clip_score_old = clip_score_real

    if save_anim and plot_func is not None:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx]
        data = torch.from_numpy(val_loader.dataset.inv_transform(data))
        data = recover_from_ric(data, 22).numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(os.path.dirname(out_dir), 'ani', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        plot_func(data, save_dir, captions, lengths)
        data = pose[rand_idx]
        data = val_loader.dataset.inv_transform(data.cpu().numpy())
        data = recover_from_ric(torch.from_numpy(data), 22).numpy()
        save_dir = os.path.join(os.path.dirname(save_dir), 'GT%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        plot_func(data, save_dir, captions, lengths)


    if cal_mm:
        return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, clip_score_old, writer, save
    else:
        return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, 0, clip_score_old, writer, save
    

#################################################################################
#                                 Util Functions                                #
#################################################################################
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

#################################################################################
#                                     Metrics                                   #
#################################################################################
def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist



def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
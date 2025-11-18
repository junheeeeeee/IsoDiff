import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models.load_model import DM_models
from utils.evaluators import Evaluators
from utils.datasets import Text2MotionDataset, collate_fn
from utils.eval_utils import evaluation
import argparse

def main(args):
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # setting this to true significantly increase training and sampling speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    #################################################################################
    #                                    Eval Data                                  #
    #################################################################################
    if args.dataset_name == "t2m":
        data_root = f'{args.dataset_dir}/HumanML3D/'
        dim_pose = 67
    else:
        data_root = f'{args.dataset_dir}/KIT-ML/'
        dim_pose = 64
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    text_dir = pjoin(data_root, 'texts')
    max_motion_length = 196
    if args.space == 'ik':
        mean = np.load(pjoin(data_root, 'Mean_ik.npy'))
        std = np.load(pjoin(data_root, 'Std_ik.npy'))
    elif args.space == '6d':
        mean = np.load(pjoin(data_root, 'Mean_6d.npy'))
        std = np.load(pjoin(data_root, 'Std_6d.npy'))
        dim_pose = 2 * (dim_pose - 4) + 4
    else:
        mean = np.load(pjoin(data_root, 'Mean.npy'))
        std = np.load(pjoin(data_root, 'Std.npy'))
    
    # mean = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_mean.npy')
    # std = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_std.npy')
    eval_mean = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_mean.npy')
    eval_std = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_std.npy')
    split_file = pjoin(data_root, 'test.txt')
    eval_dataset = Text2MotionDataset(eval_mean, eval_std, split_file, args.dataset_name, motion_dir, text_dir,
                                      4, max_motion_length, 20, evaluation=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True,
                             collate_fn=collate_fn, shuffle=True)
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')

    ema_mardm = DM_models[args.model](input_dim=dim_pose, cond_mode='text')
    model_dir = os.path.join(model_dir, 'latest.tar' ) # 'latest.tar', 'net_best_fid.tar'
    checkpoint = torch.load(model_dir, map_location='cpu')      
    missing_keys2, unexpected_keys2 = ema_mardm.load_state_dict(checkpoint['ema_mardm'], strict=False)
    assert len(unexpected_keys2) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys2])


    eval_wrapper = Evaluators(args.dataset_name, device=device)
    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    out_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'eval')
    os.makedirs(out_dir, exist_ok=True)
    f = open(pjoin(out_dir, f'eval_{args.cfg}.log'), 'w')

    ema_mardm.eval()
    ema_mardm.to(device)

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mm = []
    clip_scores = []

    repeat_time = 20
    for i in range(repeat_time):
        with torch.no_grad():
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mm, clip_score = 1000, 0, 0, 0, 0, 100, 0, -1
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mm, clip_score, writer, save_now = evaluation(
                model_dir, eval_loader, ema_mardm, None, None, i, best_fid=best_fid, clip_score_old=clip_score,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching, eval_wrapper=eval_wrapper, device=device, train_mean=mean, train_std=std,
                time_steps=args.time_steps, cond_scale=args.cfg, temperature=args.temperature, cal_mm=args.cal_mm,
                draw=False, hard_pseudo_reorder=args.hard_pseudo_reorder, space=args.space)
        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)
        mm.append(best_mm)
        clip_scores.append(clip_score)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mm = np.array(mm)
    clip_scores = np.array(clip_scores)

    print(f'final result:')
    print(f'final result:', file=f, flush=True)

    msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n" \
                f"\tCLIP-Score:{np.mean(clip_scores):.3f}, conf.{np.std(clip_scores) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
    print(msg_final)
    print(msg_final, file=f, flush=True)
    f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='IsoDiff_default')
    parser.add_argument('--model', type=str, default='IsoDiff')
    parser.add_argument("--space", type=str, default='ik', help="features space: xyz, ik, 6d")
    parser.add_argument('--dataset_name', type=str, default='t2m')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument("--time_steps", default=18, type=int)
    parser.add_argument("--cfg", default=4.5, type=float)
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument('--cal_mm', action="store_true")
    parser.add_argument('--hard_pseudo_reorder', action="store_true")

    arg = parser.parse_args()
    main(arg)
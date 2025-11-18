import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import torch.optim as optim
from models.load_model import DM_models
from utils.evaluators import Evaluators
from utils.datasets import Text2MotionDataset, collate_fn
import time
import copy
from collections import OrderedDict, defaultdict
from utils.train_utils import update_lr_warm_up, def_value, save, print_current_loss, update_ema
from utils.eval_utils import evaluation
import argparse
from utils.plot_pose import plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain
import wandb


def main(args):
    def plot_t2m(data, save_dir, captions, m_lengths):

        # print(ep_curves.shape)
        for ii, (caption, joint_data) in enumerate(zip(captions, data)):
            joint = joint_data[:m_lengths[ii]]
            save_path = pjoin(save_dir, '%02d.mp4'%ii)
            plot_3d_motion(save_path, t2m_kinematic_chain, joint, title=caption, fps=20, radius=4)
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.autograd.set_detect_anomaly(True)
    # setting this to true significantly increase training and sampling speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    #################################################################################
    #                                    Train Data                                 #
    #################################################################################
    if args.dataset_name == "t2m":
        data_root = f'{args.dataset_dir}/HumanML3D/'
        dim_pose = 67
    else:
        data_root = f'{args.dataset_dir}/KIT-ML/'
        dim_pose = 64
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    text_dir = pjoin(data_root, 'texts')
    if arg.space == 'ik':
        print("Using IK features for training")
        mean = np.load(pjoin(data_root, 'Mean_ik.npy'))
        std = np.load(pjoin(data_root, 'Std_ik.npy'))
    elif arg.space == '6d':
        print("Using 6d features for training")
        mean = np.load(pjoin(data_root, 'Mean_6d.npy'))
        std = np.load(pjoin(data_root, 'Std_6d.npy'))
        dim_pose = 2 * (dim_pose - 4) + 4
    else:
        print("Using raw features for training")
        mean = np.load(pjoin(data_root, 'Mean.npy'))
        std = np.load(pjoin(data_root, 'Std.npy'))
        # mean *= 0
        # std *= 0
        # std += 1
    train_split_file = pjoin(data_root, 'train.txt')
    val_split_file = pjoin(data_root, 'val.txt')

    train_dataset = Text2MotionDataset(mean, std, train_split_file, args.dataset_name, motion_dir, text_dir,
                                          args.unit_length, args.max_motion_length, 20, evaluation=False, space = args.space)
    val_dataset = Text2MotionDataset(mean, std, val_split_file, args.dataset_name, motion_dir, text_dir,
                                          args.unit_length, args.max_motion_length, 20, evaluation=False, space = args.space)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                            shuffle=True)

    #################################################################################
    #                                    Eval Data                                  #
    #################################################################################
    if args.need_evaluation:
        eval_mean = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_mean.npy')
        eval_std = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_std.npy')
        split_file = pjoin(data_root, 'val.txt')
        eval_dataset = Text2MotionDataset(eval_mean, eval_std, split_file, args.dataset_name, motion_dir, text_dir,
                                          4, 196, 20, evaluation=True)
        eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True,
                                 collate_fn=collate_fn, shuffle=True)
        log_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        logger = wandb.init(project="DM", dir=log_dir, name=args.name if hasattr(args, 'name') else None)
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DM_models[args.model](input_dim=dim_pose, cond_mode='text')
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)

    all_params = 0
    pc_transformer = sum(param.numel() for param in
                         [p for name, p in model.named_parameters() if not name.startswith('clip_model.')])
    all_params += pc_transformer
    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))


    if args.need_evaluation:
        eval_wrapper = Evaluators(args.dataset_name, device=device)
    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    model.to(device)
    ema_model.to(device)

    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.99), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)

    epoch = 0
    it = 0
    if args.is_continue:
        model_dir = pjoin(model_dir, 'latest.tar')
        checkpoint = torch.load(model_dir, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        missing_keys2, unexpected_keys2 = ema_model.load_state_dict(checkpoint['ema_mardm'], strict=False)
        assert len(unexpected_keys) == 0
        assert len(unexpected_keys2) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])
        assert all([k.startswith('clip_model.') for k in missing_keys2])
        optimizer.load_state_dict(checkpoint['opt_model'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch, it = checkpoint['ep'], checkpoint['total_it']
        print("Load model epoch:%d iterations:%d" % (epoch, it))

    
    total_iters = args.epoch * len(train_loader)
    print(f'Total Epochs: {args.epoch}, Total Iters: {total_iters}')
    print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))

    logs = defaultdict(def_value, OrderedDict())

    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, clip_score = 1000, 0, 0, 0, 0, 100, -1
    worst_loss = 100
    start_time = time.time()
    while epoch < args.epoch:
        model.train()

        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < args.warm_up_iter:
                update_lr_warm_up(it, args.warm_up_iter, optimizer, args.lr)

            conds, motion, m_lens = batch_data
            motion = motion.detach().float().to(device)
            # mask = lengths_to_mask(m_lens, max_len=motion.shape[1]).to(device)
            # a = FK(IK(motion.clone())) * mask.unsqueeze(-1)
        
            # print((abs(a - motion)).max())
            # exit()
            m_lens = m_lens.detach().long().to(device)

            conds = conds.to(device).float() if torch.is_tensor(conds) else conds

            latent = motion
            loss = model.forward_loss(latent, conds, m_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            logs['loss'] += loss.item()
            logs['lr'] += optimizer.param_groups[0]['lr']
            update_ema(model, ema_model, 0.9999)

            mean_loss = OrderedDict()
            for tag, value in logs.items():
                if args.need_evaluation:
                    logger.log({'Train/%s' % tag: value, 'step':it})
                mean_loss[tag] = value
            logs = defaultdict(def_value, OrderedDict())
            print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i, it_max = len(train_loader))

        save(pjoin(model_dir, 'latest.tar'), epoch, model, optimizer, scheduler,
             it, 'model', ema_mardm=ema_model)
        epoch += 1
        #################################################################################
        #                                      Eval Loop                                #
        #################################################################################
        print('Validation time:')
        model.eval()
        val_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                conds, motion, m_lens = batch_data
                motion = motion.detach().float().to(device)
                m_lens = m_lens.detach().long().to(device)

                conds = conds.to(device).float() if torch.is_tensor(conds) else conds

           
                latent = motion

                loss = model.forward_loss(latent, conds, m_lens)
                val_loss.append(loss.item())

        print(f"Validation loss:{np.mean(val_loss):.3f}")
        if args.need_evaluation:
            logger.log({'Val/loss': np.mean(val_loss), 'epoch':epoch-1})
        if np.mean(val_loss) < worst_loss:
            print(f"Improved loss from {worst_loss:.02f} to {np.mean(val_loss)}!!!")
            worst_loss = np.mean(val_loss)
        if args.need_evaluation:
            if epoch % 10 == 0:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _, clip_score, writer, save_now= evaluation(
                    model_dir, eval_loader, ema_model, None, logger, epoch-1, best_fid=best_fid, clip_score_old=clip_score,
                    best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, eval_wrapper=eval_wrapper, device=device, train_mean=mean, train_std=std, plot_func= plot_t2m, space=args.space)
                if save_now:
                    save(pjoin(model_dir, 'net_best_fid.tar'), epoch-1, model, optimizer, scheduler,
                        it, 'model', ema_mardm=ema_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='IsoDiff_default')
    parser.add_argument('--model', type=str, default='IsoDiff', help="model name")
    parser.add_argument('--dataset_name', type=str, default='t2m')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument("--space", type=str, default='ik', help="features space: xyz, ik means so(3), 6d")
    parser.add_argument("--max_motion_length", type=int, default=196)
    parser.add_argument("--unit_length", type=int, default=4)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--warm_up_iter', default=2000, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--milestones', default=[50_000], nargs="+", type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)

    parser.add_argument('--diffmlps_batch_mul', type=int, default=4)
    parser.add_argument('--need_evaluation', action="store_true" )

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--is_continue', action="store_true")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    parser.add_argument('--log_every', default=50, type=int)

    arg = parser.parse_args()
    main(arg)
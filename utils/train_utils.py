import torch
import math
import time

#################################################################################
#                                  Util Functions                               #
#################################################################################
def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask #(b, len)


def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask


def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


def update_ema(model, ema_model, ema_decay):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=(1 - ema_decay))

#################################################################################
#                                Logging Functions                              #
#################################################################################
def def_value():
    return 0.0


def update_lr_warm_up(nb_iter, warm_up_iter, optimizer, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return current_lr


def save(file_name, ep, model, optimizer, scheduler, total_it, name, ema_mardm=None):
    state = {
        name: model.state_dict(),
        f"opt_{name}": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        'ep': ep,
        'total_it': total_it,
    }
    if ema_mardm is not None:
        mardm_state_dict = model.state_dict()
        ema_mardm_state_dict = ema_mardm.state_dict()
        clip_weights = [e for e in mardm_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del mardm_state_dict[e]
            del ema_mardm_state_dict[e]
        state[name] = mardm_state_dict
        state["ema_mardm"] = ema_mardm_state_dict
    torch.save(state, file_name)


def print_current_loss(start_time, niter_state, total_niters, losses, epoch=None, sub_epoch=None,
                       inner_iter=None, it_max=None ,tf_ratio=None, sl_steps=None):

    def as_time(s):
        d = int(s // 86400)
        h = int((s % 86400) // 3600)
        m = int((s % 3600) // 60)
        s = int(s % 60)
        parts = []
        if d > 0:
            parts.append(f"{d}d")
        if h > 0 or (d > 0 and (m > 0 or s > 0)):
            parts.append(f"{h}h")
        if m > 0 or ((d > 0 or h > 0) and s > 0):
            parts.append(f"{m}m")
        if s > 0 or (d == 0 and h == 0 and m == 0):
            parts.append(f"{s}s")
        return ' '.join(parts)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        if percent == 0:
            es = 0
            rs = 0
        else:
            es = s / percent
            rs = es - s
        return '%s (- %s)' % (as_time(s), as_time(rs))

    if epoch is not None:
        print('\r', end='')  # Clear the current line
        print('ep/it:%2d-%4d(%.1f%%)' % (epoch, inner_iter, inner_iter / it_max * 100), end=" ")

    message = ' %s completed:%.2f%%' % (time_since(start_time, niter_state / total_niters), niter_state / total_niters * 100)

    for k, v in losses.items():
        if k == "lr":
            message += f' {k}: {v} '
        elif k == "acc":
            message += ' %s: %.1f mm' % (k, v)
        else:
            message += ' %s: %.4f ' % (k, v)

    print(message, end='', flush=True)  # Update the line without adding a new one
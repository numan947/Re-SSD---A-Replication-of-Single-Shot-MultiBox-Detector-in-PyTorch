import torch.nn as nn
from commons import *
from model import MultiBoxLoss, SSD300
from data import PascalDataset
from tqdm import tqdm
import pandas as pd
import time, math, datetime, gc
from torch.optim.optimizer import Optimizer, required
import numpy as np
import matplotlib.pyplot as plt
from apex import amp
import apex

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss

class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)


                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup = 0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup = warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """Coding Credit: https://github.com/Bjarten/early-stopping-pytorch"""
    def __init__(self, track="min", patience=7, verbose=False, delta=0, save_model_name="checkpoint.pt"):
        """
        Args:
            track (str): What to track, possible value: min, max (e.g. min validation loss, max validation accuracy (%))
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.track = track
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_best = -np.Inf
        if self.track=="min":
            self.val_best = np.Inf
        self.delta = delta
        self.save_model_name = save_model_name

    def __call__(self, current_score, model):

        score = -current_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(current_score, model)
        elif ((score < self.best_score + self.delta) and self.track=="min") or ((score>self.best_score+self.delta) and self.track=="max"):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(current_score, model)
            self.counter = 0

    def save_checkpoint(self, new_best, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Found better solution ({self.val_best:.6f} --> {new_best:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_model_name)
        self.val_best = new_best

###############################################################################################################################
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def adjust_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*scale
    return optimizer

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']/2.0

def clear_cuda():
    torch.cuda.empty_cache()
    gc.collect()

def print_epoch_stat(epoch_idx, time_elapsed_in_seconds, history=None, train_loss=None, train_accuracy=None, valid_loss=None, valid_accuracy=None):
    print("\n\nEPOCH {} Completed, Time Taken: {}".format(epoch_idx+1, datetime.timedelta(seconds=time_elapsed_in_seconds)))
    if train_loss is not None:
        if history is not None:
            history.loc[epoch_idx, "train_loss"] = train_loss
        print("\tTrain Loss \t{:0.9}".format(train_loss))
    if train_accuracy is not None:
        if history is not None:
            history.loc[epoch_idx, "train_accuracy"] = 100.0*train_accuracy
        print("\tTrain Accuracy \t{:0.9}%".format(100.0*train_accuracy))
    if valid_loss is not None:
        if history is not None:
            history.loc[epoch_idx, "valid_loss"] = valid_loss
        print("\tValid Loss \t{:0.9}".format(valid_loss))
    if valid_accuracy is not None:
        if history is not None:
            history.loc[epoch_idx, "valid_accuracy"] = 100.0*valid_accuracy
        print("\tValid Accuracy \t{:0.9}%".format(100.0*valid_accuracy))

    return history


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def train_single_epoch(epoch, model, train_loader, optimizer, criterion, clip_grad=1.0, plot=True, accumulation_factor=1):
    model.train()
    avgl,avga = 0.0, 0.0

    if plot:
        plt.figure(figsize=(13,13))

    accumulated_loss = 0.0
    cnt = 0
    optimizer.zero_grad()
    for i, (images, boxes, labels, diffs) in enumerate(tqdm(train_loader, total=len(train_loader))):
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        predicted_locs, predicted_scores = model(images)

        loss = criterion(predicted_locs, predicted_scores, boxes, labels)
        accumulated_loss+=loss.item()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        cnt+=1

        if (i+1)%accumulation_factor == 0:
            if plot:
                plot_grad_flow(model.named_parameters())

            if clip_grad is not None:
                # clip_gradient(optimizer, clip_grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()
            optimizer.zero_grad()
            avgl+=(accumulated_loss/accumulation_factor)
            accumulated_loss = 0.0
            cnt=0

    if accumulated_loss>0.0:
        if plot:
            plot_grad_flow(model.named_parameters())

        if clip_grad is not None:
            clip_gradient(optimizer, clip_grad)

        optimizer.step()
        optimizer.zero_grad()
        avgl+=(accumulated_loss/cnt)
        accumulated_loss = 0.0

    if plot:
        plt.tight_layout()
        plt.savefig("./output/grad_flow/"+str(epoch)+".png", bbox_inches='tight')
        plt.close()

    return avgl/len(train_loader), avga/len(train_loader)


def evaluate(model, valid_loader, criterion):
    model.eval()
    avgl,avga = 0.0, 0.0

    with torch.no_grad():
        for images, boxes, labels, diffs in tqdm(valid_loader, total=len(valid_loader)):
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            predicted_locs, predicted_scores = model(images)

            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            avgl+=loss.item()

    return avgl/len(valid_loader), avga/len(valid_loader)


def train(seed, resume=False, opt_level='O1', ignore_first=True):
    clear_cuda()
    torch.manual_seed(seed)
    data_folder = "./output/"
    keep_difficult = True
    n_classes = len(label_map)

    other_checkpoint = data_folder+"oth_checkpoint.pt"
    model_checkpoint = data_folder+"checkpoint.pt"
    early_stopping = EarlyStopping(save_model_name=model_checkpoint, patience=3)
    pin_memory = True

    batch_size = 8
    accumulation_factor = 2
    iterations = 100000
    workers = 4*torch.cuda.device_count()
    lr = 1e-3
    decay_lr_at = [80000]
    decay_lr_to = 0.2

    early_stopper_lr_decrease = 0.5
    early_stopper_lr_decrease_count = 7

    momentum = 0.9
    weight_decay = 5e-3
    grad_clip = None
    torch.backends.cudnn.benchmark = True

    optimizer = None
    start_epoch = None

    history = pd.DataFrame()
    history.index.name="Epoch"
    history_path = "./output/HISTORY"+"_SEED_{}".format(seed)+".csv"
    model = SSD300(n_classes)
    biases = list()
    not_biases = list()

    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    # optimizer = RAdam(params=[{'params':biases,'lr':2*lr},{'params':not_biases}], lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.RMSprop(params=[{'params':biases,'lr':2*lr},{'params':not_biases}], lr=lr, weight_decay=weight_decay, momentum=momentum)
    # optimizer = torch.optim.SGD(params=[{'params':biases,'lr':2*lr},{'params':not_biases}], lr=lr, weight_decay=weight_decay, momentum=momentum)
    optimizer = apex.optimizers.FusedLAMB(params=[{'params':biases,'lr':2*lr},{'params':not_biases}], lr=lr, weight_decay=weight_decay)#, momentum=momentum)
    
    model = model.to(device)
    criterion = MultiBoxLoss(model.priors_cxcy).to(device)
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level, loss_scale=1.0)

    start_epoch = 0
    

    if resume and os.path.exists(other_checkpoint):
        ocheckpt = torch.load(other_checkpoint, map_location=device)
        optimizer.load_state_dict(ocheckpt['optimizer_state'])
        start_epoch = ocheckpt['epoch'] + 1
        lr = get_lr(optimizer)
        history = pd.read_csv(history_path, index_col="Epoch")
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        # adjust_learning_rate(optimizer, 100.0)
        lr = get_lr(optimizer)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

            amp.load_state_dict(ocheckpt['amp_state'])





    train_dataset = PascalDataset(data_folder, split="train", keep_difficult=keep_difficult, resize_dims=(500,500))
    train_len = int(0.9*len(train_dataset))
    valid_len = len(train_dataset) - train_len

    train_data, valid_data = torch.utils.data.dataset.random_split(train_dataset, [train_len, valid_len])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn,
                                               num_workers=workers, pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=train_dataset.collate_fn,
                                               num_workers=workers, pin_memory=pin_memory)



    

    total_epochs = iterations//(len(train_dataset)//32)
    decay_lr_at = [it//(len(train_dataset)//32) for it in decay_lr_at]

    total_epochs = 125
    decay_lr_at = [100]


    print("Training For: {}".format(total_epochs))
    print("Decay LR at:", decay_lr_at)

    for epoch in range(start_epoch, total_epochs):
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)
            lr*=decay_lr_to
            print("Learning Rate Adjusted")
        st = time.time()

        print("EPOCH: {}/{} -- Current LR: {}".format(epoch+1, total_epochs, lr))
        clear_cuda()
        tl,ta = train_single_epoch(epoch, model, train_loader, optimizer, criterion, plot=False, clip_grad=grad_clip, accumulation_factor=accumulation_factor)
        clear_cuda()
        vl, va = evaluate(model, valid_loader, criterion)


        print_epoch_stat(epoch, time.time()-st, history=history, train_loss=tl, valid_loss=vl)
        early_stopping(tl, model)
        other_state = {
            'epoch': epoch,
            'optimizer_state': optimizer.state_dict(),
            'amp_state': amp.state_dict()
            }

        torch.save(other_state, other_checkpoint)

        history.loc[epoch, "LR"] = lr
        history.to_csv(history_path)

        if early_stopping.early_stop:
            if early_stopper_lr_decrease_count>0:
                early_stopper_lr_decrease_count = early_stopper_lr_decrease_count-1
                adjust_learning_rate(optimizer, early_stopper_lr_decrease)
                early_stopping.early_stop = False
                early_stopping.counter = 0
                lr*=early_stopper_lr_decrease
                print("Learning Rate Adjusted")
            else:
                break







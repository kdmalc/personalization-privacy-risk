def zero_grad(self):
        self.grad = zero_copy(self.model,self.rnn)
def zero_copy(model,rnn=False):
    if rnn:
        model.hidden = None
    tmp_model = deepcopy(model)
    for tp in tmp_model.parameters():
        tp.data = torch.zeros_like(tp.data)
    if rnn:
        model.init_hidden()
    return tmp_model
################################################################################################# 
def is_sync_fed(args):
    if args.federated_sync_type == 'local_step':
        local_step = get_current_local_step(args)
        return args.local_index % local_step == 0
    elif args.federated_sync_type == 'epoch':
        return args.epoch_ % args.num_epochs_per_comm == 0
    else:
        raise NotImplementedError
#################################################################################################
def update_client_epoch(args):
    args.client_epoch_total += args.local_index / args.num_batches_train_per_device_per_epoch
    return
#################################################################################################
def get_current_local_step(args):
    """design a specific local step adjustment schme based on lr_decay_by_epoch"""
    try:
        return args.local_steps[args.epoch]
    except:
        return args.local_steps[-1]
#################################################################################################
def alpha_update(model_local, model_personal,alpha, eta):
    grad_alpha = 0
    for l_params, p_params in zip(model_local.parameters(), model_personal.parameters()):
        dif = p_params.data - l_params.data
        grad = alpha * p_params.grad.data + (1-alpha)*l_params.grad.data
        grad_alpha += dif.view(-1).T.dot(grad.view(-1))
    
    grad_alpha += 0.02 * alpha
    alpha_n = alpha - eta*grad_alpha
    alpha_n = np.clip(alpha_n.item(),0.0,1.0)
    return alpha_n
    
    
    

## scheduler.py

def adjust_learning_rate(args, optimizer, lr_scheduler, lr_external=None):
    """Sets the learning rate to the initial LR decayed by # of accessed sample
        We should decay the learning rate based on the number of samples that
        we have accessed.
    """
    # adjust and assign learning rate.
    if lr_external is None:
        lr = lr_scheduler(args.epoch_)

        if lr is None:
            lr = args.old_learning_rate

        if args.old_learning_rate != lr:
            args.old_learning_rate = lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_external
        lr = lr_external
    return lr
    



## eval.py

def inference(model, criterion, metrics, _input, _target, classes=None, rnn=False):
    """Inference on the given model and get loss and accuracy."""
    output = model(_input)
    loss = criterion(output, _target)
    performance = accuracy(output.data, _target, topk=metrics,rnn=rnn)

    if classes is not None:
        acc_per_class, count_per_class = accuracy_per_class(output.data, _target, classes)
        return loss, performance, (acc_per_class, count_per_class)
    return loss, performance

def inference_personal(model1, model2, alpha, criterion, metrics, _input, _target):
    """Inference on the given model and get loss and accuracy."""
    # TODO: merge with inference
    output1 = model1(_input)
    output2 = model2(_input)
    output = alpha * output1 + (1-alpha) * output2
    loss = criterion(output, _target)
    performance = accuracy(output.data, _target, topk=metrics)
    return loss, performance

def do_validate(args, model, optimizer, criterion, metrics, data_loader, group,data_mode='validation',personal=False,model_personal=None,alpha=0.0,local=False):
    """Evaluate the model on the validation dataset."""
    model_mode = 'personal' if personal or local else 'global'
    tracker = define_val_tracker()
    if 'robust' in args.arch:
        tmp_noise = torch.clone(model.noise.data)
        # model.noise.data = torch.randn(tmp_noise.shape) * 0.1
        for _input, _target in data_loader:
            _input, _target = _load_data_batch(args, _input, _target)
            loss, performance = inference(model, criterion, metrics, _input, _target)
            grad = torch.autograd.grad(loss, model.noise)[0]
            model.noise.data.add_(grad,alpha=0.01)
            if torch.norm(model.noise.data) > 1:
                model.noise.data /= torch.norm(model.noise.data)
    # switch to evaluation mode
    model.eval()
    if personal:
        if model_personal is None:
            raise ValueError("model_personal should not be None for personalized mode for APFL!")
        model_personal.eval()
    for _input, _target in data_loader:
        # load data and check performance.
        _input, _target = _load_data_batch(args, _input, _target)
        # Skip batches with one sample because of BatchNorm issue in some models!
        if _input.size(0)==1:
            break
        with torch.no_grad():
            if personal:
                loss, performance = inference_personal(model_personal, model, alpha, criterion, metrics, _input, _target)
            else:
                loss, performance = inference(model, criterion, metrics, _input, _target)
            tracker = update_performancec_tracker(tracker, loss, performance, _input.size(0))
    if len(metrics) == 1:
        tracker['top5'].count = 1.0
        tracker['top5'].sum = 0.0
        tracker['top5'].avg = 0.0
    if data_mode == 'test' and model_mode =='global':
        # Only the server performs the test, do not need for aggregation
        performance = [evaluate_local_performance(tracker[x]) for x in ['top1', 'top5','losses']]
    else:
        performance = [evaluate_gloabl_performance(tracker[x], group) for x in ['top1', 'top5','losses']]
    logging_display_val(args,performance, mode=data_mode, personal=model_mode=='personal')
    if data_mode == 'test' and not personal:
        # remember best prec@1 and save checkpoint.
        args.cur_prec1 = performance[0]
        is_best = args.cur_prec1 > args.best_prec1
        if is_best:
            args.best_prec1 = performance[0]
            args.best_epoch += [args.epoch_]
        # logging and display val info.
        logging_display_test_summary(args, debug=args.debug)
        # save to the checkpoint. --> KAI deleted this code
    if 'robust' in args.arch:
        model.noise.data = tmp_noise
    return 
    
    
    
## AdamW
#class AdamW(Optimizer):

def step(self, closure=None, apply_lr=True, scale=1.0, **kargs):
    """Performs a single optimization step.
    Arguments: closure (callable, optional): A closure that reevaluates the model and returns the loss."""
    loss = None
    if closure is not None:
        loss = closure()
    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            state = self.state[p]
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            if not apply_lr:
                p.data.add_(grad, alpha=-scale)
            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']
            if group['weight_decay'] != 0 and not group['correct_wd']:
                grad = grad.add(p.data,alpha=group['weight_decay'])
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad,alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
            denom = exp_avg_sq.sqrt().add_(group['eps'])
            state['step'] += 1
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
            # apply gradients.
            if not group['correct_wd']:
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
            else:
                p.data.add_(torch.mul(p.data, group['weight_decay'],alpha=-step_size).addcdiv_(exp_avg, denom, value=1))
    return loss
    
    
## meter.py

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count == 0:
            self.avg=0
        else:
            self.avg = self.sum / self.count

def define_local_training_tracker():
    return define_trackers([
        'computing_time', 'global_time', 'data_time',
        'sync_time', 'load_time', 'losses', 'top1', 'top5','learning_rate'])

def define_trackers(names):
    return dict((name, AverageMeter()) for name in names) 
    
    
    
    
## init_config.py
# I have no idea what the graph stuff is in PyTorch

def init_config(args):
    # define the graph for the computation.
    cur_rank = dist.get_rank()
    args.graph = FCGraph(cur_rank, args.blocks, args.on_cuda, args.world)

    # TODO: Add parameter for this to enable it for other nodes
    if args.graph.rank != 0:
        args.debug=False
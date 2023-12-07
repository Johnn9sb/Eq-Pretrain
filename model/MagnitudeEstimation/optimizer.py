import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class Optimizer(object):
    def __init__(self, method, learning_rate, max_grad_norm,
                 lr_decay=1, start_decay_steps=None, decay_steps=None,
                 beta1=0.9, beta2=0.999,
                 decay_method=None,
                 warmup_steps=4000
                 ):
        self.last_ppl = None
        self.learning_rate = learning_rate               # v
        self.original_lr = learning_rate                 # v
        self.max_grad_norm = max_grad_norm
        self.method = method                             # adam
        self.lr_decay = lr_decay                         #
        self.start_decay_steps = start_decay_steps
        self.decay_steps = decay_steps                   # v
        self.start_decay = False                         # v
        self._step = 0
        self.betas = [beta1, beta2]                      # v
        self.decay_method = decay_method                 # v
        self.warmup_steps = warmup_steps                 # v

    def set_parameters(self, params):
        self.params = []
        self.sparse_params = []
        for k, p in params:
            if p.requires_grad:
                if self.method != 'sparseadam' or "embed" not in k:
                    self.params.append(p)
                else:
                    self.sparse_params.append(p)
       
        self.optimizer = optim.Adam(self.params, lr=self.learning_rate,
                                        betas=self.betas, eps=1e-9)
        
    def _set_rate(self, learning_rate):
        self.learning_rate = learning_rate
        if self.method != 'sparseadam':
            self.optimizer.param_groups[0]['lr'] = self.learning_rate
        else:
            for op in self.optimizer.optimizers:
                op.param_groups[0]['lr'] = self.learning_rate

    def step(self):
        """Update the model parameters based on current gradients.
        Optionally, will employ gradient modification or update learning
        rate.
        """
        self._step += 1

        # Decay method used in tensor2tensor.
        if self.decay_method == "noam":
            self._set_rate(
                self.original_lr *

                 min(self._step ** (-0.5),
                     self._step * self.warmup_steps**(-1.5)))

            # self._set_rate(self.original_lr *self.model_size ** (-0.5) *min(1.0, self._step / self.warmup_steps)*max(self._step, self.warmup_steps)**(-0.5))
        
        # Decay based on start_decay_steps every decay_steps
        else:
            if ((self.start_decay_steps is not None) and (
                     self._step >= self.start_decay_steps)):
                self.start_decay = True
            if self.start_decay:
                if ((self._step - self.start_decay_steps)
                   % self.decay_steps == 0):
                    self.learning_rate = self.learning_rate * self.lr_decay

        if self.method != 'sparseadam':
            self.optimizer.param_groups[0]['lr'] = self.learning_rate

        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
            
        self.optimizer.step()

def build_optim(opt, model, resume_training, device, checkpoint=None):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if resume_training:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            opt['method'], opt['lr'], opt['max_grad_norm'],
            beta1=opt['beta1'], beta2=opt['beta2'],
            decay_method=opt['decay_method'],
            warmup_steps=opt['warmup_steps'])

    optim.set_parameters(list(model.named_parameters()))

    if resume_training:
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        for state in optim.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    return optim
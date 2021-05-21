import time

import numpy as np
import torch
import tqdm

def fold_clip_in_place(x, min_val, max_val, momentum=None):
    with torch.no_grad():
        width = max_val - min_val
        x.add_(-min_val + width)
        x.remainder_(width * 2)
        x.add_(-width)
        if momentum is not None:
            momentum.mul_(torch.sign(x))
        x.abs_()
        x.add_(min_val)


class HMCSampler:
    def __init__(
            self,
            model,
            x,
            step_size=1,
            target_prob=0.6,
            step_adj_factor=None,
            adjusted=False,
            temp=None,
            noise_ratio=1,
            clip_grad=None,
            clip_data=None,
            rand_gen=None,
            use_pbar=False,
            pbar_args=None,
            pbar_msg_func=None,
            viz_func = None,
        ):

        if pbar_args is None:
            pbar_args = {}
        def msg_func(sampler, i):
            return f'S:{sampler.step_size.mean().item():.2g}, A:{sampler.step_prob.mean().item():.2f}, E:{sampler.energy.mean().item(): 6g}'
        if pbar_msg_func is None:
            pbar_msg_func = msg_func

        self._x_shape = x.shape
        self._device_id = x.device

        self._model = None
        self._x = None
        self._energy = None
        self._grad = None
        self._momentum = torch.empty(x.shape, device=self._device_id)

        self._x_ref = torch.empty(x.shape, device=self._device_id)
        self._energy_ref = torch.empty(x.shape[0], device=self._device_id)
        self._grad_ref = torch.empty(x.shape, device=self._device_id)
        self._momentum_ref = torch.empty(x.shape, device=self._device_id)

        self._step_size = None
        self._adjusted = None
        self._target_prob = None
        self._step_adj_factor = None
        self._temp = None
        self._noise_ratio = None
        self._clip_grad = None
        self._clip_data = None
        self._rand_gen = rand_gen

        self._use_pbar = None
        self._pbar_args = None
        self._pbar_msg_func = None
        self._viz_func = None

        self._energy_diff = None
        self._kinetic_diff = None
        self._log_step_prob = None
        self._step_prob = None
        self._accepted = None
        self._acceptance_rate = 0
        self._n_acceptance = 0

        if self._rand_gen is None:
            self._rand_gen = np.random.RandomState()  # pylint: disable=no-member

        if temp is None:
            temp = torch.ones(1, device=self._device_id)

        self.update(model=model,
                    x=x,
                    step_size=step_size,
                    target_prob=target_prob,
                    step_adj_factor=step_adj_factor,
                    adjusted=adjusted,
                    temp=temp,
                    noise_ratio=noise_ratio,
                    clip_grad=clip_grad,
                    clip_data=clip_data,
                    use_pbar=use_pbar,
                    pbar_args=pbar_args,
                    pbar_msg_func=pbar_msg_func,
                    viz_func=viz_func,
                    )

    def _unsqueeze(self, data):
        return data[(...,) + (None,) * (len(self._x_shape) - data.ndim)]

    def update(self,
               model=None,
               x=None,
               step_size=None,
               target_prob=None,
               step_adj_factor=None,
               adjusted=None,
               temp=None,
               noise_ratio=None,
               clip_grad=None,
               clip_data=None,
               use_pbar=None,
               pbar_args=None,
               pbar_msg_func=None,
               viz_func=None,
               ):

        if model is not None:
            self._model = model
        if clip_data is not None:
            self._clip_data = clip_data
        if x is not None:
            self._x = x.detach().requires_grad_(True)
            if self._clip_data is not None:
                fold_clip_in_place(self._x, self._clip_data[0], self._clip_data[1])
            self._grad, self._energy = self._get_grad_and_energy()
        if step_size is not None:
            if not isinstance(step_size, torch.Tensor):
                step_size = torch.tensor(step_size, dtype=torch.float, device=self._device_id)  # pylint: disable=not-callable
            self._step_size = self._unsqueeze(step_size)
        if target_prob is not None:
            self._target_prob = target_prob
        if step_adj_factor is not None:
            self._step_adj_factor = step_adj_factor
        if adjusted is not None:
            self._adjusted = adjusted
        if temp is not None:
            if not isinstance(temp, torch.Tensor):
                temp = torch.tensor(temp, dtype=torch.float, device=self._device_id)  # pylint: disable=not-callable
            self._temp = self._unsqueeze(temp)
        if noise_ratio is not None:
            self._noise_ratio = noise_ratio
        if clip_grad is not None:
            self._clip_grad = clip_grad
        if use_pbar is not None:
            self._use_pbar = use_pbar
        if pbar_args is not None:
            self._pbar_args = dict(
                leave=False,
                )
            self._pbar_args.update(pbar_args)
        if pbar_msg_func is not None:
            self._pbar_msg_func = pbar_msg_func
        if viz_func is not None:
            self._viz_func = viz_func

    def _get_grad_and_energy(self):
        with torch.enable_grad():
            energy = self._model(self._x)
            grad = torch.autograd.grad(energy.sum(), self._x)[0]

        if self._clip_grad is not None:
            grad.clamp_(-self._clip_grad, self._clip_grad)  # pylint: disable=invalid-unary-operand-type

        return grad, energy.detach()

    @property
    def step_size(self):
        return torch.squeeze(self._step_size)

    @property
    def x(self):
        return self._x.detach()

    @property
    def energy(self):
        return self._energy

    @property
    def energy_ref(self):
        return self._energy_ref

    @property
    def kinetic(self):
        return (self._momentum.view(self._x.shape[0], -1) ** 2).sum(dim=1) / 2 / self._noise_ratio ** 2

    @property
    def kinetic_ref(self):
        return (self._momentum_ref.view(self._x.shape[0], -1) ** 2).sum(dim=1) / 2 / self._noise_ratio ** 2

    @property
    def energy_diff(self):
        return self._energy_diff

    @property
    def kinetic_diff(self):
        return self._kinetic_diff

    @property
    def log_step_prob(self):
        return self._log_step_prob

    @property
    def step_prob(self):
        return self._step_prob

    @property
    def accepted(self):
        return self._accepted

    @property
    def acceptance_rate(self):
        return self._acceptance_rate

    def init_step(self, **kwargs):
        self.update(**kwargs)
        with torch.no_grad():
            self._momentum.normal_(0, self._noise_ratio)

            self._x_ref.copy_(self._x.detach())
            self._energy_ref.copy_(self._energy)
            self._grad_ref.copy_(self._grad)
            self._momentum_ref.copy_(self._momentum)

    def leap(self):
        with torch.no_grad():
            self._momentum.addcmul_(self._step_size / self._temp ** 0.5, self._grad, value=-0.5)
            self._x.addcmul_(self._step_size * self._temp ** 0.5, self._momentum)
            if self._clip_data is not None:
                fold_clip_in_place(self._x, self._clip_data[0], self._clip_data[1], self._momentum)
            self._grad, self._energy = self._get_grad_and_energy()
            self._momentum.addcmul_(self._step_size / self._temp ** 0.5, self._grad, value=-0.5)

        return self

    # def calc_acceptence(self, track_acceptance=None):
    #     return self

    def _calc_acceptence(self):
        self._energy_diff = (self._energy - self._energy_ref) / self._temp.view(-1)
        self._kinetic_diff = self.kinetic - self.kinetic_ref
        self._log_step_prob = -self._energy_diff - self._kinetic_diff  # pylint: disable=invalid-unary-operand-type
        self._step_prob = torch.exp(torch.clamp(self._log_step_prob, max=0)).cpu().numpy()
        self._step_prob[np.isnan(self._step_prob)] = 0
        self._accepted = self._rand_gen.rand(len(self._step_prob)) < self._step_prob

        if self._n_acceptance > 0:
            self._acceptance_rate = (self._acceptance_rate * self._n_acceptance + self._accepted.mean()) / (self._n_acceptance + 1)
        else:
            self._acceptance_rate = self._accepted.mean()
        self._n_acceptance += 1

        return self

    def reset_acceptance_rate(self):
        self._acceptance_rate = 0
        self._n_acceptance = 0
        return self

    def adjust(self):
        self._x.data[~self._accepted] = self._x_ref[~self._accepted]  # pylint: disable=invalid-unary-operand-type
        self._energy[~self._accepted] = self._energy_ref[~self._accepted]  # pylint: disable=invalid-unary-operand-type
        self._grad[~self._accepted] = self._grad_ref[~self._accepted]  # pylint: disable=invalid-unary-operand-type
        return self

    def adjust_step_size(self, step_adj_factor=None):
        if step_adj_factor is None:
            step_adj_factor = self._step_adj_factor
        if step_adj_factor is not None:
            if self._step_size.shape[0] == 1:
                self._step_size = self._step_size * step_adj_factor ** ((self._step_prob > self._target_prob) * 2 - 1).mean()
            else:
                self._step_size[self._step_prob > self._target_prob] *= step_adj_factor
                self._step_size[self._step_prob <= self._target_prob] /= step_adj_factor
        return self

    def step(self, n_leaps=1, **kwargs):
        self.init_step(**kwargs)
        for _ in range(n_leaps):
            self.leap()
        self._calc_acceptence()
        if self._adjusted:
            self.adjust()
        self.adjust_step_size()

        return self

    def run(self, n_steps, n_leaps_range=(1, 1), temp_list=None, **kwargs):
        self.update(**kwargs)

        pbar = range(n_steps)
        if self._use_pbar:
            pbar = tqdm.tqdm(pbar, total=n_steps, **self._pbar_args)

        last_msg = 0
        for i in pbar:
            if temp_list is not None:
                self.update(temp=temp_list[i])
            n_leaps = self._rand_gen.randint(n_leaps_range[0], n_leaps_range[1] + 1)
            self.step(n_leaps)
            if self._use_pbar and (time.time() - last_msg) > 1:
                pbar.set_description_str(self._pbar_msg_func(self , i))
                last_msg = time.time()
            if self._viz_func is not None:
                self._viz_func(self, i)

        if self._viz_func is not None:
            self._viz_func(self, -1)
        if self._use_pbar:
            pbar.close()
        return self

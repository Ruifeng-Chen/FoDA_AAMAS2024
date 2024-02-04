from copy import deepcopy
from typing import Any, Optional, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import StepLR

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.types import DistLogProbBatchProtocol, RolloutBatchProtocol
from tianshou.exploration import BaseNoise
from tianshou.policy import DDPGPolicy


class SACPolicy(DDPGPolicy):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient. Default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param BaseNoise exploration_noise: add a noise to action for exploration.
        Default to None. This is useful when solving hard-exploration problem.
    :param bool deterministic_eval: whether to use deterministic action (mean
        of Gaussian policy) instead of stochastic action sampled by the policy.
        Default to True.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        criticv: torch.nn.Module,
        criticv_optim: torch.optim.Optimizer,
        lfiw_critic: torch.nn.Module,
        lfiw_critic_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        exploration_noise: Optional[BaseNoise] = None,
        deterministic_eval: bool = True,
        lfiw: float = 0.5,
        foresight_eta: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            None,
            None,
            None,
            None,
            tau,
            gamma,
            exploration_noise,
            reward_normalization,
            estimation_step,
            **kwargs,
        )
        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self.criticv, self.criticv_old = criticv, deepcopy(criticv)
        self.criticv_old.eval()
        self.criticv_optim = criticv_optim
        self.lfiw_critic = lfiw_critic
        self.lfiw_critic_optim = lfiw_critic_optim
        self.lfiw = lfiw
        self.foresight_eta = foresight_eta
        if self.foresight_eta:
            self.critic1_sche = StepLR(self.critic1_optim, step_size=10000, gamma=0.99)
            self.critic2_sche = StepLR(self.critic2_optim, step_size=10000, gamma=0.99)
            self.actor_sche = StepLR(self.actor_optim, step_size=10000, gamma=0.99)
        else:
            self.critic1_sche = None
            self.critic2_sche = None
            self.actor_sche = None
            
        self._is_auto_alpha = False
        self._alpha: Union[float, torch.Tensor]
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            assert alpha[1].shape == torch.Size([1])
            assert alpha[1].requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self._deterministic_eval = deterministic_eval
        self.__eps = np.finfo(np.float32).eps.item()

    def train(self, mode: bool = True) -> "SACPolicy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        self.soft_update(self.critic1_old, self.critic1, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)

    # TODO: violates Liskov substitution principle
    def forward(  # type: ignore
        self,
        batch: RolloutBatchProtocol,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> DistLogProbBatchProtocol:
        obs = batch[input]
        logits, hidden = self.actor(obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)
        dist = Independent(Normal(*logits), 1)
        if self._deterministic_eval and not self.training:
            act = logits[0]
        else:
            act = dist.rsample()
        log_prob = dist.log_prob(act).unsqueeze(-1)
        # apply correction for Tanh squashing when computing logprob from Gaussian
        # You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
        # in appendix C to get some understanding of this equation.
        squashed_action = torch.tanh(act)
        log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) + self.__eps).sum(
            -1,
            keepdim=True,
        )
        result = Batch(
            logits=logits,
            act=squashed_action,
            state=hidden,
            dist=dist,
            log_prob=log_prob,
        )
        return cast(DistLogProbBatchProtocol, result)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs: s_{t+n}
        obs_next_result = self(batch, input="obs_next")
        act_ = obs_next_result.act
        return (
            torch.min(
                self.critic1_old(batch.obs_next, act_),
                self.critic2_old(batch.obs_next, act_),
            )
            - self._alpha * obs_next_result.log_prob
        )

    def lfiw_learn(self, batch, fast_batch):
        
        slow_obs_batch = batch.obs
        obs_result = self(batch)
        slow_action_batch = obs_result.act

        fast_obs_batch = fast_batch.obs
        obs_result = self(fast_batch)
        fast_action_batch = obs_result.act

        # slow_samples = torch.cat([slow_obs_batch, slow_action_batch], dim = 1)
        # fast_samples = torch.cat([fast_obs_batch, fast_action_batch], dim = 1)
    
        # zeros = torch.zeros(slow_samples.size[0]).to(util.device)
        # ones = torch.ones(fast_samples.size[0]).to(util.device)

        slow_preds = self.lfiw_critic(slow_obs_batch, slow_action_batch)
        fast_preds = self.lfiw_critic(fast_obs_batch, fast_action_batch)

        zeros = torch.zeros_like(slow_preds).to(slow_preds)
        ones = torch.ones_like(fast_preds).to(slow_preds)

        loss = F.binary_cross_entropy(torch.sigmoid(slow_preds), zeros) + \
                F.binary_cross_entropy(torch.sigmoid(fast_preds), ones)
        
        self.lfiw_critic_optim.zero_grad()
        loss.backward()
        self.lfiw_critic_optim.step()    
        
        return {"lfiw/slow": slow_preds.mean().detach().cpu().numpy().item(), "lfiw/fast": fast_preds.mean().detach().cpu().numpy().item()}

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> dict[str, float]:
        # critic 1&2
        acc_reward_batch = torch.from_numpy(batch["acc_return"]).to(self.criticv.device)
        next_obs_batch = batch["obs_next"]
        initial_obs_batch = batch["initial_obs"]
        act = batch["act"]

        with torch.no_grad():
            advantage_batch = acc_reward_batch  + self.criticv(next_obs_batch).flatten() - self.criticv(initial_obs_batch).flatten()
            advantage_batch = torch.clip(advantage_batch, 0, None)
            # advantage_batch = torch.sigmoid(advantage_batch)
            advantage_batch = advantage_batch / (torch.sum(advantage_batch) + 1e-5)
            advantage_batch = torch.sqrt(advantage_batch)

            raw_weight = self.lfiw_critic(batch.obs, act)
            weight = torch.sigmoid(raw_weight * self.lfiw)    
            weight = (weight / torch.mean(weight)).flatten()
            weight = torch.sqrt(weight)
        
        #compute q loss and backward
        td1, critic1_loss, td1_mean, foda1_mean = self._mse_optimizer(batch, self.critic1, self.critic1_optim, lfiw_weight=weight, advantage_batch=advantage_batch, critic_sche=self.critic1_sche, foresight_eta=self.foresight_eta, lfiw = self.lfiw)
        td2, critic2_loss, td2_mean, foda2_mean = self._mse_optimizer(batch, self.critic2, self.critic2_optim, lfiw_weight=weight, advantage_batch=advantage_batch, critic_sche=self.critic2_sche, foresight_eta=self.foresight_eta, lfiw = self.lfiw)
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        obs_batch =  torch.from_numpy(batch["obs"]).to(self.criticv.device)
        reward_batch = torch.from_numpy(batch["rew"]).to(self.criticv.device)
        curr_state_value = self.criticv(obs_batch)
        next_state_value = self.criticv_old(next_obs_batch)
        with torch.no_grad():
            target_v = reward_batch.unsqueeze_(dim = -1) + self._gamma * next_state_value

        v_loss = F.mse_loss(curr_state_value.float(), target_v.float())
        self.criticv_optim.zero_grad()
        v_loss.backward()
        self.criticv_optim.zero_grad()
        self.soft_update(self.criticv_old, self.criticv, self.tau)

        # actor
        obs_result = self(batch)
        act = obs_result.act
        current_q1a = self.critic1(batch.obs, act).flatten()
        current_q2a = self.critic2(batch.obs, act).flatten()
        actor_loss = (
            self._alpha * obs_result.log_prob.flatten() - torch.min(current_q1a, current_q2a)
        ).mean()


        self.actor_optim.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)

        self.actor_optim.step()
        if self.actor_sche is not None:
            self.actor_sche.step()

        if self._is_auto_alpha:
            log_prob = obs_result.log_prob.detach() + self._target_entropy
            # please take a look at issue #258 if you'd like to change this line
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self.sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "td1_mean": td1_mean.item(),
            "foda1_mean": foda1_mean.item(),
            "td2_mean": td2_mean.item(),
            "foda2_mean": foda2_mean.item()
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore

        return result

import warnings
from copy import deepcopy
from typing import Any, Literal, Optional, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import BatchWithReturnsProtocol, RolloutBatchProtocol
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.policy import BasePolicy


class DDPGPolicy(BasePolicy):
    """Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic: the critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic_optim: the optimizer for critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param BaseNoise exploration_noise: the exploration noise,
        add to the action. Default to ``GaussianNoise(sigma=0.1)``.
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        Default to False.
    :param int estimation_step: the number of steps to look ahead. Default to 1.
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
        actor: Optional[torch.nn.Module],
        actor_optim: Optional[torch.optim.Optimizer],
        critic: Optional[torch.nn.Module],
        critic_optim: Optional[torch.optim.Optimizer],
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        reward_normalization: bool = False,
        estimation_step: int = 1,
        action_scaling: bool = True,
        action_bound_method: Optional[Literal["clip", "tanh"]] = "clip",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs,
        )
        assert action_bound_method != "tanh", (
            "tanh mapping is not supported"
            "in policies where action is used as input of critic , because"
            "raw action in range (-inf, inf) will cause instability in training"
        )
        try:
            if (
                actor is not None
                and action_scaling
                and not np.isclose(actor.max_action, 1.0)  # type: ignore
            ):
                import warnings

                warnings.warn(
                    "action_scaling and action_bound_method are only intended to deal"
                    "with unbounded model action space, but find actor model bound"
                    f"action space with max_action={actor.max_action}."
                    "Consider using unbounded=True option of the actor model,"
                    "or set action_scaling to False and action_bound_method to None.",
                )
        except Exception:
            pass
        if actor is not None and actor_optim is not None:
            self.actor: torch.nn.Module = actor
            self.actor_old = deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim: torch.optim.Optimizer = actor_optim
        if critic is not None and critic_optim is not None:
            self.critic: torch.nn.Module = critic
            self.critic_old = deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim: torch.optim.Optimizer = critic_optim
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self.tau = tau
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
        self._gamma = gamma
        self._noise = exploration_noise
        # it is only a little difference to use GaussianNoise
        # self.noise = OUNoise()
        self._rew_norm = reward_normalization
        self._n_step = estimation_step

    def set_exp_noise(self, noise: Optional[BaseNoise]) -> None:
        """Set the exploration noise."""
        self._noise = noise

    def train(self, mode: bool = True) -> "DDPGPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        self.soft_update(self.actor_old, self.actor, self.tau)
        self.soft_update(self.critic_old, self.critic, self.tau)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        return self.critic_old(batch.obs_next, self(batch, model="actor_old", input="obs_next").act)

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> Union[RolloutBatchProtocol, BatchWithReturnsProtocol]:
        return self.compute_nstep_return(
            batch,
            buffer,
            indices,
            self._target_q,
            self._gamma,
            self._n_step,
            self._rew_norm,
        )

    def forward(
        self,
        batch: RolloutBatchProtocol,
        state: Optional[Union[dict, BatchProtocol, np.ndarray]] = None,
        model: Literal["actor", "actor_old"] = "actor",
        input: str = "obs",
        **kwargs: Any,
    ) -> BatchProtocol:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        actions, hidden = model(obs, state=state, info=batch.info)
        return Batch(act=actions, state=hidden)

    @staticmethod
    def _mse_optimizer(
        batch: RolloutBatchProtocol,
        critic: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lfiw_weight = None,
        advantage_batch = None,
        critic_sche = None,
        foresight_eta = 0.3,
        lfiw = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        foda = 1 + foresight_eta * advantage_batch.pow(2)
        td = current_q - target_q

        # critic_loss = F.mse_loss(current_q1, target_q)
        if lfiw > 0: 
            critic_loss = ((lfiw_weight).pow(2) * foda * td.pow(2) * weight).mean()
        else: 
            critic_loss = (td.pow(2) * weight).mean()

        optimizer.zero_grad()
        critic_loss.backward()
        
        if foresight_eta:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        optimizer.step()
        if critic_sche is not None: 
            critic_sche.step()
        return td, critic_loss, td.mean(), foda.mean()

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> dict[str, float]:
        # critic
        td, critic_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        batch.weight = td  # prio-buffer
        # actor
        actor_loss = -self.critic(batch.obs, self(batch).act).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {"loss/actor": actor_loss.item(), "loss/critic": critic_loss.item()}

    def exploration_noise(
        self,
        act: Union[np.ndarray, BatchProtocol],
        batch: RolloutBatchProtocol,
    ) -> Union[np.ndarray, BatchProtocol]:
        if self._noise is None:
            return act
        if isinstance(act, np.ndarray):
            return act + self._noise(act.shape)
        warnings.warn("Cannot add exploration noise to non-numpy_array action.")
        return act

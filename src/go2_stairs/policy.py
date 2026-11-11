"""Affine Beta actor: PPO scores the same bounded actions the robot executes."""
import math
import torch
from torch.distributions import Beta
from rsl_rl.modules import ActorCritic


class BoundedBetaActorCritic(ActorCritic):
    def __init__(self, obs, obs_groups, num_actions, **kwargs):
        if kwargs.get('state_dependent_std') or kwargs.get('noise_std_type', 'scalar') != 'scalar':
            raise ValueError('Gaussian noise options do not apply to the Beta actor')
        super().__init__(obs, obs_groups, num_actions*2, **kwargs)
        del self.std  # no unused/unbounded Gaussian exploration parameter
        self.num_actions = num_actions
        # Independent alpha/beta heads start near Beta(2,2), not a collapsed old policy.
        with torch.no_grad():
            self.actor[-1].weight.mul_(.01)
            self.actor[-1].bias.fill_(math.log(math.expm1(.99)))

    def _update_distribution(self, obs):
        logits = self.actor(obs)
        if not bool(torch.isfinite(logits).all()) or bool((logits.abs()>1000).any()):
            raise ValueError('nonfinite or runaway Beta concentration logits')
        alpha, beta = (torch.nn.functional.softplus(logits)+1.01).chunk(2, dim=-1)
        self.distribution = Beta(alpha, beta, validate_args=False)

    @property
    def action_mean(self):
        return 2*self.distribution.mean-1

    @property
    def action_std(self):
        return 2*self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(-1) + self.num_actions*math.log(2.)

    def act(self, obs, **kwargs):
        self._update_distribution(self.actor_obs_normalizer(self.get_actor_obs(obs)))
        unit = self.distribution.sample()
        # Keep float32 affine roundoff away from the exact open-support endpoints.
        eps = torch.finfo(unit.dtype).eps
        return 2*unit.clamp(eps, 1-eps)-1

    def act_inference(self, obs):
        self._update_distribution(self.actor_obs_normalizer(self.get_actor_obs(obs)))
        return self.action_mean

    def get_actions_log_prob(self, actions):
        if not bool(torch.isfinite(actions).all()) or bool((actions.abs()>=1).any()):
            raise ValueError('Beta actions must be finite and strictly inside (-1,1)')
        return self.distribution.log_prob((actions+1)/2).sum(-1) - self.num_actions*math.log(2.)

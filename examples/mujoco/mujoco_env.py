import warnings

import gymnasium as gym
import numpy as np
from tianshou.env import ShmemVectorEnv, VectorEnvNormObs
from gymnasium import core
# from gym import spaces
from gymnasium import spaces 
from dm_env import specs
try:

    import envpool
except ImportError:
    envpool = None
DMC_ENVS = [
    "acrobot-swingup", "acrobot-swingup_sparse", "ball_in_cup-catch", "cartpole-balance", 
    "cartpole-balance_sparse", "cartpole-swingup", "cartpole-swingup_sparse", "cheetah-run", 
    "finger-spin", "finger-turn_easy", "finger-turn_hard", "fish-upright", "fish-swim", 
    "hopper-stand", "hopper-hop", "humanoid-stand", "humanoid-walk", "humanoid-run", 
    "manipulator-bring_ball", "pendulum-swingup", "point_mass-easy", "reacher-easy", 
    "reacher-hard", "swimmer-swimmer6", "swimmer-swimmer15", "walker-stand", "walker-walk", "walker-run"
]

GYM_SINGLE_ENVS = [
    # Mujoco Envs, all continuous action space
    'Ant-v2', 'Ant-v3', 'Ant-v4',
    'HalfCheetah-v2', 'HalfCheetah-v3','HalfCheetah-v4',
    'Hopper-v2', 'Hopper-v3',  'Hopper-v4',
    'Humanoid-v2', 'Humanoid-v3', 'Humanoid-v4',
    'InvertedDoublePendulum-v2', 'InvertedDoublePendulum-v4',
    'InvertedPendulum-v2', 'InvertedPendulum-v3',  'InvertedPendulum-v4',
    'Swimmer-v2', 'Swimmer-v3', 'Swimmer-v4',
    'Walker2d-v2', 'Walker2d-v3', 'Walker2d-v4',
    'Reacher-v4',
    'Pusher-v4',
    # Classic Control Envs
        ## discrete action space
        'Acrobot-v1', 'CartPole-v1', 'MountainCar-v0',
        ## continuous action space
        'Pendulum-v1', 'MountainCarContinuous-v0', 
    #Box2d Envs
        ## discrete action space
        'LunarLander-v2',
        ## continuous action space
        'BipedalWalker-v3', 'CarRacing-v2', 
    ]


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.box.Box(low, high, dtype=dtype)


class DMCWrapper(core.Env):
    def __init__(
        self,
        domain_name,
        task_name,
        task_kwargs=None,
        visualize_reward=True,
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        channels_first=True
    ):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        # create task
        from dm_control import suite
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.box.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.box.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values(),
                np.float64
            )
            
        self._state_space = _spec_to_box(
            self._env.observation_spec().values(),
            np.float64
        )
        
        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs.get('random', 1))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def reward_range(self):
        return 0, self._frame_skip

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        return obs, reward, done, False, extra

    def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs, {}

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )

def make_mujoco_env(task, seed, training_num, test_num, obs_norm):
    """Wrapper function for Mujoco env.

    If EnvPool is installed, it will automatically switch to EnvPool's Mujoco env.

    :return: a tuple of (single env, training envs, test envs).
    """
    if envpool is not None:
        train_envs = env = envpool.make_gymnasium(task, num_envs=training_num, seed=seed)
        test_envs = envpool.make_gymnasium(task, num_envs=test_num, seed=seed)
    else:
        warnings.warn(
            "Recommend using envpool (pip install envpool) "
            "to run Mujoco environments more efficiently.",
        )
        if task in GYM_SINGLE_ENVS:
            env = gym.make(task)
            train_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(training_num)])
            test_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(test_num)])
            env.reset(seed = seed)
            train_envs.action_space[0].seed(seed)
            test_envs.action_space[0].seed(seed)

        elif task in DMC_ENVS:
            domain_name, task_name = task.split("-")
            env = DMCWrapper(domain_name, task_name, task_kwargs = {"random": seed})
            train_envs = ShmemVectorEnv([lambda: DMCWrapper(domain_name, task_name, task_kwargs = {"random": seed}) for _ in range(training_num)])
            test_envs = ShmemVectorEnv([lambda: DMCWrapper(domain_name, task_name, task_kwargs = {"random": seed}) for _ in range(test_num)])
    if obs_norm:
        # obs norm wrapper
        train_envs = VectorEnvNormObs(train_envs)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
    return env, train_envs, test_envs

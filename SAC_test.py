import gym
import numpy as np
import os

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import HER, SAC, DDPG
from common import get_args, experiment_setup
from envs import make_env
import time
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.bit_flipping_env import BitFlippingEnv


class EvaluateCallback(BaseCallback):
    def __init__(self, args, agent, eval_env , rank=1,
                 experiment_name: str = None):
        super().__init__()
        self._agent = agent
        self._eval_env = eval_env
        self.epoch = 0
        self.cycle = 0
        experiment_name = args.env
        self._log_fname = f"{experiment_name}-{rank}-performance.csv"
        with open(self._log_fname, "w") as file:
            file.write(f"Epoch,Cycle,successRate\n")

    def _on_step(self) -> bool:
        if self.num_timesteps % 2500 == 0:
            self._log_full_goal_space_performance()
            self.cycle += 1
            if self.cycle == 20:
                self.cycle = 0
                self.epoch += 1
        return True

    def _log_full_goal_space_performance(self):
        reached, not_reached = self.evaluate(agent=self._agent, env=self._eval_env)
        pct = reached / (reached + not_reached)
        log = f"{self.epoch}/20,{self.cycle}/20,{round(pct, 6)}\n"
        with open(self._log_fname, "a") as file:
            file.write(log)

    def evaluate(self, agent, env):
        reached = 0
        not_reached = 0
        for i in range(100):
            is_success = False
            obs = env.reset()
            for timestep in range(50):
                action, _states = agent.predict(obs)
                obs, rewards, dones, info = env.step(action)
                if info['is_success']:
                    is_success= True
            if is_success:
                reached += 1
            else:
                not_reached += 1

        return reached, not_reached









if __name__ == '__main__':
    args = get_args()
    env = gym.make(args.env)

    # args.logger.add_item('Epoch')
    # args.logger.add_item('Cycle')
    # args.logger.add_item('Episodes@green')
    # args.logger.add_item('Timesteps')
    # args.logger.add_item('TimeCost(sec)')

    # args.logger.summary_setup()
    agent = HER('MlpPolicy', env, SAC, n_sampled_goal=4,
                goal_selection_strategy='future',
                verbose=1, buffer_size=int(1e6),
                gamma=0.98, batch_size=256)
    eval_callback = EvaluateCallback(args, agent, env)
    agent.learn(total_timesteps=1000000, callback=eval_callback)
    agent.save("SAC_FetchPush")

    obs = env.reset()



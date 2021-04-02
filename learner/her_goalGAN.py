import copy
import numpy as np
from envs import make_env
from envs.utils import goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double, c_int
from envs.distance_graph import DistanceGraph
from learner.GenerativeGoalLearning import train_goalGAN, initialize_GAN
from typing import Tuple, Mapping, List
from itertools import cycle
from algorithm import create_agent

GoalHashable = Tuple[float]
NUM = 300
SCALE = 4
class TrajectoryPool:
    def __init__(self, args, pool_length):
        self.args = args
        self.length = pool_length

        self.pool = []
        self.pool_init_state = []
        self.counter = 0

    def insert(self, trajectory, init_state):
        if self.counter < self.length:
            self.pool.append(trajectory.copy())
            self.pool_init_state.append(init_state.copy())
        else:
            self.pool[self.counter % self.length] = trajectory.copy()
            self.pool_init_state[self.counter % self.length] = init_state.copy()
        self.counter += 1

    def pad(self):
        if self.counter >= self.length:
            return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
        pool = copy.deepcopy(self.pool)
        pool_init_state = copy.deepcopy(self.pool_init_state)
        while len(pool) < self.length:
            pool += copy.deepcopy(self.pool)
            pool_init_state += copy.deepcopy(self.pool_init_state)
        return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])


class MatchSampler:
    def __init__(self, args, env, use_random_starting_pos=False):
        self.args = args
        self.env = env
        self.delta = self.env.distance_threshold
        self.init_state = self.env.reset()['observation'].copy()
        self.possible_goals = None
        self.successes_per_goal: Mapping[GoalHashable, List[bool]] = dict()
        self.use_random_starting_pos = use_random_starting_pos
        self.start_pos = self.env.get_obs()['achieved_goal'].copy()
        self.agent_pos = None
        self.step_num = 0

    def add_noise(self, pre_goal, noise_std=None):
        goal = pre_goal.copy()
        dim = 2 if self.args.env[:5] == 'fetch' else self.dim
        if noise_std is None: noise_std = self.delta
        goal[:dim] += np.random.normal(0, noise_std, size=dim)
        return goal.copy()

    def sample(self):
        return next(self.possible_goals).copy()

    def new_initial_pos(self):
        if not self.use_random_starting_pos:
            return self.start_pos

    def reset(self):
        obs = self.env.reset()
        self.start_pos = obs['achieved_goal'].copy()
        self.env.goal = self.sample().copy()
        self.step_num = 0

    def set_possible_goals(self, goals, entire_space=False) -> None:
        if goals is None and entire_space:
            self.possible_goals = None
            self.successes_per_goal = dict()
            return
        self.possible_goals = cycle(np.random.permutation(goals))
        self.successes_per_goal = {tuple(g): [] for g in goals}

    def get_successes_of_goals(self) -> Mapping[GoalHashable, List[bool]]:
        return dict(self.successes_per_goal)

    def step(self):
        self.step_num += 1


class HERGoalGANLearner:
    def __init__(self, args):
        self.args = args
        self.env = make_env(args)
        self.env_test = make_env(args)
        self.env_List = []
        for i in range(args.episodes):
            self.env_List.append(make_env(args))
        self.agent = create_agent(args)
        self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)
        self.stop_hgg_threshold = self.args.stop_hgg_threshold
        self.stop = False
        self.learn_calls = 0

    def learn(self, args, env, env_test, agent, buffer, write_goals=0):
        # Actual learning cycle takes place here!
        initial_goals = []
        desired_goals = []
        episodes = args.episodes // 5
        # get initial position and goal from environment for each epsiode
        for i in range(episodes):
            obs = self.env_List[i].reset()
            goal_a = obs['achieved_goal'].copy()
            goal_d = obs['desired_goal'].copy()
            initial_goals.append(goal_a.copy())
            desired_goals.append(goal_d.copy())
        goal_list = []
        achieved_trajectories = []
        achieved_init_states = []
        explore_goals = []
        test_goals = []
        inside = []
        for i in range(episodes):
            obs = self.env_List[i].get_obs()
            init_state = obs['observation'].copy()
            sampler = MatchSampler(args, self.env_List[i])
            loop = train_goalGAN(agent, initialize_GAN(env=self.env_List[i]), sampler, 5, True)
            next(loop)
            if not self.stop:
                explore_goal = sampler.sample()
            else:
                explore_goal = desired_goals[i]

            # store goals in explore_goals list to check whether goals are within goal space later
            explore_goals.append(explore_goal)
            test_goals.append(self.env.generate_goal())

            # Perform HER training by interacting with the environment
            self.env_List[i].goal = explore_goal.copy()
            if write_goals != 0 and len(goal_list) < write_goals:
                goal_list.append(explore_goal.copy())
            current = None
            trajectory = None
            for iters in range(NUM):
                if iters < 20:
                    obs = self.env_List[i].get_obs()
                    current = Trajectory(obs)
                    trajectory = [obs['achieved_goal'].copy()]
                has_success = False
                for timestep in range(args.timesteps // SCALE):
                    # get action from the ddpg policy
                    action = agent.step(obs, explore=True)
                    # feed action to environment, get observation and reward
                    obs, reward, done, info = self.env_List[i].step(action)
                    is_success = reward == 0
                    if iters < 20:
                        trajectory.append(obs['achieved_goal'].copy())
                        current.store_step(action, obs, reward, done)
                    if is_success and not has_success:
                        has_success = True
                        if len(sampler.successes_per_goal) > 0:
                            sampler.successes_per_goal[tuple(self.env_List[i].goal)].append(is_success)
                    if timestep == args.timesteps // SCALE -1:
                        if len(sampler.successes_per_goal) > 0:
                            sampler.successes_per_goal[tuple(self.env_List[i].goal)].append(is_success)

                next(loop)
                sampler.reset()
                if iters < 20:
                    achieved_trajectories.append(np.array(trajectory))
                    achieved_init_states.append(init_state)
                    # Trajectory is stored in replay buffer, replay buffer can be normal or EBP
                    buffer.store_trajectory(current)
                    agent.normalizer_update(buffer.sample_batch())



            if buffer.steps_counter >= args.warmup:
                for _ in range(args.train_batches):
                    # train with Hindsight Goals (HER step)
                    info = agent.train(buffer.sample_batch())
                    args.logger.add_dict(info)
                # update target network
                agent.target_update()

        selection_trajectory_idx = {}
        for i in range(episodes):
            # only add trajectories with movement to the trajectory pool --> use default (L2) distance measure!
            if goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1]) > 0.01:
                selection_trajectory_idx[i] = True
        for idx in selection_trajectory_idx.keys():
            self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())

        # unless in first call: Check which of the explore goals are inside the target goal space target goal space
        # is represented by a sample of test_goals directly generated from the environment an explore goal is
        # considered inside the target goal space, if it is closer than the distance_threshold to one of the test
        # goals (i.e. would yield a non-negative reward if that test goal was to be achieved)
        if self.learn_calls > 0:
            assert len(explore_goals) == len(test_goals)
            for ex in explore_goals:
                is_inside = 0
                for te in test_goals:
                    # TODO: check: originally with self.sampler.get_graph_goal_distance, now trying with goal_distance (L2)
                    if goal_distance(ex, te) <= self.env.env.env.distance_threshold:
                        is_inside = 1
                inside.append(is_inside)
            assert len(inside) == len(test_goals)
            inside_sum = 0
            for i in inside:
                inside_sum += i

            # If more than stop_hgg_threshold (e.g. 0.9) of the explore goals are inside the target goal space, stop HGG
            # and continue with normal HER.
            # By default, stop_hgg_threshold is disabled (set to a value > 1)
            average_inside = inside_sum / len(inside)
            self.args.logger.info("Average inside: {}".format(average_inside))
            if average_inside > self.stop_hgg_threshold:
                self.stop = True
                self.args.logger.info("Continue with normal HER")

        self.learn_calls += 1

        return goal_list if len(goal_list) > 0 else None

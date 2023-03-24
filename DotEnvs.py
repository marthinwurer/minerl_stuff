import gym
import numpy as np

action_to_vector = {
    0: np.array([0, -1]),  # Up
    1: np.array([1, 0]),   # Right
    2: np.array([0, 1]),   # Down
    3: np.array([-1, 0])   # Left
}

class DotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.width = 64
        self.height = 64
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self.dot_position = np.array([0, 0], dtype='int32')
        self.dot_color = np.array([255, 255, 255])
        self.goal_position = np.array([self.width/2, self.height/2], dtype='int32')
        self.viewer = None
        self.reward_range = (-float('inf'), float('inf'))
        self.max_episode_steps = 200
        self.current_step = 0

    def step(self, action):
        # Move the dot according to the action
        if action == 0:  # Up
            self.dot_position[1] = max(self.dot_position[1] - 1, 0)
        elif action == 1:  # Right
            self.dot_position[0] = min(self.dot_position[0] + 1, self.width - 2)
        elif action == 2:  # Down
            self.dot_position[1] = min(self.dot_position[1] + 1, self.height - 2)
        elif action == 3:  # Left
            self.dot_position[0] = max(self.dot_position[0] - 1, 0)

        # Compute the reward based on the new position of the dot
        distance = np.linalg.norm(self.dot_position - self.goal_position)
        reward = 1 if distance < np.linalg.norm(self.dot_position - action_to_vector[action] - self.goal_position) else -1

        # Update the observation
        observation = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        observation[self.dot_position[1]:self.dot_position[1]+2, self.dot_position[0]:self.dot_position[0]+2] = self.dot_color

        self.current_step += 1
        done = self.current_step >= self.max_episode_steps or all(self.dot_position == self.goal_position)
        info = {}

        return observation, reward, done, info
    
    def get_obs(self):
        observation = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        observation[self.dot_position[1]:self.dot_position[1]+2, self.dot_position[0]:self.dot_position[0]+2] = self.dot_color
        return observation

    def reset(self):
        self.dot_position = np.array([np.random.randint(0, self.width - 2), np.random.randint(0, self.height - 2)])
        observation = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        observation[self.dot_position[1]:self.dot_position[1]+2, self.dot_position[0]:self.dot_position[0]+2] = self.dot_color
        self.current_step = 0
        return observation

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(self.reset())


class ContinuousDotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.width = 64
        self.height = 64
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.dot_position = np.array([0, 0], dtype='int32')
        self.dot_color = np.array([255, 255, 255])
        self.goal_position = np.array([self.width/2, self.height/2], dtype='int32')
        self.viewer = None
        self.reward_range = (-float('inf'), float('inf'))
        self.max_episode_steps = 200
        self.current_step = 0

    def step(self, action):
        # Move the dot according to the action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.dot_position[0] = np.clip(self.dot_position[0] + action[0], 0, self.width - 2)
        self.dot_position[1] = np.clip(self.dot_position[1] + action[1], 0, self.height - 2)

        # Compute the reward based on the new position of the dot
        distance = np.linalg.norm(self.dot_position - self.goal_position)
        reward = 1 if distance < np.linalg.norm(self.dot_position - action - self.goal_position) else -1

        # Update the observation
        observation = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        observation[self.dot_position[1]:self.dot_position[1]+2, self.dot_position[0]:self.dot_position[0]+2] = self.dot_color

        self.current_step += 1
        done = self.current_step >= self.max_episode_steps or all(self.dot_position == self.goal_position)
        info = {}

        return observation, reward, done, info
    
    def get_obs(self):
        observation = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        observation[self.dot_position[1]:self.dot_position[1]+2, self.dot_position[0]:self.dot_position[0]+2] = self.dot_color
        return observation

    def reset(self):
        self.dot_position = np.array([np.random.randint(0, self.width - 2), np.random.randint(0, self.height - 2)])
        observation = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        observation[self.dot_position[1]:self.dot_position[1]+2, self.dot_position[0]:self.dot_position[0]+2] = self.dot_color
        self.current_step = 0
        return observation

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(self.reset())

        
class CollectDotsEnv(gym.Env):
    """
    An environment with sparse rewards. 
    Sixteen green dots are scattered randomly around the screen,
    and the agent gets +1 reward for each one they touch.
    """
    metadata = {'render.modes': ['human']}
    colors = [
        np.array([255, 0, 0]),
        np.array([0, 255, 0]),
        np.array([0, 0, 255]),
    ]

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self.dot_position = np.array([0, 0], dtype='int32')
        self.dot_color = np.array([255, 255, 255])
        self.viewer = None
        self.reward_range = (-float('inf'), float('inf'))
        self.current_step = 0
        self.colored_dots = []

    def step(self, action):
        # Move the dot according to the action
        if action == 0:  # Up
            self.dot_position[1] = max(self.dot_position[1] - 1, 0)
        elif action == 1:  # Right
            self.dot_position[0] = min(self.dot_position[0] + 1, 64 - 2)
        elif action == 2:  # Down
            self.dot_position[1] = min(self.dot_position[1] + 1, 64 - 2)
        elif action == 3:  # Left
            self.dot_position[0] = max(self.dot_position[0] - 1, 0)

        # Compute the reward based on the new position of the dot
        reward = 0
        for i, d in enumerate(self.colored_dots):
            if sum(np.abs(self.dot_position - d[0])) <= 1:
                # we hit the dot!
                reward += 1
                self.colored_dots.pop(i)
                break
            
        self.current_step += 1
        done = len(self.colored_dots) == 0 or self.current_step >= 1000
        info = {}

        # Update the observation
        observation = self.get_obs()
        
        return observation, reward, done, info
    
    def render_dot(self, obs, position, color):
        obs[position[1]:position[1]+2, position[0]:position[0]+2] = color
    
    def get_obs(self):
        observation = np.zeros((64, 64, 3), dtype=np.uint8)

        # render the dots in reverse order to ensure that the target is always visible
        for p, c in reversed(self.colored_dots):
            self.render_dot(observation, p, c)

        self.render_dot(observation, self.dot_position, np.array([255, 255, 255]))
        return observation
    
    def rand_point(self):
        return np.array([np.random.randint(0, 64 - 2), np.random.randint(0, 64 - 2)])
        
    def reset(self):
        self.dot_position = self.rand_point()
        self.current_step = 0
        self.colored_dots = []
        for i in range(64):
            self.colored_dots.append((self.rand_point(), np.array([0, 255, 0])))
        observation = self.get_obs()
        return observation

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(self.get_obs())

class RGBDotEnv(gym.Env):
    """
    An environment with sparse rewards. 
    Three dots are scattered randomly around the screen,
    and the agent gets +1 reward for each one they touch in order.
    """
    metadata = {'render.modes': ['human']}
    colors = [
        np.array([255, 0, 0]),
        np.array([0, 255, 0]),
        np.array([0, 0, 255]),
    ]

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self.dot_position = np.array([0, 0], dtype='int32')
        self.dot_color = np.array([255, 255, 255])
        self.viewer = None
        self.reward_range = (-float('inf'), float('inf'))
        self.current_step = 0
        self.colored_dots = []

    def step(self, action):
        # Move the dot according to the action
        if action == 0:  # Up
            self.dot_position[1] = max(self.dot_position[1] - 1, 0)
        elif action == 1:  # Right
            self.dot_position[0] = min(self.dot_position[0] + 1, 64 - 2)
        elif action == 2:  # Down
            self.dot_position[1] = min(self.dot_position[1] + 1, 64 - 2)
        elif action == 3:  # Left
            self.dot_position[0] = max(self.dot_position[0] - 1, 0)

        # Compute the reward based on the new position of the dot
        reward = 0
        if all(self.dot_position == self.colored_dots[0][0]):
            reward = 1
            self.colored_dots.pop(0)
            
        self.current_step += 1
        done = len(self.colored_dots) == 0 or self.current_step >= 1000
        info = {}

        # Update the observation
        observation = self.get_obs()
        
        return observation, reward, done, info
    
    def render_dot(self, obs, position, color):
        obs[position[1]:position[1]+2, position[0]:position[0]+2] = color
    
    def get_obs(self):
        observation = np.zeros((64, 64, 3), dtype=np.uint8)

        # render the dots in reverse order to ensure that the target is always visible
        for p, c in reversed(self.colored_dots):
            self.render_dot(observation, p, c)

        self.render_dot(observation, self.dot_position, np.array([255, 255, 255]))
        return observation
    
    def rand_point(self):
        return np.array([np.random.randint(0, 64 - 2), np.random.randint(0, 64 - 2)])
        
    def reset(self):
        self.dot_position = self.rand_point()
        self.current_step = 0
        self.colored_dots = []
        for c in self.colors:
            self.colored_dots.append((self.rand_point(), c))
        observation = self.get_obs()
        return observation

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(self.get_obs())

        
import gym
import numpy as np

action_to_vector = {
    0: np.array([0, -1]),  # Up
    1: np.array([1, 0]),   # Right
    2: np.array([0, 1]),   # Down
    3: np.array([-1, 0])   # Left
}

class SparseDotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.width = 64
        self.height = 64
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self.dot_position = np.array([0, 0], dtype='int32')
        self.dot_color = np.array([255, 255, 255])
        self.goal_position = np.array([self.width/2, self.height/2], dtype='int32')
        self.viewer = None
        self.reward_range = (-float('inf'), float('inf'))
        self.max_episode_steps = 200
        self.current_step = 0

    def step(self, action):
        # Move the dot according to the action
        if action == 0:  # Up
            self.dot_position[1] = max(self.dot_position[1] - 1, 0)
        elif action == 1:  # Right
            self.dot_position[0] = min(self.dot_position[0] + 1, self.width - 2)
        elif action == 2:  # Down
            self.dot_position[1] = min(self.dot_position[1] + 1, self.height - 2)
        elif action == 3:  # Left
            self.dot_position[0] = max(self.dot_position[0] - 1, 0)

        # Update the observation
        observation = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        observation[self.dot_position[1]:self.dot_position[1]+2, self.dot_position[0]:self.dot_position[0]+2] = self.dot_color

        self.current_step += 1
        finished = all(self.dot_position == self.goal_position)
        done = self.current_step >= self.max_episode_steps or finished
        info = {}
        
        reward = 1 if finished else 0

        return observation, reward, done, info
    
    def get_obs(self):
        observation = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        observation[self.dot_position[1]:self.dot_position[1]+2, self.dot_position[0]:self.dot_position[0]+2] = self.dot_color
        return observation

    def reset(self):
        self.dot_position = np.array([np.random.randint(0, self.width - 2), np.random.randint(0, self.height - 2)])
        observation = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        observation[self.dot_position[1]:self.dot_position[1]+2, self.dot_position[0]:self.dot_position[0]+2] = self.dot_color
        self.current_step = 0
        return observation

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(self.reset())

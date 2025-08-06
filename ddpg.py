# Micro-level Reinforcement Learning control (micro_ddpg.py)
# This script creates a Gym environment for car-following and trains a DDPG/TD3 agent.
# It uses trajectory data (veh_s, veh_f from data(1).mat) converted to the ego-vehicle's frame [oai_citation:15‡yulin-dev.github.io](https://yulin-dev.github.io/2019/11/26/NGSIM%E6%95%B0%E6%8D%AE%E9%9B%86/#:~:text=origin_x%20%3D%20trajs,y%281).
# The RL agent observes the headway distance, and both vehicle speeds, and outputs acceleration.
# The reward function combines multiple objectives (λ-weighted): maintaining a safe gap ~2 m [oai_citation:16‡github.com](https://github.com/caferavci/DPCAV#:~:text=rear,trajectories%20as%20solid%20blue%20lines), 
# minimizing relative speed (quick response to lead vehicle) [oai_citation:17‡github.com](https://github.com/caferavci/DPCAV#:~:text=While%20human,world%20geometry%20features), and smooth acceleration changes (speed stability).

import numpy as np
import scipy.io as sio
import gym
from gym import spaces

# Load trajectory data (MATLAB format)
mat = sio.loadmat('data(1).mat')
# Assuming structure with veh_s (ego vehicle) and veh_f (lead vehicle) in data(1)
veh_s_y = veh_f_y = None
veh_s_v = veh_f_v = None
dt = 0.1  # 10 Hz sampling rate [oai_citation:18‡github.com](https://github.com/caferavci/DPCAV#:~:text=Above%20Figure%20shows%20the%20layout,and%20green%20is%2020%20s)
if 'data' in mat:
    scenario = mat['data'][0,0]
    # Ego vehicle (subject vehicle)
    veh_s = scenario['veh_s'][0,0]
    veh_s_x = veh_s['x'].squeeze()
    veh_s_y = veh_s['y'].squeeze()
    # Lead vehicle (front vehicle)
    if 'veh_f' in scenario.dtype.names:
        veh_f = scenario['veh_f'][0,0]
        veh_f_x = veh_f['x'].squeeze()
        veh_f_y = veh_f['y'].squeeze()
    else:
        # No lead vehicle present
        veh_f_x = np.array([10000])  # marker as in data [oai_citation:19‡yulin-dev.github.io](https://yulin-dev.github.io/2019/11/26/NGSIM%E6%95%B0%E6%8D%AE%E9%9B%86/#:~:text=end%20if%20%28trajs.veh_f.x%281%29%20,trajs.veh_f.y%281%3A100%29%3B%20end)
        veh_f_y = np.array([0])
    # Convert trajectories to ego-vehicle frame [oai_citation:20‡yulin-dev.github.io](https://yulin-dev.github.io/2019/11/26/NGSIM%E6%95%B0%E6%8D%AE%E9%9B%86/#:~:text=origin_x%20%3D%20trajs,y%281)
    origin_x = veh_s_x[0]; origin_y = veh_s_y[0]
    veh_s_y_rel = veh_s_y - origin_y
    if veh_f_x.size > 0 and veh_f_x[0] != 10000:
        veh_f_y_rel = veh_f_y - origin_y
    else:
        veh_f_y_rel = np.zeros_like(veh_s_y_rel)
    # Approximate velocities (finite difference)
    veh_s_v = np.concatenate(([0.0], np.diff(veh_s_y) / dt))
    veh_f_v = np.concatenate(([0.0], np.diff(veh_f_y) / dt)) if veh_f_y is not None else np.zeros_like(veh_s_v)
else:
    raise RuntimeError("Trajectory data not found in data(1).mat")

# Define a Gym environment for the car-following scenario
class CarFollowingEnv(gym.Env):
    def __init__(self, leader_pos, leader_speed, total_steps):
        super().__init__()
        self.dt = dt
        self.total_steps = total_steps
        # Observation: [headway_distance, ego_speed, leader_speed]
        self.observation_space = spaces.Box(low=np.array([-1e3, 0.0, 0.0], dtype=np.float32),
                                            high=np.array([1e3, 50.0, 50.0], dtype=np.float32))
        # Action: acceleration (continuous, in m/s^2)
        self.action_space = spaces.Box(low=np.array([-3.0], dtype=np.float32),
                                       high=np.array([2.0], dtype=np.float32))
        # Save leader trajectory (relative positions & speeds)
        self.leader_pos = leader_pos
        self.leader_speed = leader_speed
        # Initialize state
        self.reset()
    def reset(self):
        # Reset environment to start state
        self.t_step = 0
        # Initial ego vehicle state (starting at position 0)
        self.ego_pos = 0.0
        self.ego_speed = float(veh_s_v[0]) if veh_s_v is not None else 0.0
        # Initial headway (distance to leader)
        self.current_headway = float(self.leader_pos[0] - self.ego_pos)
        # Initialize last acceleration for jerk calculation
        self.last_acc = 0.0
        # Return initial observation
        return np.array([self.current_headway, self.ego_speed, float(self.leader_speed[0])], dtype=np.float32)
    def step(self, action):
        acc = float(action[0])
        # Clip acceleration to allowable range
        acc = float(np.clip(acc, self.action_space.low[0], self.action_space.high[0]))
        # Update ego vehicle speed and position
        new_speed = self.ego_speed + acc * self.dt
        if new_speed < 0.0:
            new_speed = 0.0
        new_pos = self.ego_pos + self.ego_speed * self.dt + 0.5 * acc * (self.dt**2)
        # Advance leader to next step
        next_idx = min(self.t_step + 1, self.total_steps - 1)
        leader_pos_next = float(self.leader_pos[next_idx])
        leader_speed_next = float(self.leader_speed[next_idx])
        # Compute new headway distance
        headway = leader_pos_next - new_pos
        # Calculate reward components
        d0 = 2.0  # desired headway (m) [oai_citation:21‡github.com](https://github.com/caferavci/DPCAV#:~:text=rear,trajectories%20as%20solid%20blue%20lines)
        headway_error = headway - d0
        rel_speed = leader_speed_next - new_speed  # relative speed (leader minus ego) 
        jerk = acc - self.last_acc  # change in acceleration (for smoothness)
        # λ weights for each objective (tunable parameters)
        lam_headway = 1.0
        lam_resp   = 0.5
        lam_smooth = 0.2
        # Reward: negative weighted sum of squared errors for the objectives
        reward = - (lam_headway * (headway_error**2) + lam_resp * (rel_speed**2) + lam_smooth * (jerk**2))
        # Check for collision or unsafe scenario
        done = False
        if headway <= 0:  # collision or ego passed leader
            reward -= 100.0  # large penalty for collision
            done = True
        # Update ego state
        self.ego_speed = new_speed
        self.ego_pos = new_pos
        self.current_headway = headway
        self.last_acc = acc
        self.t_step += 1
        # End episode if out of data range
        if self.t_step >= self.total_steps - 1:
            done = True
        # Observation for next state
        obs = np.array([headway, self.ego_speed, leader_speed_next], dtype=np.float32)
        return obs, reward, done, {}
    def render(self, mode='human'):
        print(f"t={self.t_step * self.dt:.1f}s, headway={self.current_headway:.2f} m, ego_speed={self.ego_speed:.2f} m/s")

# Create the environment using the processed trajectory data
total_steps = len(veh_s_y)
env = CarFollowingEnv(veh_f_y_rel, veh_f_v, total_steps)

# Train a DDPG/TD3 agent on the environment (requires stable-baselines3)
try:
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
except ImportError:
    raise ImportError("Please install stable-baselines3 to train the RL agent.")
# Configure TD3 model and exploration noise
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, learning_rate=1e-3,
            buffer_size=10000, batch_size=64, train_freq=(1, "step"), seed=0)
# Train the agent (this may take time)
model.learn(total_timesteps=50000)
# Save the trained model
model.save("cav_acc_control_model")
print("Training completed and model saved.")

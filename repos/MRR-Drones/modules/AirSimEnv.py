# https://github.com/sunghoonhong/AirsimDRL

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import airsim
import time
import matplotlib.pyplot as plt
from resize_right import resize

import Airsim.lidar as lidar
import Airsim.camera as camera
from Airsim.drone_state import get_drone_state
from torchvision.transforms.v2 import Compose, Normalize, ToDtype, Grayscale
import torch
from collections import deque


class CustomEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        output_shape = (5, 7)
        # 10 point trajectory x,y,z,yaw (deltas?)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=output_shape, dtype=np.float32)

        image_shape = (5, 224, 224)
        self.image_space = spaces.Box(
            low=-1, high=1, shape=image_shape, dtype=np.float32)  # 300 * 300 *3 depth image

        meta_data_shape = (5, 15)
        # 15 * 15 time series based input data
        self.vector_space = spaces.Box(
            low=-1, high=1, shape=meta_data_shape, dtype=np.float32)

        self.observation_space = spaces.Dict({
            'image': self.image_space,
            'vector': self.vector_space
        })

        self.target = np.array([70, 0, -5])

        self.image_transforms = Compose([ToDtype(torch.float32), Grayscale(
            num_output_channels=3), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }
        # start env
        self.history = deque(maxlen=5)
        self.start()
        self.steps = 0

    def start(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        self.init_pose = self.client.simGetVehiclePose()

        self.client.simSetVehiclePose(self.init_pose, True)
        # self.client.simSetObjectPose()
        self.client.simPause(False)

    def step(self, action):
        delta = action[0, :3]
        # Only clip if exceeds max speed
        yawrot = action[0, 3].clip(-1, 1)

        if np.linalg.norm(delta) > 1:
            delta /= np.linalg.norm(delta)
        delta *= 3  # ~10kmph
        print(delta)

        # move
        # Convert numpy floats to python floats for msgpack
        delta = delta.tolist()
        moveTime = time.monotonic()
        self.client.rotateByYawRateAsync(yawrot, 1)
        # self.client.moveByVelocityBodyFrameAsync(delta[0], delta[1], delta[2], 1).join()
        self.client.moveByVelocityAsync(delta[0], delta[1], delta[2], 1).join()
        moveTime = time.monotonic() - moveTime
        # self.client.moveByVelocityAsync(position[0], position[1], position[2], 3).join()

        # get obvservation
        observation = self.get_obs()

        # compute reward
        current_position = self.state['position']

        distance_to_traget = np.linalg.norm(
            self.target - np.array([current_position.x_val, current_position.y_val, current_position.z_val]))

        # closest_points, closest_distances = lidar.get_closest_points(self.client, vis=False)
        # reward1 = -1/(np.sum(closest_distances) / len(closest_points))

        # reward = math.exp(-1 * distance_to_traget) - 0.5
        reward = np.linalg.norm(self.target)*2-distance_to_traget

        terminated = False
        if distance_to_traget < 1:
            terminated = True
            reward = 100

        if self.state["collision"]:
            reward = -100
            terminated = True

        if self.steps >= 35:
            terminated = True

        self.steps += 1
        print(f'{self.steps=} {reward=} {current_position=} {moveTime=}')
        return observation, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        # self.client.simPause(False)
        resetTime = time.monotonic()
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.client.simSetVehiclePose(self.init_pose, True)
        print(f'resetTime={time.monotonic() - resetTime}')
        # self.client.simPause(True)

        self.history.clear()
        self.steps = 0
        observation = self.get_obs()
        return observation, {}

    def render(self):
        ...

    def close(self):
        ...

    def get_obs(self):
        # get observation
        image = camera.get_image(self.client, "front_camera")
        print(image.shape)
        image = resize(image, out_shape=(224, 224))
        image = self.image_transforms(image)
        print(image.shape)
        current_drone_state = get_drone_state(self.client, self.target)

        drone_state = self.client.getMultirotorState()
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = drone_state.kinematics_estimated.position
        self.state["velocity"] = drone_state.kinematics_estimated.linear_velocity
        collision = self.client.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        self.history.append((image, current_drone_state))

        hist = list(self.history)  # copy references
        vector_feat = np.stack([h[1] for h in hist], axis=0)
        image_feat = np.stack([h[0] for h in hist], axis=0)
        if len(vector_feat) < 5:
            vector_feat = np.pad(vector_feat, ((0, 5-len(vector_feat)), (0, 0)),
                                mode='constant', constant_values=0)

        if len(image_feat) < 5:
            image_feat = np.pad(image_feat, ((0, 5-len(image_feat)), (0, 0), (0, 0)),
                                mode='constant', constant_values=0)

        return {'image': image_feat, 'vector': vector_feat}

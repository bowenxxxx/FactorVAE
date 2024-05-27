import logging
import random

import imgaug as ia
from imgaug import augmenters as iaa

import cv2
import h5py
import os
import torch
import numpy as np
from base_dataset import TrajectoryDataset
from libero_singleTask_dataset import sim_framework_path

log = logging.getLogger(__name__)

def get_max_data_len(data_directory: os.PathLike):
    if os.path.exists(data_directory):
        data_dir = data_directory
    else:
        print("data_path is missing")

    max_data_len = 0

    f = h5py.File(data_dir, 'r')
    demos = f['data']
    num_demos = len(list(f["data"].keys()))

    for i in range(num_demos):
        demo_name = f'demo_{i}'
        state = demos[demo_name]['states']
        length = state.shape[0]

        if length > max_data_len:
            max_data_len = length

    f.close()

    return max_data_len


data_aug_list = [
    # iaa.AdditiveGaussianNoise(scale=(5, 20)),
    iaa.ChangeColorTemperature((8000, 12000)),
    iaa.GammaContrast((0.8, 1.1), per_channel=True),
    # iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
    # iaa.WithColorspace(
    #     to_colorspace="HSV",
    #     from_colorspace="RGB",
    #     children=iaa.WithChannels(
    #         0,
    #         iaa.Add((0, 50))
    #     )
    # ),
    # iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))
]


class LiberoDataset(TrajectoryDataset):
    def __init__(
            self,
            data_directory: os.PathLike,
            # data='train',
            obs_keys,  # low_dim or rgb
            obs_modalities,
            dataset_keys=None,  # [actions, dones, obs, rewards, states]
            filter_by_attribute=None,
            padding=True,
            device="cpu",
            obs_dim: int = 32,
            action_dim: int = 7,
            state_dim: int = 45,
            max_len_data: int = 136,
            window_size: int = 1,
            traj_per_task: int = 1,
            data_aug: bool = False,
            aug_data_factor: float = 0.3,

    ):
        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size
        )

        self.data_aug = data_aug

        logging.info("Loading Libero Dataset")

        self.obs_keys = obs_keys  # low_dim || rgb
        logging.info("The dataset is {}".format(self.obs_keys))  #show low_dim or rgb

        self.data_dir = sim_framework_path(self.data_directory)
        logging.info("The dataset is loading from {}".format(self.data_dir))  # show the dataset directory

        self.obs_modalities = obs_modalities["obs"][self.obs_keys]
        logging.info("The obs_modalities list is {}".format(self.obs_modalities))

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.dataset_keys = dataset_keys  # [actions, dones, obs, rewards, states]
        self.filter_by_attribute = filter_by_attribute
        self.data_directory = data_directory
        self.aug_data_factor = aug_data_factor

        actions = []
        states = []
        rewards = []
        dones = []
        masks = []
        agentview_rgb = []
        eye_in_hand_rgb = []

        # self.data_dir: abspath
        file_list = os.listdir(self.data_dir)

        for file in file_list:
            if not file.endswith('.hdf5'):
                continue

            f = h5py.File(os.path.join(self.data_dir, file), 'r')

            log.info("Loading demo: {}".format(file))

            # get the image's basic shape from demo_0
            if self.obs_keys == "rgb":
                H, W, C = f["data"]["demo_0"]["obs"][self.obs_modalities[0]].shape[1:]

            # determinate which demo should be loaded using demo_keys_list
            if filter_by_attribute is not None:
                self.demo_keys_list = [elem.decode("utf-8") for elem in
                                       np.array(f["mask/{}".format(filter_by_attribute)][:])]
            else:
                self.demo_keys_list = list(f["data"].keys())

            indices = np.argsort([int(elem[5:]) for elem in self.demo_keys_list])
            num_demos = len(self.demo_keys_list)

            # load the states and actions in demos according to demo_keys_list
            for i in indices[-1 * traj_per_task:]:
                demo_name = f'demo_{i}'
                demo = f["data"][demo_name]
                demo_length = demo.attrs["num_samples"]

                # zero_states = np.zeros((1, self.max_len_data, self.state_dim), dtype=np.float32)
                zero_actions = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
                # zero_rewards = np.zeros((1, self.max_len_data), dtype=np.float32)
                # zero_dones = np.zeros((1, self.max_len_data), dtype=np.float32)
                zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

                # states_data = demo['states'][:]
                action_data = demo['actions'][:]

                zero_actions[0, :demo_length, :] = action_data
                zero_mask[0, :demo_length] = 1

                the_last_action = action_data[-1][:]

                agent_view = demo['obs']['agentview_rgb'][:]
                eye_in_hand = demo['obs']['eye_in_hand_rgb'][:]

                actions.append(zero_actions)
                masks.append(zero_mask)

                agentview_rgb.append(agent_view)
                eye_in_hand_rgb.append(eye_in_hand)

            f.close()

        self.actions = torch.from_numpy(np.concatenate(actions)).to(device).float()  # shape: B, T, D

        self.agentview_rgb = agentview_rgb
        self.eye_in_hand_rgb = eye_in_hand_rgb

        self.masks = torch.from_numpy(np.concatenate(masks)).to(device).float()

        self.num_data = len(self.agentview_rgb)

        self.slices = self.get_slices()

    def get_slices(self):
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.window_size < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={self.window_size}")
            else:
                slices += [
                    (i, start, start + self.window_size) for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        result = []
        # mask out invalid observations
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.agentview_rgb[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        i, start, end = self.slices[idx]

        agentview_rgb = self.agentview_rgb[i][start:end]
        eye_in_hand_rgb = self.eye_in_hand_rgb[i][start:end]

        # cv2.imshow('agent', agentview_rgb[0])

        if self.data_aug is True and random.random() > (1 - self.aug_data_factor):
            seq = random.sample(data_aug_list, 1)[0]
            agentview_rgb = seq(images=agentview_rgb)
            # cv2.imshow('aug', agentview_rgb[0])
            # cv2.waitKey(0)

        # add torch here
        agentview_rgb = torch.from_numpy(agentview_rgb).to(self.device).float().permute(0, 3, 1, 2) / 255.
        eye_in_hand_rgb = torch.from_numpy(eye_in_hand_rgb).to(self.device).float().permute(0, 3, 1, 2) / 255.

        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        return agentview_rgb, eye_in_hand_rgb, act, mask


class LiberoDataset2(TrajectoryDataset):
    def __init__(
            self,
            data_directory: os.PathLike,
            # data='train',
            obs_keys,  # low_dim or rgb
            obs_modalities,
            dataset_keys=None,  # [actions, dones, obs, rewards, states]
            filter_by_attribute=None,
            padding=True,
            device="cpu",
            obs_dim: int = 64,
            action_dim: int = 7,
            state_dim: int = 110,
            max_len_data: int = 260,
            window_size: int = 1,
            traj_per_task: int = 10,
            data_aug: bool = True,
            aug_data_factor: float = 0.3,
    ):
        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size
        )

        self.data_aug = data_aug

        logging.info("Loading Libero Dataset")

        self.obs_keys = obs_keys  # low_dim || rgb
        logging.info("The dataset is {}".format(self.obs_keys))  #show low_dim or rgb

        self.data_dir = sim_framework_path(self.data_directory)
        logging.info("The dataset is loading from {}".format(self.data_dir))  # show the dataset directory

        self.obs_modalities = obs_modalities["obs"][self.obs_keys]
        logging.info("The obs_modalities list is {}".format(self.obs_modalities))

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.dataset_keys = dataset_keys  # [actions, dones, obs, rewards, states]
        self.filter_by_attribute = filter_by_attribute
        self.data_directory = data_directory
        self.aug_data_factor = aug_data_factor

        actions = []
        states = []
        rewards = []
        dones = []
        masks = []
        agentview_rgb = []
        eye_in_hand_rgb = []

        # self.data_dir: abspath
        file_list = os.listdir(self.data_dir)

        for file in file_list:
            if not file.endswith('.hdf5'):
                continue

            f = h5py.File(os.path.join(self.data_dir, file), 'r')

            log.info("Loading demo: {}".format(file))

            # get the image's basic shape from demo_0
            if self.obs_keys == "rgb":
                H, W, C = f["data"]["demo_0"]["obs"][self.obs_modalities[0]].shape[1:]

            # determinate which demo should be loaded using demo_keys_list
            if filter_by_attribute is not None:
                self.demo_keys_list = [elem.decode("utf-8") for elem in
                                       np.array(f["mask/{}".format(filter_by_attribute)][:])]
            else:
                self.demo_keys_list = list(f["data"].keys())

            indices = np.argsort([int(elem[5:]) for elem in self.demo_keys_list])
            num_demos = len(self.demo_keys_list)

            # load the states and actions in demos according to demo_keys_list
            for i in indices[:traj_per_task]:
                demo_name = f'demo_{i}'
                demo = f["data"][demo_name]
                demo_length = demo.attrs["num_samples"]

                # zero_states = np.zeros((1, self.max_len_data, self.state_dim), dtype=np.float32)
                zero_actions = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
                # zero_rewards = np.zeros((1, self.max_len_data), dtype=np.float32)
                # zero_dones = np.zeros((1, self.max_len_data), dtype=np.float32)
                zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

                # states_data = demo['states'][:]
                action_data = demo['actions'][:]
                # rewards_data = demo['rewards'][:]
                # dones_data = demo['dones'][:]

                # zero_states[0, :demo_length, :] = states_data  # would be T0, ...,Tn-1, Tn, 0, 0
                zero_actions[0, :demo_length, :] = action_data
                # zero_rewards[0, :demo_length] = rewards_data
                # zero_dones[0, :demo_length] = dones_data
                zero_mask[0, :demo_length] = 1

                # the_last_state = states_data[-1][:]
                the_last_action = action_data[-1][:]
                # the_last_reward = rewards_data[-1]
                # the_last_done = dones_data[-1]

                # zero_agentview = np.zeros((self.max_len_data, H, W, C), dtype=np.float32)
                # zero_inhand = np.zeros((self.max_len_data, H, W, C), dtype=np.float32)
                agent_view = demo['obs']['agentview_rgb'][:]
                eye_in_hand = demo['obs']['eye_in_hand_rgb'][:]

                the_last_agentview = agent_view[-1, :, :, :]
                the_last_eye_in_hand_rgb = eye_in_hand[-1, :, :, :]

                # zero_agentview[ :demo_length, :, :, :] = agent_view
                # zero_inhand[ :demo_length, :, :, :] = eye_in_hand

                # if padding:
                #     zero_states[0, demo_length:, :] = the_last_state  # the sequence would be T0, ..., Tn-1, Tn, Tn, Tn
                #     zero_actions[0, demo_length:, :] = the_last_action
                #     zero_rewards[0, demo_length:] = the_last_reward
                #     zero_dones[0, demo_length:] = the_last_done
                #     zero_mask[0, :] = 1
                #     zero_agentview[0, demo_length:, :, :, :] = the_last_agentview
                #     zero_inhand[0, demo_length:, :, :, :] = the_last_eye_in_hand_rgb

                # states.append(zero_states)
                actions.append(zero_actions)
                # rewards.append(zero_rewards)
                # dones.append(zero_dones)
                masks.append(zero_mask)

                # zero_agentview = torch.from_numpy(agent_view).to(device).float().permute(0, 3, 1, 2) / 255.
                # zero_inhand = torch.from_numpy(eye_in_hand).to(device).float().permute(0, 3, 1, 2) / 255.

                agentview_rgb.append(agent_view)
                eye_in_hand_rgb.append(eye_in_hand)

            f.close()

        # self.states = torch.from_numpy(np.concatenate(states)).to(device).float()
        self.actions = torch.from_numpy(np.concatenate(actions)).to(device).float()  # shape: B, T, D

        self.agentview_rgb = agentview_rgb
        self.eye_in_hand_rgb = eye_in_hand_rgb

        # self.rewards = torch.from_numpy(np.concatenate(rewards)).to(device).float()
        # self.dones = torch.from_numpy(np.concatenate(dones)).to(device).float()
        self.masks = torch.from_numpy(np.concatenate(masks)).to(device).float()

        self.num_data = len(self.agentview_rgb)

        self.slices = self.get_slices()

    def get_slices(self):  #Extract sample slices that meet certain conditions
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.window_size < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={self.window_size}")
            else:
                slices += [
                    (i, start, start + self.window_size) for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    # def get_obs_dim(self, obs_modalities: list):
    #     f = h5py.File(self.data_dir, 'r')
    #
    #     obs_dim = 0
    #     for key in obs_modalities:
    #         obs_modality = f["data"]["demo_0"]["obs"]["{}".format(key)]
    #         obs_dim += obs_modality.shape[1]
    #
    #     f.close()
    #     return obs_dim

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        result = []
        # mask out invalid observations
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.agentview_rgb[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        i, start, end = self.slices[idx]

        agentview_rgb = self.agentview_rgb[i][start:end]
        eye_in_hand_rgb = self.eye_in_hand_rgb[i][start:end]


        if self.data_aug is True and random.random() > (1 - self.aug_data_factor):
            # cv2.imshow('agent', agentview_rgb[0])
            seq = random.sample(data_aug_list, 1)[0]
            agentview_rgb = seq(images=agentview_rgb)
            # cv2.imshow('aug', agentview_rgb[0])
            # cv2.waitKey(0)

        # add torch here
        agentview_rgb = torch.from_numpy(agentview_rgb).to(self.device).float().permute(0, 3, 1, 2) / 255.
        eye_in_hand_rgb = torch.from_numpy(eye_in_hand_rgb).to(self.device).float().permute(0, 3, 1, 2) / 255.

        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        return agentview_rgb, eye_in_hand_rgb, act, mask

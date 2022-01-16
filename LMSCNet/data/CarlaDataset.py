## Maintainer: Jingyu Song, Arthur Zhang #####
## Contact: jingyuso@umich.edu, arthurzh@umich.edu #####

import os
import numpy as np
import random
import json
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def collate_fn_test(data):
    input_batch = []
    output_batch = []
    for bi in data:
        input_batch.append(bi[0])
        output_batch.append(bi[1])
    return input_batch, torch.from_numpy(np.array(output_batch)).cuda()

class CarlaDataset_dataloader(Dataset):
    """Carla Simulation Dataset for 3D mapping project
    
    Access to the processed data, including evaluation labels predictions velodyne poses times
    """
    def __init__(self, dataset,
        phase,
        device='cuda',
        num_frames=4
        ):
        '''Constructor.
        Parameters:
            :param dataset: The dataset configuration (data augmentation, input encoding, etc)
            :param phase_tag: To differentiate between training, validation and test phase
        '''

        self._directory = dataset['ROOT_DIR']
        self._num_frames = num_frames
        # self._out_dim = out_dim
        self._device = device
        print("directory ", self._directory)
        self._scenes = sorted(os.listdir(self._directory))
        self._num_scenes = len(self._scenes)
        self._num_frames_scene = []

        '''
        LMSCNet specific
        '''
        yaml_path, _ = os.path.split(os.path.realpath(__file__))
        self.dataset_config = yaml.safe_load(open(os.path.join(yaml_path, 'carla.yaml'), 'r'))
        self.nbr_classes = self.dataset_config['nbr_classes']
        self.grid_dimensions = self.dataset_config['grid_dims']   # [W, H, D]
        self.remap_lut = self.get_remap_lut()
        self.rgb_mean = np.array([0.34749558, 0.36745213, 0.36123651])  # images mean:  [88.61137282 93.70029365 92.11530949]
        self.rgb_std = np.array([0.30599035, 0.3129534 , 0.31933814])   # images std:  [78.02753826 79.80311686 81.43122464]

        self.filepaths = {}
        self.phase = phase
        self.class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
                                        6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
                                        2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
                                        2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
                                        2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])

        self.split = self.dataset_config['split']

        # load evaluation

        # grid_size = self._eval_param['grid_size']
        # min_bound = self._eval_param['min_bound']
        # max_bound = self._eval_param['max_bound']
        # num_channels = self._eval_param['num_channels']
        # coordinates = self._eval_param['coordinates']
        # self._cylinder_mat = ShapeContainer(grid_size, min_bound, max_bound, num_channels, coordinates)
        self._eval_size = [self.nbr_classes] + list(np.uint32(self.grid_dimensions))

        self._velodyne_list = []
        self._label_list = []
        self._pred_list = []
        self._eval_list = []
        self._frames_list = []
        self._timestamps = []
        self._poses = []

        # for each scene, fill in the data list
        self.get_filepaths()

        # self._frames = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(self._velodyne_dir))]
        # self._length = len(self._frames)

    def get_filepaths(self):
        '''
        Set modality filepaths with split according to phase (train, val, test)
        '''

        for scene in self._scenes:
            self.filepaths['3D_OCCUPANCY'] = os.path.join(self._directory, scene, 'velodyne')
            
            self._num_frames_scene.append(len(os.listdir(self.filepaths['3D_OCCUPANCY'])))

            frames_list = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(self.filepaths['3D_OCCUPANCY']))]
            self._velodyne_list.extend([os.path.join(self.filepaths['3D_OCCUPANCY'], str(frame).zfill(6)+'.bin') for frame in frames_list])
            self._timestamps.append(np.loadtxt(os.path.join(self._directory, scene, 'times.txt')))
            self._poses.append(np.loadtxt(os.path.join(self._directory, scene, 'poses.txt')))
            # for poses and timestamps

            if self.phase != 'test':
                self.filepaths['3D_LABEL'] = os.path.join(self._directory, scene, 'labels')
                self.filepaths['3D_PREDICTIONS'] = os.path.join(self._directory, scene, 'predictions')
                self.filepaths['3D_EVALUATION'] = os.path.join(self._directory, scene, 'evaluation')
                self._label_list.extend([os.path.join(self.filepaths['3D_LABEL'], str(frame).zfill(6)+'.label') for frame in frames_list])
                self._pred_list.extend([os.path.join(self.filepaths['3D_PREDICTIONS'], str(frame).zfill(6)+'.bin') for frame in frames_list])
                self._eval_list.extend([os.path.join(self.filepaths['3D_EVALUATION'], str(frame).zfill(6)+'.bin') for frame in frames_list])
        
        self._timestamps = np.array(self._timestamps).reshape(sum(self._num_frames_scene))
        self._poses = np.array(self._poses).reshape(sum(self._num_frames_scene), 12)
        
        self._cum_num_frames = np.cumsum(np.array(self._num_frames_scene) - self._num_frames + 1)


    def get_inv_remap_lut(self):
        '''
        remap_lut to remap classes of semantic kitti for training...
        :return:
        '''

        # make lookup table for mapping
        maxkey = max(self.dataset_config['learning_map_inv'].keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 1), dtype=np.int32)
        remap_lut[list(self.dataset_config['learning_map_inv'].keys())] = list(self.dataset_config['learning_map_inv'].values())

        return remap_lut


    def get_remap_lut(self):
        '''
        remap_lut to remap classes of semantic kitti for training...
        :return:
        '''

        # make lookup table for mapping
        maxkey = max(self.dataset_config['learning_map'].keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(self.dataset_config['learning_map'].keys())] = list(self.dataset_config['learning_map'].values())

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.

        return remap_lut

    def __len__(self):
        return sum(self._num_frames_scene)-(self._num_frames-1)*len(self._scenes)
    
    def __getitem__(self, idx):
        # for idx i: list of frame i - i+num_frames-1 were loaded and returned
        # write a function to get the range
        idx_range = self.find_horizon(idx)
        
        # idx_range = np.arange(idx, idx+self._num_frames) 
        self._current_horizon = []
        for i in idx_range:
            pcl = np.fromfile(self._velodyne_list[i],dtype=np.float32).reshape(-1,4)
            pred = np.fromfile(self._pred_list[i],dtype=np.float32).reshape(-1,3)
            label = np.fromfile(self._label_list[i],dtype=np.uint32)
            
            # one hot encoding for each label
            label_oh = np.zeros((label.size, self.nbr_classes))
            label_oh[np.arange(label.size),label] = 1
            label_oh = np.float32(label_oh)
            points = np.c_[pcl, pred, label_oh]
            self._current_horizon.append(torch.from_numpy(points).to(self._device))

        output = np.fromfile(self._eval_list[idx_range[-1]],dtype=np.float32).reshape(self._eval_size)

        return self._current_horizon, output
        
        # no enough frames
    
    def find_horizon(self, idx):
        scene_idx = len(np.where(self._cum_num_frames <= idx)[0])
        idx_range = np.arange(idx + (self._num_frames-1)*scene_idx, idx + (self._num_frames-1)*scene_idx+self._num_frames)


        return idx_range
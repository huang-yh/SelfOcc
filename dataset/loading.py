import os.path
import mmcv
import numpy as np
from mmengine import BaseStorageBackend, FileClient
from typing import Union
from pathlib import Path
from ffrecord import FileReader
import pickle


class LoadMultiViewImageFromFilesHF:
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def load(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)
        filename = results
        # img is of shape (h, w, c, num_views)
        if self.file_client_args['backend'] == 'ffrecord':
            img_list = self.file_client.get(filename)
        else:
            img_list = [self.file_client.get(name ) for name in filename]
        img_list = [mmcv.imfrombytes(img, flag=self.color_type) for img in img_list]
        img = np.stack(img_list, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        img_list = [img[..., i] for i in range(img.shape[-1])]
        return img_list

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)
        if isinstance(results, dict):
            filename = results['img_filename']
        else:
            filename = results
        # img is of shape (h, w, c, num_views)
        if self.file_client_args['backend'] == 'ffrecord':
            img_list = self.file_client.get(filename)
        else:
            img_list = [self.file_client.get(name) for name in filename]
        img_list = [mmcv.imfrombytes(img, flag=self.color_type) for img in img_list]
        img = np.stack(img_list, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        img_list = [img[..., i] for i in range(img.shape[-1])]
        return img_list


class OriLoadMultiViewImageFromFilesHF(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        if self.file_client_args['backend'] == 'ffrecord':
            img_list = self.file_client.get(filename)
        else:
            img_list = [self.file_client.get(name) for name in filename]
        img_list = [mmcv.imfrombytes(img, flag=self.color_type) for img in img_list]
        img = np.stack(img_list, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


class LoadPtsFromFilesHF:

    def __init__(self,
                 to_float32=True,
                 file_client_args=None):
        self.to_float32 = to_float32
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def load(self, filename):

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)
        # img is of shape (h, w, c, num_views)
        if self.file_client_args['backend'] == 'ffrecord':
            pts = self.file_client.get([filename])[0]
        else:
            pts = self.file_client.get(filename)
        # print(pts)
        # print(len(pts))
        # print(type(pts))
        # print(type(pts[0]))
        tot_len = len(pts)
        counts = int((tot_len - 159) / 4)
        pts = np.frombuffer(pts, dtype=np.float32, offset=152, count=counts)
        # print(pts)
        # print(len(pts))
        # print(type(pts))
        # print(type(pts[0]))
        return pts


@FileClient.register_backend('ffrecord')
class FFrecordClient(BaseStorageBackend):
    """FFRecord storage backend."""
    _allow_symlink = True

    def __init__(self, fname, check_data=False, filename2idx='filename2idx.pkl'):
        assert os.path.exists(fname), f'Pls convert to ffr first! {fname} not exists!'
        self.reader = FileReader(fname, check_data)
        filename2idx_file = os.path.join(fname, filename2idx)
        assert os.path.exists(filename2idx_file), f'Pls convert to ffr first! {filename2idx_file} not exists!'
        with open(filename2idx_file, 'rb') as handle:
            filename2idx_file = pickle.load(handle)
        self.filename2idx_file = {}
        for k, v in filename2idx_file.items():
            self.filename2idx_file[os.path.basename(k)] = v

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        single_flag = False
        if not isinstance(filepath, list):
            filepath = [filepath]
            single_flag = True

        indices = []
        for name in filepath:
            file_basename = os.path.basename(name)
            idx = self.filename2idx_file[file_basename]
            indices.append(idx)
        sample = self.reader.read(indices)

        if single_flag:
            sample = sample[0]
        return sample

    def get_text(self,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        with open(filepath, 'r', encoding=encoding) as f:
            value_buf = f.read()
        return value_buf


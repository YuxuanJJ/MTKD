from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY
from os import path as osp
import cv2
import random
import torch


@DATASET_REGISTRY.register()
class MultiteaImageDataset3(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_tea1 (str): Data root path for tea1 output.
        dataroot_tea2 (str): Data root path for tea2 output.
        dataroot_tea3 (str): Data root path for tea3 output.

        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(MultiteaImageDataset3, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.tea1_folder, self.tea2_folder, self.tea3_folder = opt['dataroot_gt'], opt['dataroot_tea1'], opt['dataroot_tea2'], opt['dataroot_tea3']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = multitea_paths_from_folder([self.tea1_folder, self.tea2_folder, self.tea3_folder, self.gt_folder], ['tea1', 'tea2', 'tea3', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        tea1_path = self.paths[index]['tea1_path']
        img_bytes = self.file_client.get(tea1_path, 'tea1')
        img_tea1 = imfrombytes(img_bytes, float32=True)

        tea2_path = self.paths[index]['tea2_path']
        img_bytes = self.file_client.get(tea2_path, 'tea2')
        img_tea2 = imfrombytes(img_bytes, float32=True)

        tea3_path = self.paths[index]['tea3_path']
        img_bytes = self.file_client.get(tea3_path, 'tea3')
        img_tea3 = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_tea1, img_tea2, img_tea3 = paired_random_crop(img_gt, img_tea1, img_tea2, img_tea3, gt_size, gt_path)
            # flip, rotation
            img_gt, img_tea1, img_tea2, img_tea3 = augment([img_gt, img_tea1, img_tea2, img_tea3], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_tea1 = bgr2ycbcr(img_tea1, y_only=True)[..., None]
            img_tea2 = bgr2ycbcr(img_tea2, y_only=True)[..., None]
            img_tea3 = bgr2ycbcr(img_tea3, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_tea1.shape[0], 0:img_tea1.shape[1], :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_tea1, img_tea2, img_tea3 = img2tensor([img_gt, img_tea1, img_tea2, img_tea3], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_tea1, self.mean, self.std, inplace=True)
            normalize(img_tea2, self.mean, self.std, inplace=True)
            normalize(img_tea3, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'tea1': img_tea1, 'tea2': img_tea2, 'tea3': img_tea3, 'gt': img_gt, 'tea1_path': tea1_path, 'tea2_path': tea2_path, 'tea3_path': tea3_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


def multitea_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [tea1_folder, tea2_folder, tea3_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['tea1', 'tea2', 'tea3', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 4, ('The len of folders should be 4 with [tea1_folder, tea2_folder, tea3_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 4, f'The len of keys should be 4 with [tea1_key, tea2_key, tea3_key, gt_key]. But got {len(keys)}'
    tea1_folder, tea2_folder, tea3_folder, gt_folder = folders
    tea1_key, tea2_key, tea3_key, gt_key = keys

    tea1_paths = list(scandir(tea1_folder))
    tea2_paths = list(scandir(tea2_folder))
    tea3_paths = list(scandir(tea3_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(tea1_paths) == len(gt_paths), (f'{tea1_key} and {gt_key} datasets have different number of images: '
                                               f'{len(tea1_paths)}, {len(gt_paths)}.')
    assert len(tea2_paths) == len(gt_paths), (f'{tea2_key} and {gt_key} datasets have different number of images: '
                                              f'{len(tea2_paths)}, {len(gt_paths)}.')
    assert len(tea3_paths) == len(gt_paths), (f'{tea3_key} and {gt_key} datasets have different number of images: '
                                              f'{len(tea3_paths)}, {len(gt_paths)}.')
    paths = []
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))
        tea1_name = f'{filename_tmpl.format(basename)}{ext}'
        tea1_path = osp.join(tea1_folder, tea1_name)
        tea2_name = f'{filename_tmpl.format(basename)}{ext}'
        tea2_path = osp.join(tea2_folder, tea2_name)
        tea3_name = f'{filename_tmpl.format(basename)}{ext}'
        tea3_path = osp.join(tea3_folder, tea3_name)
        assert tea1_name in tea1_paths, f'{tea1_name} is not in {tea1_key}_paths.'
        assert tea2_name in tea2_paths, f'{tea2_name} is not in {tea2_key}_paths.'
        assert tea3_name in tea3_paths, f'{tea3_name} is not in {tea3_key}_paths.'
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(dict([(f'{tea1_key}_path', tea1_path), (f'{tea2_key}_path', tea2_path), (f'{tea3_key}_path', tea3_path), (f'{gt_key}_path', gt_path)]))

    return paths


def paired_random_crop(img_gts, img_tea1s, img_tea2s, img_tea3s, gt_patch_size, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_tea1s (list[ndarray] | ndarray): Tea1 images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_tea2s (list[ndarray] | ndarray): Tea2 images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_tea3s (list[ndarray] | ndarray): Tea3 images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_tea1s, list):
        img_tea1s = [img_tea1s]
    if not isinstance(img_tea2s, list):
        img_tea2s = [img_tea2s]
    if not isinstance(img_tea3s, list):
        img_tea3s = [img_tea3s]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_tea1, w_tea1 = img_tea1s[0].size()[-2:]
        h_tea2, w_tea2 = img_tea2s[0].size()[-2:]
        h_tea3, w_tea3 = img_tea3s[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_tea1, w_tea1 = img_tea1s[0].shape[0:2]
        h_tea2, w_tea2 = img_tea2s[0].shape[0:2]
        h_tea3, w_tea3 = img_tea3s[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    tea1_patch_size = gt_patch_size
    tea2_patch_size = gt_patch_size
    tea3_patch_size = gt_patch_size

    if h_gt != h_tea1 or w_gt != w_tea1 or h_gt != h_tea2 or w_gt != w_tea2 or h_gt != h_tea3 or w_gt != w_tea3:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not equal to ',
                         f'multiplication of Tea1 ({h_tea1}, {w_tea1}) Tea2 ({h_tea2}, {w_tea2}) Tea3 ({h_tea3}, {w_tea3}).')
    if h_tea1 < tea1_patch_size or w_tea1 < tea1_patch_size:
        raise ValueError(f'LQ ({h_tea1}, {w_tea1}) is smaller than patch size '
                         f'({tea1_patch_size}, {tea1_patch_size}). '
                         f'Please remove {gt_path}.')
    if h_tea2 < tea2_patch_size or w_tea2 < tea2_patch_size:
        raise ValueError(f'LQ ({h_tea2}, {w_tea2}) is smaller than patch size '
                         f'({tea2_patch_size}, {tea2_patch_size}). '
                         f'Please remove {gt_path}.')
    if h_tea3 < tea3_patch_size or w_tea3 < tea3_patch_size:
        raise ValueError(f'LQ ({h_tea3}, {w_tea3}) is smaller than patch size '
                         f'({tea3_patch_size}, {tea3_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for patch
    top = random.randint(0, h_gt - gt_patch_size)
    left = random.randint(0, w_gt - gt_patch_size)

    # crop patch
    if input_type == 'Tensor':
        img_tea1s = [v[:, :, top:top + gt_patch_size, left:left + gt_patch_size] for v in img_tea1s]
    else:
        img_tea1s = [v[top:top + gt_patch_size, left:left + gt_patch_size, ...] for v in img_tea1s]

    if input_type == 'Tensor':
        img_tea2s = [v[:, :, top:top + gt_patch_size, left:left + gt_patch_size] for v in img_tea2s]
    else:
        img_tea2s = [v[top:top + gt_patch_size, left:left + gt_patch_size, ...] for v in img_tea2s]

    if input_type == 'Tensor':
        img_tea3s = [v[:, :, top:top + gt_patch_size, left:left + gt_patch_size] for v in img_tea3s]
    else:
        img_tea3s = [v[top:top + gt_patch_size, left:left + gt_patch_size, ...] for v in img_tea3s]

    if input_type == 'Tensor':
        img_gts = [v[:, :, top:top + gt_patch_size, left:left + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top:top + gt_patch_size, left:left + gt_patch_size, ...] for v in img_gts]

    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_tea1s) == 1:
        img_tea1s = img_tea1s[0]
    if len(img_tea2s) == 1:
        img_tea2s = img_tea2s[0]
    if len(img_tea3s) == 1:
        img_tea3s = img_tea3s[0]
    return img_gts, img_tea1s, img_tea2s, img_tea3s


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


import numpy as np
# from mmseg.utils import get_root_logger
from mmengine import MMLogger
logger = MMLogger.get_instance('selfocc')
import torch, torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

def cityscapes2semantickitti(sem):
    lut_table = torch.tensor([
        9, # 0: road -> road
        11, # 1: sidewalk -> sidewalk
        13, # 2: building -> building
        13, # 3: wall -> building
        14, # 4: fence -> fence
        18, # 5: pole -> pole
        19, # 6: traffic light -> traffic sign
        19, # 7: traffic sign -> traffic sign
        15, # 8: vegetation -> vegetation
        17, # 9: terrain -> terrain
        0, # 10: sky -> unlabeled
        6, # 11: person -> person
        7, # 12: rider -> bicyclist
        1, # 13: car -> car
        4, # 14: truck -> truck
        5, # 15: bus -> other vehicle
        5, # 16: train -> other vehicle
        3, # 17: motorcycle -> motorcycle
        2, # 18: bicycle -> bicycle
        # "motorcyclist", "parking", "other-ground", "trunk"
    ], dtype=sem.dtype, device=sem.device)
    sem_shape = sem.shape
    sem = lut_table[sem.flatten()].reshape(*sem_shape)
    return sem

def openseed2nuscenes(sem):
    lut_table = torch.tensor([
        1, # 0: barrier -> barrier
        2, # 1: bicycle -> bicycle
        3, # 2: bus -> bus
        4, # 3: car -> car
        5, # 4: construction_vehicle -> construction_vehicle
        5, # 5: crane -> construction_vehicle
        6, # 6: motorcycle -> motorcycle
        7, # 7: person -> person
        8, # 8: traffic_cone -> traffic_cone
        9, # 9: trailer -> trailer
        9, # 10: trailer_truck -> trailer_truck
        10, # 11: truck -> truck
        11, # 12: road -> road
        12, # 13: other_flat -> other_flat
        13, # 14: sidewalk -> sidewalk
        14, # 15: terrain -> terrain
        14, # 16: grass -> terrain
        15, # 17: building -> building
        15, # 18: wall -> building
        16, # 19: tree -> vegetation
        0, # 20: sky -> unlabeled
        # "motorcyclist", "parking", "other-ground", "trunk"
    ], dtype=sem.dtype, device=sem.device)
    sem_shape = sem.shape
    sem = lut_table[sem.flatten()].reshape(*sem_shape)
    return sem

class MeanIoU:

    def __init__(self,
                 class_indices,
                #  ignore_label: int,
                 empty_label,
                 label_str,
                 use_mask=False,
                 dataset_empty_label=17,
                 name = 'none'):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        # self.ignore_label = ignore_label
        self.empty_label = empty_label
        self.dataset_empty_label = dataset_empty_label
        self.label_str = label_str
        self.use_mask = use_mask
        self.name = name

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes+1).cuda()
        self.total_correct = torch.zeros(self.num_classes+1).cuda()
        self.total_positive = torch.zeros(self.num_classes+1).cuda()

    def _after_step(self, outputs, targets, mask=None):
        # outputs = outputs[targets != self.ignore_label]
        # targets = targets[targets != self.ignore_label]
        if not isinstance(targets, (torch.Tensor, np.ndarray)):
            assert mask is None
            labels = torch.from_numpy(targets['semantics']).cuda()
            masks = torch.from_numpy(targets['mask_camera']).bool().cuda()
            targets = labels
            targets[targets == self.dataset_empty_label] = self.empty_label
            max_z = (targets != self.empty_label).nonzero()[:, 2].max()
            min_z = (targets != self.empty_label).nonzero()[:, 2].min()
            outputs[..., (max_z + 1):] = self.empty_label
            outputs[..., :min_z] = self.empty_label
            if self.use_mask:
                outputs = outputs[masks]
                targets = targets[masks]
        else:
            if mask is not None:
                outputs = outputs[mask]
                targets = targets[mask]

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(targets == c).item()
            self.total_correct[i] += torch.sum((targets == c)
                                               & (outputs == c)).item()
            self.total_positive[i] += torch.sum(outputs == c).item()
        
        self.total_seen[-1] += torch.sum(targets != self.empty_label).item()
        self.total_correct[-1] += torch.sum((targets != self.empty_label)
                                            & (outputs != self.empty_label)).item()
        self.total_positive[-1] += torch.sum(outputs != self.empty_label).item()

    def _after_epoch(self):
        if dist.is_initialized():
            dist.all_reduce(self.total_seen)
            dist.all_reduce(self.total_correct)
            dist.all_reduce(self.total_positive)
            dist.barrier()

        ious = []
        precs = []
        recas = []

        for i in range(self.num_classes):
            if self.total_positive[i] == 0:
                precs.append(0.)
            else:
                cur_prec = self.total_correct[i] / self.total_positive[i]
                precs.append(cur_prec.item())
            if self.total_seen[i] == 0:
                ious.append(1)
                recas.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                cur_reca = self.total_correct[i] / self.total_seen[i]
                ious.append(cur_iou.item())
                recas.append(cur_reca)

        miou = np.mean(ious)
        # logger = get_root_logger()
        logger.info(f'Validation per class iou {self.name}:')
        for iou, prec, reca, label_str in zip(ious, precs, recas, self.label_str):
            logger.info('%s : %.2f%%, %.2f, %.2f' % (label_str, iou * 100, prec, reca))
        
        logger.info(self.total_seen.int())
        logger.info(self.total_correct.int())
        logger.info(self.total_positive.int())

        occ_iou = self.total_correct[-1] / (self.total_seen[-1]
                                            + self.total_positive[-1]
                                            - self.total_correct[-1])
        # logger.info(f'iou: {occ_iou}')
        
        return miou * 100, occ_iou * 100


class IoU(nn.Module):

    def __init__(self, use_mask=False):
        super().__init__()        
        self.class_indices = [0]
        self.num_classes = 1
        self.label_str = ['occupied']
        self.use_mask = use_mask
        xx = torch.linspace(-40.0, 40.0, 200)
        yy = torch.linspace(-40.0, 40.0, 200)
        zz = torch.linspace(-1.0, 5.4, 16)
        xyz = torch.stack([
            xx[:, None, None].expand(-1, 200, 16),
            yy[None, :, None].expand(200, -1, 16),
            zz[None, None, :].expand(200, 200, -1)
        ], dim=-1)
        self.register_buffer('xyz', xyz, persistent=False)

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes).cuda()
        self.total_correct = torch.zeros(self.num_classes).cuda()
        self.total_positive = torch.zeros(self.num_classes).cuda()

    def _after_step(self, outputs, targets, occ3d=False):
        if occ3d:
            self._after_step_occ3d(outputs, targets)
            return
        seen = targets.shape[0]
        correct = outputs[targets.transpose(0, 1).tolist()].sum()
        positive = outputs.sum()

        self.total_seen[0] += seen
        self.total_correct[0] += correct
        self.total_positive[0] += positive
    
    def _after_step_occ3d(self, outputs, targets):
        mask = torch.from_numpy(targets['mask_camera']).cuda()
        label = torch.from_numpy(targets['semantics']).cuda()
        label = label != 17
        if self.use_mask:
            label = label[mask]
        label = torch.nonzero(label)

        if self.use_mask:
            outputs = outputs[mask]
        seen = label.shape[0]
        correct = outputs[label.transpose(0, 1).tolist()].sum()
        positive = outputs.sum()

        self.total_seen[0] += seen
        self.total_correct[0] += correct
        self.total_positive[0] += positive
        
    def _after_epoch(self):
        dist.all_reduce(self.total_seen)
        dist.all_reduce(self.total_correct)
        dist.all_reduce(self.total_positive)

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou.item())

        miou = np.mean(ious)
        # logger = get_root_logger()
        logger.info(f'Validation per class iou:')
        for iou, label_str in zip(ious, self.label_str):
            logger.info('%s : %.2f%%' % (label_str, iou * 100))
        logger.info(f'Final iou: {miou * 100}')
        
        return miou * 100


def cal_depth_metric(depth_pred, depth_gt):
    depth_pred = torch.clamp(depth_pred, 1e-3, 80)

    thresh = torch.maximum((depth_gt / depth_pred), (depth_pred / depth_gt))
    a1 = (thresh < 1.25).to(torch.float)
    a2 = (thresh < 1.25 ** 2).to(torch.float)
    a3 = (thresh < 1.25 ** 3).to(torch.float)
    a1 = a1.mean()
    a2 = a2.mean()
    a3 = a3.mean()

    rmse = (depth_gt - depth_pred) ** 2
    rmse = rmse.mean() ** .5

    rmse_log = (torch.log(depth_gt) - torch.log(depth_pred)) ** 2
    rmse_log = rmse_log.mean() ** .5

    abs_rel = torch.abs(depth_gt - depth_pred) / depth_gt
    abs_rel = abs_rel.mean()

    sq_rel = ((depth_gt - depth_pred) ** 2) / depth_gt
    sq_rel = sq_rel.mean()

    metrics_dict = {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "a1": a1,
        "a2": a2,
        "a3": a3
    }
    return metrics_dict


class DepthMetric(nn.Module):

    def __init__(self, camera_names=['front'], eval_types=['raw', 'median']):
        super().__init__()
        self.num_cams = len(camera_names)
        self.camera_names = camera_names
        self.num_types = len(eval_types)
        self.eval_types = eval_types
        self.register_buffer('abs_rel', torch.zeros(self.num_types, self.num_cams))
        self.register_buffer('sq_rel', torch.zeros(self.num_types, self.num_cams))
        self.register_buffer('rmse', torch.zeros(self.num_types, self.num_cams))
        self.register_buffer('rmse_log', torch.zeros(self.num_types, self.num_cams))
        self.register_buffer('a1', torch.zeros(self.num_types, self.num_cams))
        self.register_buffer('a2', torch.zeros(self.num_types, self.num_cams))
        self.register_buffer('a3', torch.zeros(self.num_types, self.num_cams))
        self.register_buffer('count', torch.zeros(1))
        self.register_buffer('scaling', torch.zeros(self.num_types, self.num_cams))

    def _reset(self):
        self.abs_rel.zero_()
        self.sq_rel.zero_()
        self.rmse.zero_()
        self.rmse_log.zero_()
        self.a1.zero_()
        self.a2.zero_()
        self.a3.zero_()
        self.count.zero_()
        self.scaling.zero_()

    def _after_step(self, depth_loc, depth_gt, depth_mask, depth_pred):
        # depth_loc: N, n, 2
        # depth_gt: N, n
        # depth_mask: N, n
        # depth_pred: N, h, w
        num_cams, num_points = depth_gt.shape
        depth_pred = F.grid_sample(
            depth_pred.unsqueeze(1),
            depth_loc.unsqueeze(1) * 2 - 1,
            mode='bilinear',
            padding_mode='border',
            align_corners=True) # N, 1, 1, n
        depth_pred = depth_pred.reshape(num_cams, num_points)
        
        for cam, (depth_gt_i, depth_pred_i, depth_mask_i) in enumerate(
                zip(depth_gt, depth_pred, depth_mask)):
            depth_gt_i_mask = depth_gt_i[depth_mask_i]
            depth_pred_i_mask = depth_pred_i[depth_mask_i]
            for type_idx, type in enumerate(self.eval_types):
                if type == 'raw':
                    depth_pred_i_cal = depth_pred_i_mask
                    self.scaling[type_idx, cam] += 1.0
                elif type == 'median':
                    scaling = torch.median(depth_gt_i_mask) / \
                        torch.median(depth_pred_i_mask)
                    depth_pred_i_cal = scaling * depth_pred_i_mask
                    self.scaling[type_idx, cam] += scaling
                else:
                    raise NotImplementedError
            
                metrics = cal_depth_metric(depth_pred_i_cal, depth_gt_i_mask)
                self.abs_rel[type_idx, cam] += metrics['abs_rel']
                self.sq_rel[type_idx, cam] += metrics['sq_rel']
                self.rmse[type_idx, cam] += metrics['rmse']
                self.rmse_log[type_idx, cam] += metrics['rmse_log']
                self.a1[type_idx, cam] += metrics['a1']
                self.a2[type_idx, cam] += metrics['a2']
                self.a3[type_idx, cam] += metrics['a3']
        self.count += 1

    def _after_epoch(self):
        dist.barrier()
        dist.all_reduce(self.count)
        dist.all_reduce(self.abs_rel)
        dist.all_reduce(self.sq_rel)
        dist.all_reduce(self.rmse)
        dist.all_reduce(self.rmse_log)
        dist.all_reduce(self.a1)
        dist.all_reduce(self.a2)
        dist.all_reduce(self.a3)
        dist.all_reduce(self.scaling)
        dist.barrier()
        
        abs_rel = self.abs_rel / self.count
        sq_rel = self.sq_rel / self.count
        rmse = self.rmse / self.count
        rmse_log = self.rmse_log / self.count
        a1 = self.a1 / self.count
        a2 = self.a2 / self.count
        a3 = self.a3 / self.count
        scaling = self.scaling / self.count

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f'Averaging over {self.count.item()} samples.')
            for type_idx, type in enumerate(self.eval_types):
                logger.info("{} evaluation:".format(type))
                logger.info(("{:>12} | " * 9).format("cam_name", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "scale"))
                for cam, cam_name in enumerate(self.camera_names):
                    logger.info((f"{cam_name:>12} | " + "&{: 12.3f}  " * 8).format(
                        abs_rel[type_idx, cam],
                        sq_rel[type_idx, cam],
                        rmse[type_idx, cam],
                        rmse_log[type_idx, cam],
                        a1[type_idx, cam],
                        a2[type_idx, cam],
                        a3[type_idx, cam],
                        scaling[type_idx, cam]) + "\\\\")
                logger.info(("{:>12} | " + "&{: 12.3f}  " * 8).format(
                        "All",
                        abs_rel[type_idx].mean(),
                        sq_rel[type_idx].mean(),
                        rmse[type_idx].mean(),
                        rmse_log[type_idx].mean(),
                        a1[type_idx].mean(),
                        a2[type_idx].mean(),
                        a3[type_idx].mean(),
                        scaling[type_idx].mean()) + "\\\\")

       
def compute_depth_errors(gt, pred, min_depth=1e-3, max_depth=80):
    """Computation of error metrics between predicted and ground truth depths
    """
    pred[pred < min_depth] = min_depth
    pred[pred > max_depth] = max_depth

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_depth_errors_torch(gt, pred, min_depth=1e-3, max_depth=80):
    """Computation of error metrics between predicted and ground truth depths
    """
    pred[pred < min_depth] = min_depth
    pred[pred > max_depth] = max_depth

    thresh = torch.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
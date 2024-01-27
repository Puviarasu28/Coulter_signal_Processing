import torch
import numpy as np
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class custom_loss(nn.Module):
    ''' Custom Yolov2 Loss Class'''

    def __init__(self):
        super(custom_loss, self).__init__()
        self.BOX = 5
        self.GRID_H = 13
        self.GRID_W = 13
        self.CLASSES = 80
        self.COORD_FACTOR = 1.0
        self.NO_OBJECT_FACTOR = 1.0
        self.OBJECT_FACTOR = 5.0
        self.CLASS_FACTOR = 1.0
        self.CLASS_WEIGHTS = torch.ones(self.CLASSES, dtype=torch.float32)
        self.ANCHORS = torch.tensor(
            [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])

    def forward(self, y_true, y_pred, true_boxes):
        mask_shape = torch.Tensor.size(y_true)[:4]
        BATCH_SIZE = torch.Tensor.size(y_true)[0]
        self.ANCHORS = self.ANCHORS.to(DEVICE)

        cell_w = torch.reshape(torch.tile(torch.range(0, self.GRID_H - 1, dtype=torch.float32), (self.GRID_W,)),
                               (1, self.GRID_H, self.GRID_W, 1, 1))
        cell_h = torch.transpose(cell_w, 1, 2)

        cell_grid = torch.tile(torch.cat((cell_w, cell_h), -1), (BATCH_SIZE, 1, 1, self.BOX, 1))
        cell_grid = cell_grid.to(DEVICE)

        # Adjust prediction
        # adjust x & y
        pred_box_xy = torch.sigmoid(y_pred[..., :2]) + cell_grid
        # adjust w & h
        pred_box_wh = torch.exp(y_pred[..., 2:4]) * torch.reshape(self.ANCHORS, [1, 1, 1, self.BOX, 2])
        # adjust confidence
        pred_box_conf = torch.sigmoid(y_pred[..., 4])
        # adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        # Adjust ground truth
        # adjust x and y
        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell
        # adjust w and h
        true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_xy_mins = true_box_xy - true_wh_half
        true_xy_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_xy_mins = pred_box_xy - pred_wh_half
        pred_xy_maxes = pred_box_xy + pred_wh_half

        intersect_mins = torch.maximum(pred_xy_mins, true_xy_mins)
        intersect_maxes = torch.minimum(pred_xy_maxes, true_xy_maxes)
        intersect_wh = torch.maximum(intersect_maxes - intersect_mins, torch.tensor(0.))

        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = torch.true_divide(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = torch.argmax(y_true[..., 5:], -1)

        """
        Finding the masks
        """
        ### coordinate mask: position of the ground truth boxes
        coord_mask = torch.unsqueeze(y_true[..., 4], dim=-1) * self.COORD_FACTOR

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box less than 0.6
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_xy_mins = true_xy - true_wh_half
        true_xy_maxes = true_xy + true_wh_half

        pred_xy = torch.unsqueeze(pred_box_xy, 4)
        pred_wh = torch.unsqueeze(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_xy_mins = pred_xy - pred_wh_half
        pred_xy_maxes = pred_xy + pred_wh_half

        intersect_mins = torch.maximum(pred_xy_mins, true_xy_mins)
        intersect_maxes = torch.minimum(pred_xy_maxes, true_xy_maxes)
        intersect_xy = torch.maximum(intersect_maxes - intersect_mins, torch.tensor(0.))
        intersect_areas = intersect_xy[..., 1] * intersect_xy[..., 0]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = torch.true_divide(intersect_areas, union_areas)
        best_ious = torch.max(iou_scores, 4)[0]
        conf_mask = torch.Tensor.float(best_ious < torch.tensor(0.6)) * (1 - y_true[..., 4]) * self.NO_OBJECT_FACTOR

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + true_box_conf * self.OBJECT_FACTOR
        ### class mask: position of the ground truth boxes
        class_mask = y_true[..., 4] * self.CLASS_FACTOR

        nb_coord_box = torch.sum(torch.Tensor.float(coord_mask > 0.0))
        nb_conf_box = torch.sum(torch.Tensor.float(conf_mask > 0.0))
        nb_class_box = torch.sum(torch.Tensor.float(class_mask > 0.0))

        loss_xy = torch.sum(torch.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = torch.sum(torch.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf = torch.sum(torch.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.

        loss_class = torch.nn.functional.cross_entropy(pred_box_class.reshape([-1, self.CLASSES]),
                                                       true_box_class.reshape(-1, ), reduction='none')
        loss_class = loss_class.reshape(mask_shape)
        loss_class = torch.sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = loss_xy + loss_wh + loss_conf + loss_class

        return loss
import copy
import time

import cv2
import numpy as np
import pyclipper
from onnxruntime import InferenceSession
from shapely.geometry import Polygon


class SimpleDataset:
    def __call__(self, img: np.ndarray, bboxes: np.ndarray):
        """
        bboxes: (N, 4, 2)
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_instance = np.zeros(img.shape[:2], dtype='uint8')
        for i in range(len(bboxes)):
            cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_text = gt_text[None, None, ...].astype(np.float32)

        canvas, shrink_mask, mask_ori = self.get_seg_map(img, bboxes)
        soft_mask = canvas + mask_ori
        index_mask = np.where(soft_mask > 1)
        soft_mask[index_mask] = 1
        soft_mask = soft_mask[None, None, ...].astype(np.float32)

        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        img = img[None, ...]
        structure_im = copy.deepcopy(img)
        return img, structure_im, gt_text, soft_mask

    def draw_border_map(self, polygon, canvas, mask_ori, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        ### shrink box ###
        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
                   (1 - np.power(0.95, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(-distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)
        ### shrink box ###

        cv2.fillPoly(mask_ori, [polygon.astype(np.int32)], 1.0)

        polygon = padded_polygon
        polygon_shape = Polygon(padded_polygon)
        distance = polygon_shape.area * \
                   (1 - np.power(0.4, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            # import pdb;pdb.set_trace()
            absolute_distance = self.coumpute_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin:ymax_valid - ymax + height,
                xmin_valid - xmin:xmax_valid - xmax + width
                ],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1]
        )

    @staticmethod
    def coumpute_distance(xs, ys, point_1, point_2):
        """
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        """
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / \
                (2 * np.sqrt(square_distance_1 * square_distance_2) + 1e-50)
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *
                         square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]
        # extend_line(point_1, point_2, result)
        return result

    def get_seg_map(self, img, label):
        canvas = np.zeros(img.shape[:2], dtype=np.float32)
        mask = np.zeros(img.shape[:2], dtype=np.float32)
        mask_ori = np.zeros(img.shape[:2], dtype=np.float32)
        polygons = label

        for i in range(len(polygons)):
            self.draw_border_map(polygons[i], canvas, mask_ori, mask=mask)
        return canvas, mask, mask_ori


class CTRNetInfer:
    def __init__(self, model_path) -> None:
        self.session = InferenceSession(model_path,
                                        providers=['CPUExecutionProvider'])
        self.dataset = SimpleDataset()
        self.input_shape = (512, 512)

    def __call__(self, ori_img, bboxes):
        ori_img_shape = ori_img.shape[:2]

        # resize img 到512x512
        resize_img = cv2.resize(
            ori_img, self.input_shape, interpolation=cv2.INTER_LINEAR
        )
        resize_bboxes = self.get_resized_points(
            bboxes, ori_img_shape, self.input_shape
        )
        img, structure_im, gt_text, soft_mask = self.dataset(
            resize_img, resize_bboxes
        )
        input_dict = {
            'input': img,
            'gt_text': gt_text,
            'soft_mask': soft_mask,
            'structure_im': structure_im
        }
        prediction = self.session.run(None, input_dict)[3]

        withMask_prediction = prediction * soft_mask + img * (1 - soft_mask)
        withMask_prediction = np.transpose(withMask_prediction, (0, 2, 3, 1)) * 255
        withMask_prediction = withMask_prediction.squeeze().astype(np.uint8)
        withMask_prediction = cv2.cvtColor(withMask_prediction,
                                           cv2.COLOR_BGR2RGB)
        ori_pred = cv2.resize(withMask_prediction, ori_img_shape[::-1],
                              interpolation=cv2.INTER_LINEAR)
        return ori_pred

    @staticmethod
    def get_resized_points(cur_points, cur_shape, new_shape):
        cur_points = np.array(cur_points)

        ratio_x = cur_shape[0] / new_shape[0]
        ratio_y = cur_shape[1] / new_shape[1]
        cur_points[:, :, 0] = cur_points[:, :, 0] / ratio_x
        cur_points[:, :, 1] = cur_points[:, :, 1] / ratio_y
        return cur_points.astype(np.int64)

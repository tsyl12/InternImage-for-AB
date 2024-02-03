import os.path
from typing import Tuple

import cv2
import numpy as np
import yaml
from mmdet.apis import inference_detector

CONFIG_PATH = './configs/utils/focal_patch_batch.yml'


def focal_patch_batch(model, image_path):
    # perform focal patch batch on the given image

    # read the cfg
    with open(CONFIG_PATH) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    scales = cfg['scales']
    input_h = cfg['patch_shape'][1]
    input_w = cfg['patch_shape'][0]
    min_overlap_h = cfg['min_overlap'][1]
    min_overlap_w = cfg['min_overlap'][0]
    remove_detections_from_edges = cfg['remove_detections_from_edges']
    remove_detections_from_edges_proximity = cfg['remove_detections_from_edges_proximity']
    postprocess_score_threshold = cfg['postprocess_score_threshold']
    nms_iou_threshold = cfg['nms_iou_threshold']
    nms_ioa_threshold = cfg['nms_ioa_threshold']

    # read the image
    image = cv2.imread(image_path)

    # initialize outputs
    total_bboxes = None
    total_masks = None

    # iterate over scales
    for scale_ind, scale in enumerate(scales):

        # scale the image
        scaled_image = cv2.resize(
            src=image.copy(),
            dsize=(int(image.shape[1] * scale), int(image.shape[0] * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
        image_h = scaled_image.shape[0]
        image_w = scaled_image.shape[1]

        # calculate the location of patches
        indexes_h, indexes_w = calculate_patches_location(
            image_h=image_h,
            image_w=image_w,
            input_h=input_h,
            input_w=input_w,
            min_overlap_h=min_overlap_h,
            min_overlap_w=min_overlap_w,
        )

        # iterate over all patches in the scaled image
        print(f'Processing image {os.path.basename(image_path)} on scale {scale} (index {scale_ind + 1}/{len(scales)}) with {len(indexes_h) * len(indexes_w)} patches ...')
        for index_h in indexes_h:
            for index_w in indexes_w:

                # crop the patch
                patch = scaled_image[index_h: index_h + input_h, index_w: index_w + input_w]

                # patch inference
                patch_result = inference_detector(model, patch)

                # filter cutoff detections on patch edges which are not image edges and detections bellow score threshold
                patch_result = filter_patch_edges_and_score(
                    results=patch_result,
                    input_h=input_h,
                    input_w=input_w,
                    filter_top=index_h != indexes_h[0],
                    filter_bottom=index_h != indexes_h[-1],
                    filter_left=index_w != indexes_w[0],
                    filter_right=index_w != indexes_w[-1],
                    remove_detections_from_edges=remove_detections_from_edges,
                    remove_detections_from_edges_proximity=remove_detections_from_edges_proximity,
                    postprocess_score_threshold=postprocess_score_threshold,
                )

                # compensate for crop
                if input_h != 0 or input_w != 0:
                    compensate_bboxes_for_cropped_input(bboxes=patch_result[0], shift_h=index_h, shift_w=index_w)
                    compensate_masks_for_cropped_input(masks=patch_result[1], shift_h=index_h, shift_w=index_w, image_h=image_h, image_w=image_w)

                # compensate for scale
                if scale != 1.0:
                    compensate_bboxes_for_scaled_input(bboxes=patch_result[0], scale=scale)
                    compensate_masks_for_scaled_input(masks=patch_result[1], scale=scale)

                # append bboxes
                if total_bboxes is None:
                    total_bboxes = patch_result[0]
                else:
                    append_bboxes(total_bboxes=total_bboxes, bboxes=patch_result[0])

                # append masks
                if total_masks is None:
                    total_masks = patch_result[1]
                else:
                    append_masks(total_masks=total_masks, masks=patch_result[1])

    # apply mask-based NMS
    total_bboxes, total_masks = nms(
        results=(total_bboxes, total_masks),
        iou_threshold=nms_iou_threshold,
        ioa_threshold=nms_ioa_threshold,
    )

    return total_bboxes, total_masks


def calculate_patches_location(image_h: int, image_w: int, input_h: int, input_w: int, min_overlap_h: int, min_overlap_w: int) -> Tuple[list, list]:
    # Image to input relation
    assert image_h > input_h and image_w > input_w, 'The image is smaller than the input. Consider adjust patch_shape or scales in the focal_patch_batch.yml.'

    # Calculating the num of patches needed for a given min overlap
    grid_h = np.ceil((image_h - min_overlap_h) / (input_h - min_overlap_h))
    grid_w = np.ceil((image_w - min_overlap_w) / (input_w - min_overlap_w))

    # Calculating the actual overlap that will be used to fit the patches inside the image
    overlap_h = np.floor((grid_h * input_h - image_h) / (grid_h - 1)) if grid_h > 1 else 0
    overlap_w = np.floor((grid_w * input_w - image_w) / (grid_w - 1)) if grid_w > 1 else 0

    # Calculate patches strides
    stride_h = int(input_h - overlap_h)
    stride_w = int(input_w - overlap_w)

    # Calculate indexes for patches s.t. patch_i = image[indexes_h[i]: indexes_h[i] + input_h, indexes_w[i]: indexes_w[i] + input_w]
    indexes_h = np.arange(0, image_h - input_h + stride_h, stride_h)
    indexes_w = np.arange(0, image_w - input_w + stride_w, stride_w)

    # Take care of the leftovers
    leftover_h = image_h - (indexes_h[-1] + input_h)
    leftover_w = image_w - (indexes_w[-1] + input_w)
    assert overlap_h + leftover_h >= min_overlap_h
    assert overlap_w + leftover_w >= min_overlap_w
    indexes_h[-1] += leftover_h
    indexes_w[-1] += leftover_w

    # Check the edges
    assert indexes_h[-1] + input_h == image_h
    assert indexes_w[-1] + input_w == image_w

    return indexes_h, indexes_w


def filter_patch_edges_and_score(
        results: Tuple[list, list],
        input_h: int,
        input_w: int,
        filter_top: bool,
        filter_bottom: bool,
        filter_left: bool,
        filter_right: bool,
        remove_detections_from_edges: bool,
        remove_detections_from_edges_proximity: float,
        postprocess_score_threshold: float,
) -> Tuple[list, list]:
    # filter cutoff detections on patch edges which are not image edges for effective NMS
    filtered_bboxes = []
    filtered_masks = []
    num_classes = len(results[0])
    for class_ind in range(num_classes):
        class_bboxes = results[0][class_ind]
        num_class_detections = class_bboxes.shape[0]
        if num_class_detections == 0:
            # no detection in this class
            filtered_bboxes.append(results[0][class_ind])
            filtered_masks.append(results[1][class_ind])
        else:
            # detections that need to be checked
            filtered_class_bboxes = np.zeros([0, 5])
            filtered_class_masks = []
            for det_ind in range(num_class_detections):
                # check if the bbox is on the patch edge but not on the image edge
                filter_on_top = class_bboxes[det_ind, 1] <= remove_detections_from_edges_proximity and filter_top
                filter_on_bottom = class_bboxes[det_ind, 3] >= input_h - 1 - remove_detections_from_edges_proximity and filter_bottom
                filter_on_left = class_bboxes[det_ind, 0] <= remove_detections_from_edges_proximity and filter_left
                filter_on_right = class_bboxes[det_ind, 2] >= input_w - 1 - remove_detections_from_edges_proximity and filter_right
                # check if the score is over the threshold (improves NMS times)
                over_score_threshold = class_bboxes[det_ind, 4] >= postprocess_score_threshold
                # if bbox is not on one of the edges and over score threshold
                if (remove_detections_from_edges and not any([filter_on_top, filter_on_bottom, filter_on_left, filter_on_right])) and over_score_threshold:
                    filtered_class_bboxes = np.row_stack((filtered_class_bboxes, class_bboxes[det_ind, :]))
                    filtered_class_masks.append(results[1][class_ind][det_ind])
            # append all filtered detections
            filtered_bboxes.append(filtered_class_bboxes)
            filtered_masks.append(filtered_class_masks)

    assert len(filtered_bboxes) == num_classes and len(filtered_masks) == num_classes
    return filtered_bboxes, filtered_masks


def compensate_bboxes_for_cropped_input(bboxes: list, shift_h: int, shift_w: int) -> None:
    # shift the bboxes to compensate for the crop. The bbox format is (x1, y1, x2, y2) where x is the horizontal axis.
    for class_bboxes in bboxes:
        if len(class_bboxes) > 0:
            class_bboxes[:, 0] += shift_w
            class_bboxes[:, 2] += shift_w
            class_bboxes[:, 1] += shift_h
            class_bboxes[:, 3] += shift_h


def compensate_bboxes_for_scaled_input(bboxes: list, scale: float) -> None:
    # scale the bboxes to compensate for scaled input
    for class_bboxes in bboxes:
        if len(class_bboxes) > 0:
            class_bboxes[:, 0] /= scale
            class_bboxes[:, 2] /= scale
            class_bboxes[:, 1] /= scale
            class_bboxes[:, 3] /= scale


def compensate_masks_for_cropped_input(masks: list, shift_h: int, shift_w: int, image_h: int, image_w: int) -> None:
    # shift masks to compensate for the crop
    for class_idx, class_masks in enumerate(masks):
        if len(class_masks) > 0:
            for mask_idx, mask in enumerate(masks[class_idx]):
                left = shift_w
                top = shift_h
                right = image_w - mask.shape[1] - left
                bottom = image_h - mask.shape[0] - top
                masks[class_idx][mask_idx] = np.pad(array=mask, pad_width=((top, bottom), (left, right)))


def compensate_masks_for_scaled_input(masks: list, scale: float) -> None:
    # shift masks to compensate for scaled input
    for class_idx, class_masks in enumerate(masks):
        if len(class_masks) > 0:
            for mask_idx, mask in enumerate(masks[class_idx]):
                float_mask = mask * 1.0
                scaled_float_mask = cv2.resize(
                    src=float_mask,
                    dsize=(int(float_mask.shape[1] * 1/scale), int(float_mask.shape[0] * 1/scale)),
                    interpolation=cv2.INTER_NEAREST,
                )
                scaled_bool_mask = scaled_float_mask != 0
                masks[class_idx][mask_idx] = scaled_bool_mask


def append_bboxes(total_bboxes: list, bboxes: list) -> None:
    # append patch bboxes to total_bboxes
    for class_idx in range(len(total_bboxes)):
        total_bboxes[class_idx] = np.row_stack((total_bboxes[class_idx], bboxes[class_idx]))


def append_masks(total_masks: list, masks: list) -> None:
    # append patch masks to total_masks
    for class_idx in range(len(total_masks)):
        total_masks[class_idx] += masks[class_idx]


def nms(results:Tuple[list, list], iou_threshold: float, ioa_threshold: float) -> Tuple[list, list]:
    # segmentation-based NMS
    print('Applying NMS ...')
    filtered_bboxes = []
    filtered_masks = []
    num_classes = len(results[0])
    for class_ind in range(num_classes):
        class_bboxes = results[0][class_ind]
        num_class_detections = class_bboxes.shape[0]
        if num_class_detections == 0:
            # no detection in this class
            filtered_bboxes.append(results[0][class_ind])
            filtered_masks.append(results[1][class_ind])
        else:
            # divide masks into groups based on iou between masks
            filtered_class_bboxes = np.zeros([0, 5])
            filtered_class_masks = []
            mask_index_groups = []
            for mask_ind1 in range(num_class_detections):

                # check if this mask was already assigned to a group
                if any([mask_ind1 in group for group in mask_index_groups]):
                    continue

                mask1 = results[1][class_ind][mask_ind1]
                # look for other masks with big enough iou while taking into account already computed IOUs
                mask_index_group = [mask_ind1]
                for mask_ind2 in np.arange(mask_ind1 + 1, num_class_detections):

                    # check if this mask was already assigned to a group
                    if any([mask_ind2 in group for group in mask_index_groups]):
                        continue

                    # calculate iou
                    mask2 = results[1][class_ind][mask_ind2]
                    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
                    union = np.count_nonzero(np.logical_or(mask1, mask2))
                    iou = intersection / union if union > 0 else 0
                    if iou >= iou_threshold:
                        mask_index_group.append(mask_ind2)
                mask_index_groups.append(mask_index_group)

            # pick one detection from each group
            best_masks = []
            for group in mask_index_groups:
                confidences = []
                for mask_ind in group:
                    score = results[0][class_ind][mask_ind, 4]
                    area = np.count_nonzero(results[1][class_ind][mask_ind])
                    confidences.append(score * area)    # imo the best indicator for a good detection
                best_masks.append(group[np.argmax(confidences)])

            # remove masks that are contained within other masks (above the IOA threshold)
            redundant_masks = []
            for mask_ind1 in best_masks:
                mask1 = results[1][class_ind][mask_ind1]
                area = np.count_nonzero(mask1)
                for mask_ind2 in best_masks:
                    if mask_ind2 == mask_ind1:
                        continue
                    mask2 = results[1][class_ind][mask_ind2]
                    # check whether mask1 is contained within mask2 by IOA
                    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
                    ioa = intersection / area if area > 0 else 0
                    if ioa >= ioa_threshold:
                        # mask1 is redundant
                        redundant_masks.append(mask_ind1)
            best_masks = list(set(best_masks) - set(redundant_masks))

            # append the best masks for this class
            for best_mask in best_masks:
                filtered_class_bboxes = np.row_stack((filtered_class_bboxes, class_bboxes[best_mask, :]))
                filtered_class_masks.append(results[1][class_ind][best_mask])

            # append to general filtered detections
            filtered_bboxes.append(filtered_class_bboxes)
            filtered_masks.append(filtered_class_masks)

    assert len(filtered_bboxes) == num_classes and len(filtered_masks) == num_classes
    print('Done applying NMS.')
    return filtered_bboxes, filtered_masks

# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser, Namespace

import mmcv
from mmdet.apis import init_detector, inference_detector

import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
from utils.data import prepare_files
from utils.inference import focal_patch_batch


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # new arguments
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='the name of the model that will be used for inference (used to select a config and a checkpoint)'
    )
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        default=None,
        help='a path to a checkpoint for the model (default - based on the model config)'
    )
    parser.add_argument(
        '--model_inputs',
        type=str,
        default="model_inputs",
        help='a path to an image or a directory with images for inference (default - use AB images)',
    )
    parser.add_argument(
        '--model_outputs',
        type=str,
        default="model_outputs",
        help='out dir'
    )
    parser.add_argument(
        '--allow_download',
        action='store_true',
        help='whether to allow the downloading of the inference images and model checkpoint',
    )
    parser.add_argument(
        '--focal_patch_batch',
        action='store_true',
        help='whether to perform focal patch batch inference for reduced scale sensitivity',
    )

    # previous arguments
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Device used for inference',
    )
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization',
    )
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='bbox score threshold',
    )
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference',
    )
    args = parser.parse_args()
    return args


def main(args):
    # prepare inference images, inference config, and model checkpoint
    files_paths = prepare_files(
        model_name=args.model_name,
        model_checkpoint=args.model_checkpoint,
        model_inputs=args.model_inputs,
        allow_download=args.allow_download,
    )

    # build the model from a config file and a checkpoint file
    model = init_detector(files_paths['config_path'], files_paths['checkpoint_path'], device=args.device)

    # make outputs directory and iterate over the inputs
    mmcv.mkdir_or_exist(args.model_outputs)

    # select the inference method
    inference_method = focal_patch_batch if args.focal_patch_batch else inference_detector

    # iterate over inputs
    for image_path in files_paths['model_input_paths']:
        out_file_path = os.path.join(args.model_outputs, os.path.basename(image_path))
        if os.path.exists(out_file_path) and input(f'File {out_file_path} already exists. Overwrite? (y, [n]): ') != 'y':
            continue

        # inference
        result = inference_method(model, image_path)

        # show the results
        print('Visualizing ...')
        model.show_result(
            image_path,
            result,
            score_thr=args.score_thr,
            show=False,
            bbox_color=args.palette,
            text_color=(200, 200, 200),
            mask_color=args.palette,
            out_file=out_file_path
        )
        print(f"The result is saved at {out_file_path}")
    print("Inference complete")


if __name__ == '__main__':
    inference_args = parse_args()
    main(inference_args)

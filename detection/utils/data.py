import os
from typing import Tuple, List

import wget

SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png']
DEFAULT_IMAGE_DIR = 'model_inputs'
DEFAULT_CHECKPOINTS_DIR = 'checkpoints'
IMAGE_URLS = {
    'image001.png': 'https://drive.google.com/uc?id=1Bus7Wl49T-4zAsg0tmGSt-r-m4r2zPTi',
    'image002.png': 'https://drive.google.com/uc?id=1H4VeUlgME8exg48nckqx1KOWpOAmUp5v',
}


def prepare_files(model_name: str, model_checkpoint: str, model_inputs: str, allow_download: bool) -> dict:
    # construct paths and download files
    config_path, checkpoint_path = prepare_model_files(model_name, model_checkpoint, allow_download)
    model_input_paths = prepare_input_files(model_inputs, allow_download)
    return {'config_path': config_path, 'checkpoint_path': checkpoint_path, 'model_input_paths': model_input_paths}


def prepare_model_files(model_name: str, model_checkpoint: str, allow_download: bool) -> Tuple[str, str]:
    # select a config file according to the model name
    config_path = f'configs/coco/{model_name}.py'
    assert os.path.exists(config_path)

    # select a checkpoint
    if model_checkpoint is not None:
        checkpoint_path = model_checkpoint
    else:
        checkpoint_url = f'https://huggingface.co/OpenGVLab/InternImage/resolve/main/{model_name}.pth'

        # download the model checkpoint
        checkpoint_path = os.path.join(DEFAULT_CHECKPOINTS_DIR, os.path.basename(checkpoint_url))
        if not os.path.exists(DEFAULT_CHECKPOINTS_DIR):
            os.mkdir(DEFAULT_CHECKPOINTS_DIR)
        if not os.path.exists(checkpoint_path):
            if not allow_download:
                raise Exception('Checkpoint {} and download permission were not granted by args.allow_download')
            else:
                print(f'Checkpoint download: {checkpoint_path} ...')
                wget.download(checkpoint_url, checkpoint_path)
                print(f'Checkpoint download complete.')
    return config_path, checkpoint_path


def prepare_input_files(model_inputs: str, allow_download: bool) -> List[str]:
    # prepare a list of paths of images for inference

    # default - use AB 2 images
    if model_inputs == "model_inputs" and (not os.path.exists(model_inputs) or not all([img_name in os.listdir(DEFAULT_IMAGE_DIR) for img_name in IMAGE_URLS.keys()])):
        if not allow_download:
            raise Exception('Some images have to be downloaded but download permission were not granted by args.allow_download')
        else:
            if not os.path.exists(DEFAULT_IMAGE_DIR):
                os.mkdir(DEFAULT_IMAGE_DIR)
            download_images(DEFAULT_IMAGE_DIR)
            return get_images_paths_from_dir(DEFAULT_IMAGE_DIR)

    # use a given directory
    elif os.path.isdir(model_inputs):
        return get_images_paths_from_dir(model_inputs)

    # use a given image
    elif any([model_inputs.endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS]):
        return [model_inputs]

    # unknown input
    else:
        raise NotImplementedError(f'Unsupported model input {model_inputs}')


def get_images_paths_from_dir(dir_path: str) -> List[str]:
    # get a list of paths of suitable images within a given directory
    images_paths = [os.path.join(dir_path, img_name) for img_name in os.listdir(dir_path) if any([img_name.endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS])]
    assert len(images_paths) > 0, f'No suitable images found in {dir_path}'
    return images_paths


def download_images(dir_path: str) -> None:
    # download 2 AB images
    for img_name, img_url in IMAGE_URLS.items():
        img_path = os.path.join(dir_path, img_name)
        if not os.path.exists(img_path):
            print(f'Downloading image {img_path} ...')
            wget.download(img_url, img_path)
    assert len(os.listdir(dir_path)) == len(IMAGE_URLS)

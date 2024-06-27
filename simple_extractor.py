#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='', help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def modify_palette(original_palette, all_components, mask_components):
    modified_palette = [0] * len(original_palette)
    for component in mask_components:
        if component in all_components:
            index = all_components.index(component)
            # modified_palette[index*3:(index+1)*3] = original_palette[index*3:(index+1)*3]
            modified_palette[index*3:(index+1)*3] = [255, 255, 255] # all white
    return modified_palette


def extraction(dataset: str, mask_components: list, model_path: str, input_dir: str, output_dir: str, gpu: str, logits: bool):
    if gpu != 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    num_classes = dataset_settings[dataset]['num_classes']
    input_size = dataset_settings[dataset]['input_size']
    label = dataset_settings[dataset]['label']
    print(f"Evaluating total class number {num_classes} with {label}")
    
    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
    state_dict = torch.load(model_path)['state_dict']
    new_state_dict = {k[7:]: v for k, v in state_dict.items()}  # remove `module.`
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    
    dataset = SimpleFolderDataset(root=input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    original_palette = get_palette(num_classes)
    modified_palette = modify_palette(original_palette, label, mask_components)
    
   
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]
            
            output = model(image.cuda())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze().permute(1, 2, 0)  # CHW -> HWC
            
            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            parsing_result_path = os.path.join(output_dir, f"{img_name[:-4]}.png")
            output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            output_img.putpalette(modified_palette)
            output_img.save(parsing_result_path)
            
            if logits:
            
                logits_result_path = os.path.join(output_dir, f"{img_name[:-4]}.npy")
                np.save(logits_result_path, logits_result)
    


def main():
    args = get_arguments()
    extraction(
        dataset=args.dataset,
        mask_components=args.mask_components,
        model_path=args.model_restore, 
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gpu=args.gpu,
        logits=args.logits
    )
    
if __name__ == '__main__':
    main()

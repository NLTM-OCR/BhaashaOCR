import os
import argparse
from tqdm import tqdm
import time
from collections import OrderedDict
from os.path import basename, join, splitext
from typing import Tuple
from PIL import Image

import torch

os.environ['USE_TORCH'] = '1'
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def logtime(t: float, msg:  str) -> None:
    print(f'[{int(time.time() - t)}s]\t {msg}')

def convert_geometry_to_bbox(
    geometry: Tuple[Tuple[float, float], Tuple[float, float]],
    dim: Tuple[int, int],
    padding: int = 0,
    line: int = 0
):
    x1 = int(geometry[0][0] * dim[1])
    y1 = int(geometry[0][1] * dim[0])
    x2 = int(geometry[1][0] * dim[1])
    y2 = int(geometry[1][1] * dim[0])
    return [x1-padding, y1-padding, x2-x1+padding*2, y2-y1+padding*2, line]

def crop_words(image_path, layout_file, word_folder):
    with open(layout_file, 'r') as f:
        a = f.read().strip().split('\n')
    a = [list(map(int, i.strip(' ,').split(','))) for i in a]
    a = [i for i in a if len(i) == 5]
    img = Image.open(image_path).convert('RGB')
    for idx,i in enumerate(a):
        img.crop((
            i[0], i[1],
            i[0]+i[2], i[1]+i[3]
        )).save(join(word_folder, f'{idx}.jpg'))

def process(image_path, output, PREDICTOR_V2):
    try:
        doc = DocumentFile.from_images([image_path])
        a = PREDICTOR_V2(doc)

        page = a.pages[0]
        dim = page.dimensions
        lines = []
        for i in page.blocks:
            lines += i.lines
        regions = []
        for i, line in enumerate(lines):
            for word in line.words:
                regions.append(
                    ','.join(list(map(str, convert_geometry_to_bbox(word.geometry, dim, padding=5, line=i+1)))),
                )
        outfile = join(output, 'layout.txt')
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write('\n'.join(regions))
        word_folder = join(output, 'words')
        if not os.path.exists(word_folder):
            os.makedirs(word_folder)
        crop_words(image_path, outfile, word_folder)
        return word_folder
    except Exception as e:
        print(f"Error processing file {image_path}: {e}")

def main(image_path, pretrained, output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    PREDICTOR_V2 = ocr_predictor(pretrained=True).to(device)
    if os.path.exists(pretrained):
        state_dict = torch.load(pretrained)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        PREDICTOR_V2.det_predictor.model.load_state_dict(new_state_dict)

    return process(image_path, output, PREDICTOR_V2)
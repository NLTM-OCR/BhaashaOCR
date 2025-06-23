from .openseg import perform_openseg
import os
from os.path import join
from PIL import Image
from .combine import merge_layouts
from .v4x import perform_v4x

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

def convert_to_Region(regions: list[str]) -> list[dict]:
    ret = []
    for order, i in enumerate(regions):
        x, y, w, h, line = map(int, i.split(','))
        ret.append({
            'bounding_box': {
                'x': x,
                'y': y,
                'w': w,
                'h': h
            },
            'line': line,
            'order': order+1,
        })
    return ret

def layout_main(image_path, pretrained, output):
    openseg_regions = convert_to_Region(perform_openseg(image_path, pretrained))
    v4x_regions = convert_to_Region(perform_v4x(image_path, pretrained))
    regions = merge_layouts(openseg_regions, v4x_regions)

    outfile = join(output, 'layout.txt')
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write('\n'.join(regions))
    word_folder = join(output, 'words')
    if not os.path.exists(word_folder):
        os.makedirs(word_folder)
    crop_words(image_path, outfile, word_folder)
    return word_folder
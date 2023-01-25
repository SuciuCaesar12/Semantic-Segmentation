import yaml
import os.path
import cv2
import tqdm
import numpy as np


def get_ground_truth_colors():
    """
    :return: list of all unique ground truth colors identified in images' masks
    """
    path_segm_class = './VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass'
    unique_colors = []

    for img in tqdm.tqdm(os.listdir(path_segm_class)):
        # read image
        img_path = os.path.join(path_segm_class, img)
        img = cv2.imread(img_path)
        # iterate through each pixel to identify unique colors
        for row in img:
            for pixel in row:
                if tuple(pixel) not in unique_colors:
                    unique_colors.append(tuple(pixel))

    return unique_colors


def create_palette(colors):
    palette = np.zeros(shape=(22 * 30, 200, 3), dtype='uint8')
    for i in range(22):
        cv2.rectangle(palette,
                      (0, 30 * i),
                      (200, 30 * (i + 1)),
                      colors[i],
                      -1)
        cv2.putText(palette, str(colors[i]), (0, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return palette


def read_yaml_file(path):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def reload_sample_to_img(sample, label_color, target):
    if not target:
        return (sample.numpy() * 255.0).astype('uint8')[:, :, ::-1]
    else:
        sample = sample.numpy()
        height, width, _ = sample.shape
        y = np.zeros(shape=(height, width, 3))
        for h in range(height):
            for w in range(width):
                color = list(label_color.values())[np.argmax(sample[h][w])]
                y[h][w] = list(eval(color))
        return y.astype('uint8')

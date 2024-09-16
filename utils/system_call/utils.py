import json
import math
import numpy as np
import matplotlib.pyplot as plt


def show_images(image_paths, scores):
    columns = int(math.sqrt(len(image_paths)))
    if columns == 0:
        return "No results"
    rows = int(np.ceil(len(image_paths) / columns))
    fig = plt.figure(figsize=(15, 10))

    for i in range(1, len(image_paths) + 1):
        image = plt.imread(image_paths[i - 1])
        ax = fig.add_subplot(rows, columns, i)

        title_name = '/'.join(image_paths[i - 1].split('/')[-3:])
        title_score = f'{scores[i - 1]:.3f}'
        tilte = title_name + '-' + title_score
        ax.set_title(tilte)

        plt.imshow(image)
        plt.axis('off')

    plt.show()


def display_images(result_info):
    """Display images from the given paths."""
    # Replace base directory for each image path
    image_paths = list(result_info.keys())
    image_paths = [image_path.replace('./', './dataset/') for image_path in image_paths]
    scores = list(result_info.values())
    show_images(image_paths, scores)


def reading_json_file(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data


def all_values_none(d):
    return all(value is None for value in d.values())

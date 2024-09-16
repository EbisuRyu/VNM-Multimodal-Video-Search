import json
import faiss
import numpy as np


def load_id2image_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return {int(k): v for k, v in data.items()}


def result_format(image_paths, scores):
    result = {image_path: score for image_path,
              score in zip(image_paths, scores)}
    return result


def find_index_from_image_path(id2image_fps, image_path_subset):
    keys = [int(index) for index, image_path in id2image_fps.items()
            if image_path in image_path_subset]
    return keys


def load_bin_file(bin_file):
    return faiss.read_index(bin_file)


def top_k_unique_in_order(array, top_k):
    seen, unique = [], []
    for value in array:
        if value not in seen:
            unique.append(value)
            seen.append(value)
            if len(unique) == top_k:
                break
    return np.array(unique)

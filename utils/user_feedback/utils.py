import json
import faiss
import numpy as np


def cosine_similarity(vec1, vec2):
    # Tính dot product giữa hai vector
    dot_product = np.dot(vec1, vec2)

    # Tính norm (độ dài) của từng vector
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # Tính cosine similarity
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)

    return cosine_sim


def find_index_from_image_path(id2image_fps, image_path_subset):
    # Tìm các chỉ số (index) theo thứ tự xuất hiện của image_path_subset
    keys = []
    for image_path in image_path_subset:
        for index, path in id2image_fps.items():
            if path == image_path:
                keys.append(int(index))
                break  # Dừng vòng lặp khi tìm được index phù hợp
    return keys


def load_bin_file(bin_file):
    return faiss.read_index(bin_file)


def load_id2image_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return {int(k): v for k, v in data.items()}

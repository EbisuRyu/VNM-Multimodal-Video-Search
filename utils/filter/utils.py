def result_format(image_paths, scores):
    result = {image_path: score for image_path, score in zip(image_paths, scores)}
    return result

def find_index_from_image_path(id2image_fps, image_path_subset):
    keys = [int(index) for index, image_path in id2image_fps.items() if image_path in image_path_subset]
    return keys

import numpy as np


def merge_searching_results_by_addition(list_results):
    '''
    Arg:
      list_results:
        [
            {
                image_path: score,
                image_path: score,
                image_path: score
            }
        ]
    '''
    if len(list_results) == 1:
        return list_results[0]

    merged_result_dict = {}

    for result_dict in list_results:
        scores = np.array(list(result_dict.values()))
        mean_score = np.mean(scores)
        std_dev_score = np.std(scores)
        normalized_scores = (scores - mean_score) / \
            (std_dev_score + 1e-6)  # Z-score normalization

        for image_path, normalized_score in zip(result_dict.keys(), normalized_scores):
            if image_path in merged_result_dict:
                merged_result_dict[image_path] += normalized_score
            else:
                merged_result_dict[image_path] = normalized_score

    sorted_merged_result_dict = dict(
        sorted(merged_result_dict.items(),
               key=lambda item: item[1], reverse=True)
    )

    return sorted_merged_result_dict

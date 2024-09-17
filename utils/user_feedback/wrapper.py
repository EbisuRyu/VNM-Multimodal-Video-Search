import numpy as np
from utils.user_feedback.utils import cosine_similarity, find_index_from_image_path


class Wrapper:
    def __init__(self, search_engine):
        self.search_engine = search_engine
        # Load FAISS index và ánh xạ id -> image path
        if search_engine:
            self.index = search_engine.index
            self.id2image_fps = search_engine.id2image_fps

    def extract_features_from_bin(self, image_path_subset):
        index_list = find_index_from_image_path(
            id2image_fps=self.id2image_fps,
            image_path_subset=image_path_subset
        )
        # Tái tạo lại các vector đặc trưng từ chỉ số
        retrieved_vectors = []
        for idx in index_list:
            feature_vector = self.index.reconstruct(idx)
            retrieved_vectors.append(feature_vector)
        return retrieved_vectors

    def __call__(self, query_text, eval_keyframe_subset: list, pos_keyframe_subset: list, neg_keyframe_subset: list):
        text_features = self.search_engine.encode_text(query_text)
        # Trích xuất các vector đặc trưng của các keyframe
        eval_keyframe_vectors = self.extract_features_from_bin(image_path_subset=eval_keyframe_subset)
        pos_keyframe_vectors = self.extract_features_from_bin(image_path_subset=pos_keyframe_subset)
        neg_keyframe_vectors = self.extract_features_from_bin(image_path_subset=neg_keyframe_subset)

        # Tạo dict chứa kết quả rerank
        rerank_result = {}

        for eval_keyframe_path, vector_keyframe in zip(eval_keyframe_subset, eval_keyframe_vectors):
            pos_sum, neg_sum = 0, 0
            # Tính tổng cosine similarity cho positive keyframes
            for vector_pos_keyframe in pos_keyframe_vectors:
                pos_sum += cosine_similarity(vector_keyframe, vector_pos_keyframe)
            # Tính tổng cosine similarity cho negative keyframes
            for vector_neg_keyframe in neg_keyframe_vectors:
                neg_sum += cosine_similarity(vector_keyframe, vector_neg_keyframe)
            # Tính điểm cho từng eval_keyframe
            score = cosine_similarity(vector_keyframe, text_features.flatten()) + pos_sum - neg_sum
            rerank_result[eval_keyframe_path] = score

        # Sắp xếp lại kết quả theo thứ tự giảm dần
        sorted_result = dict(sorted(rerank_result.items(), key=lambda x: x[1], reverse=True))
        return sorted_result

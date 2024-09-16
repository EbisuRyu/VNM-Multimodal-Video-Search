import faiss
import torch
import numpy as np
from transformers import XLMRobertaTokenizer
from extra.unilm.beit3.modeling_finetune import beit3_base_patch16_224_retrieval
from utils.user_feedback.utils import cosine_similarity, find_index_from_image_path, load_id2image_file

WEIGHT_DIR = './dict/beit/weights'
BIN_DIR = './dict/beit'


class UserFeedback:
    def __init__(self):
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load model và tokenizer
        checkpoint = torch.load(f'{WEIGHT_DIR}/beit3_base_itc_patch16_224.pth', map_location=self.__device)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(f'{WEIGHT_DIR}/beit3.spm')
        self.model = beit3_base_patch16_224_retrieval(pretrained=True)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.__device)
        self.model.eval()

        # Load FAISS index và ánh xạ id -> image path
        self.index = faiss.read_index(f'{BIN_DIR}/beit.bin')
        self.id2image_fps = load_id2image_file(json_path=f'{BIN_DIR}/beit.json')

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
        # Tokenize query text
        text_tokens = self.tokenizer(text=query_text, return_tensors='pt', truncation=True, padding=True)["input_ids"]
        text_tokens = text_tokens.to(self.__device)

        # Dùng model để trích xuất feature từ query text
        with torch.no_grad():
            _, text_features = self.model(
                text_description=text_tokens,
                only_infer=True
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Trích xuất các vector đặc trưng của các keyframe
        eval_keyframe_vectors = self.extract_features_from_bin(image_path_subset=eval_keyframe_subset)
        pos_keyframe_vectors = self.extract_features_from_bin(image_path_subset=pos_keyframe_subset)
        neg_keyframe_vectors = self.extract_features_from_bin(image_path_subset=neg_keyframe_subset)

        # Chuyển text_features từ tensor về numpy cho các phép tính cosine
        text_features_np = text_features.cpu().numpy().astype(np.float32)

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
            score = cosine_similarity(vector_keyframe, text_features_np.flatten()) + pos_sum - neg_sum
            rerank_result[eval_keyframe_path] = score

        # Sắp xếp lại kết quả theo thứ tự giảm dần
        sorted_result = dict(sorted(rerank_result.items(), key=lambda x: x[1], reverse=True))
        return sorted_result

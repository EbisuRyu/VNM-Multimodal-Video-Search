import torch
import faiss
import numpy as np
import open_clip
from utils.embedding_based_search.utils import load_bin_file, load_id2image_file, result_format, find_index_from_image_path


class CLIP:
    def __init__(self, clip_bin_file: str, clip_id2image_path: str, clip_model: str):
        super().__init__()
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.index = load_bin_file(clip_bin_file)
        self.id2image_fps = load_id2image_file(clip_id2image_path)

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name=clip_model,
            device=self.__device,
            pretrained='laion2b_s32b_b79k' if clip_model == 'ViT-H-14' else 'laion400m_e32'
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(clip_model)

    def image_search(self, query_image_path, image_path_subset, top_k):
        for index, image_path in self.id2image_fps.items():
            if image_path == query_image_path:
                id_query = index
                break
        image_features = self.index.reconstruct(id_query)
        image_features = np.expand_dims(image_features, 0)
        ##### IMAGE FEATURES EXTRACTING #####
        image_features = self.index.reconstruct(id_query).reshape(1, -1)
        ##### SEARCHING #####
        if image_path_subset is None:
            scores, idx_image = self.index.search(image_features, top_k)
        else:
            subset_index = find_index_from_image_path(
                id2image_fps=self.id2image_fps,
                image_path_subset=image_path_subset
            )
            id_selector = faiss.IDSelectorArray(subset_index)
            scores, idx_image = self.index.search(image_features, top_k, params=faiss.SearchParametersIVF(sel=id_selector))
        ##### GET INFOS KEYFRAMES_ID #####
        idx_image = idx_image.flatten()
        image_paths = list(map(self.id2image_fps.get, list(idx_image)))
        return result_format(image_paths, scores.flatten())

    def encode_text(self, query_text):
        ##### TEXT FEATURES EXTRACTING #####
        text_tokens = self.tokenizer([query_text]).to(self.__device)
        text_features = self.model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().detach().numpy().astype(np.float32)
        return text_features

    def text_search(self, query_text, image_path_subset, top_k):
        ##### TEXT FEATURES EXTRACTING #####
        text_features = self.encode_text(query_text)
        ##### SEARCHING #####
        if image_path_subset is None:
            scores, idx_image = self.index.search(text_features, top_k)
        else:
            subset_index = find_index_from_image_path(
                id2image_fps=self.id2image_fps,
                image_path_subset=image_path_subset
            )
            top_k = min(len(subset_index), top_k)
            id_selector = faiss.IDSelectorArray(subset_index)
            scores, idx_image = self.index.search(text_features, top_k, params=faiss.SearchParametersIVF(sel=id_selector))
        ##### GET INFOS KEYFRAMES_ID #####
        idx_image = idx_image.flatten()
        image_paths = list(map(self.id2image_fps.get, list(idx_image)))
        return result_format(image_paths, scores.flatten())

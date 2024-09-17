import torch
import faiss
import numpy as np
from lavis.models import load_model_and_preprocess
from utils.embedding_based_search.utils import load_bin_file, load_id2image_file, top_k_unique_in_order, result_format, find_index_from_image_path


class BLIP:
    def __init__(self, blip_bin_file: str, blip_id2image_path: str, blip_model="blip2_feature_extractor", model_type="pretrain_vitL"):
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name=blip_model,
            model_type=model_type,
            is_eval=True,
            device=self.__device
        )
        self.index = load_bin_file(bin_file=blip_bin_file)
        self.id2image_fps = load_id2image_file(json_path=blip_id2image_path)
    
    def encode_text(self, query_text):
        ##### TEXT FEATURES EXTRACTING #####
        processed_text = self.txt_processors["eval"](query_text)
        text_features = self.model.extract_features(
            {"text_input": [processed_text]},
            mode="text"
        )
        text_features = text_features.text_embeds_proj[:, 0, :]
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def text_search(self, query_text, image_path_subset, top_k):
        ##### TEXT FEATURES EXTRACTING #####
        text_features = self.encode_text(query_text)
        ##### SEARCHING #####
        if image_path_subset is None:
            scores, idx_image = self.index.search(text_features, 32*top_k)
        else:
            subset_index = find_index_from_image_path(
                id2image_fps=self.id2image_fps,
                image_path_subset=image_path_subset
            )
            top_k = min(len(subset_index), top_k)
            new_subset_index = []
            for idx in subset_index:
                start_index, end_index = idx*32, (idx+1)*32
                new_subset_index += range(start_index, end_index)
            id_selector = faiss.IDSelectorArray(new_subset_index)
            scores, idx_image = self.index.search(text_features, 32*top_k, params=faiss.SearchParametersIVF(sel=id_selector))
        ##### GET INFOS KEYFRAMES_ID #####
        idx_image = np.floor(idx_image.flatten() / 32).astype(np.int64)
        idx_image = top_k_unique_in_order(idx_image, top_k)
        image_paths = list(map(self.id2image_fps.get, list(idx_image)))
        return result_format(image_paths, scores.flatten())

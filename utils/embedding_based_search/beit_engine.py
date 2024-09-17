from utils.embedding_based_search.utils import load_bin_file, load_id2image_file, result_format, find_index_from_image_path
from extra.unilm.beit3.modeling_finetune import beit3_base_patch16_224_retrieval, beit3_large_patch16_384_retrieval
from extra.unilm.beit3.utils import load_model_and_may_interpolate
from transformers import XLMRobertaTokenizer
import numpy as np
import torch
import faiss
WEIGHT_DIR = './dict/beit/weights'


class BEIT:
    def __init__(self, model_type: str, beit_bin_file: str, beit_id2image_path: str):
        self.index = load_bin_file(bin_file=beit_bin_file)
        self.id2image_fps = load_id2image_file(json_path=beit_id2image_path)
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = XLMRobertaTokenizer(f'{WEIGHT_DIR}/beit3.spm')
        self.model_intitialize(model_type)
    
    def model_intitialize(self, model_type: str):
        if model_type == 'base':
            self.model = beit3_base_patch16_224_retrieval(pretrained=True)
            checkpoint = torch.load(f'{WEIGHT_DIR}/beit3_base_itc_patch16_224.pth')
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model = beit3_large_patch16_384_retrieval(pretrained=True)
            load_model_and_may_interpolate(f'{WEIGHT_DIR}/beit3_large_itc_patch16_224.pth', self.model, model_key='model', model_prefix='')
        self.model.to(self.__device)
        self.model.eval()
        
    def image_search(self, query_image_path, image_path_subset, top_k):
        for index, image_path in self.id2image_fps.items():
            if image_path == query_image_path:
                id_query = index
                break
        image_features = self.index.reconstruct(id_query)
        image_features = np.expand_dims(image_features, 0)
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

    def text_search(self, query_text, image_path_subset, top_k):
        text_tokens = self.tokenizer(
            text=query_text,
            return_tensors='pt'
        )["input_ids"]
        text_tokens = text_tokens.to(self.__device)
        with torch.no_grad():
            _, text_features = self.model(
                text_description=text_tokens,
                only_infer=True
            )
            text_features /= text_features.norm(dim=-1, keepdim=True)
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

from utils.embedding_based_search.clip_engine import CLIP
from utils.embedding_based_search.blip_engine import BLIP
from utils.embedding_based_search.beit_engine import BEIT
from utils.combine_module.utils import merge_searching_results_by_addition

CLIP_DIR = './dict/clip'
BLIP_DIR = './dict/blip'
BEIT_DIR = './dict/beit'


class EmbeddingBasedSearch:

    def __init__(self, use_clip_h14: bool, use_clip_l14: bool, use_blip: bool, use_beit: bool):
        if use_clip_h14:
            self.clip_h14_engine = CLIP(
                clip_bin_file=f'{CLIP_DIR}/h14_laion2b.bin',
                clip_id2image_path=f'{CLIP_DIR}/h14_laion2b.json',
                clip_model='ViT-H-14'
            )
        if use_clip_l14:
            self.clip_l14_engine = CLIP(
                clip_bin_file=f'{CLIP_DIR}/l14_laion400m.bin',
                clip_id2image_path=f'{CLIP_DIR}/l14_laion400m.json',
                clip_model='ViT-L-14'
            )
        if use_blip:
            self.blip_engine = BLIP(
                blip_bin_file=f'{BLIP_DIR}/blip_vit.bin',
                blip_id2image_path=f'{BLIP_DIR}/blip_vit.json',
                model_type='pretrain_vitL'
            )
        if use_beit:
            self.beit_engine = BEIT(
                beit_bin_file=f'{BEIT_DIR}/beit.bin',
                beit_id2image_path=f'{BEIT_DIR}/beit.json'
            )
        self.searching_mode = {
            'clip_h14_engine': True,
            'clip_l14_engine': True,
            'blip_engine': True,
            'beit_engine': True
        }

    def update_searching_mode(self, clip_h14_engine=True, clip_l14_engine=True, blip_engine=True, beit_engine=True):
        self.searching_mode = {
            'clip_h14_engine': clip_h14_engine,
            'clip_l14_engine': clip_l14_engine,
            'blip_engine': blip_engine,
            'beit_engine': beit_engine
        }

    def image_search(self, query_image_path, top_k):
        list_results = []
        if self.searching_mode['clip_h14_engine']:
            result = self.clip_h14_engine.image_search(
                query_image_path=query_image_path,
                image_path_subset=None,
                top_k=top_k
            )
            list_results.append(result)

        if self.searching_mode['clip_l14_engine']:
            result = self.clip_l14_engine.image_search(
                query_image_path=query_image_path,
                image_path_subset=None,
                top_k=top_k
            )
            list_results.append(result)

        if self.searching_mode['beit_engine']:
            result = self.beit_engine.image_search(
                query_image_path=query_image_path,
                image_path_subset=None,
                top_k=top_k
            )
            list_results.append(result)

        final_result = merge_searching_results_by_addition(list_results)
        top_k_final_result = dict(list(final_result.items())[:top_k])
        return top_k_final_result

    def text_search(self, query_text, image_path_subset, top_k):
        list_results = []
        if self.searching_mode['clip_h14_engine']:
            result = self.clip_h14_engine.text_search(
                query_text=query_text,
                image_path_subset=image_path_subset,
                top_k=top_k
            )
            list_results.append(result)

        if self.searching_mode['clip_l14_engine']:
            result = self.clip_l14_engine.text_search(
                query_text=query_text,
                image_path_subset=image_path_subset,
                top_k=top_k
            )
            list_results.append(result)

        if self.searching_mode['blip_engine']:
            result = self.blip_engine.text_search(
                query_text=query_text,
                image_path_subset=image_path_subset,
                top_k=top_k
            )
            list_results.append(result)

        if self.searching_mode['beit_engine']:
            result = self.beit_engine.text_search(
                query_text=query_text,
                image_path_subset=image_path_subset,
                top_k=top_k
            )
            list_results.append(result)

        final_result = merge_searching_results_by_addition(list_results)
        top_k_final_result = dict(list(final_result.items())[:top_k])
        return top_k_final_result

from utils.embedding_based_search.embedding_based_search import EmbeddingBasedSearch
from utils.temporal_search.temporal_search import TemporalSearch
from utils.user_feedback.user_feedback import UserFeedback
from utils.embedding_based_search.clip_engine import CLIP
from utils.embedding_based_search.blip_engine import BLIP
from utils.embedding_based_search.beit_engine import BEIT
from utils.system_call.utils import reading_json_file
from utils.system_call.utils import all_values_none

CLIP_DIR = './dict/clip'
BLIP_DIR = './dict/blip'
BEIT_DIR = './dict/beit'

class SearchingMethod:
    def __init__(self, clip_h14_engine=None, clip_h14_xlm_engine=None, clip_l14_engine=None, blip_vit_engine=None, blip_pretrain_engine=None, beit_base_engine=None, beit_large_engine=None):
        self.search_engine = EmbeddingBasedSearch(
            clip_h14_engine=clip_h14_engine,
            clip_h14_xlm_engine=clip_h14_xlm_engine,
            clip_l14_engine=clip_l14_engine,            
            blip_vit_engine=blip_vit_engine,
            blip_pretrain_engine=blip_pretrain_engine, 
            beit_base_engine=beit_base_engine,
            beit_large_engine=beit_large_engine
        )
        self.local_keyframe_dict = reading_json_file(json_path='./dict/local/local_dict.json')

    def search(self, query_text_1, query_text_2=None, top_k=10, image_path_subset=None, video_info=None, is_fusion=False):
        search_func = (self.search_engine.text_search if query_text_2 is None else TemporalSearch(self.search_engine).search)
        tag = self._determine_tag(query_text_2, image_path_subset, video_info, is_fusion)
        if video_info:
            image_path_subset = self.local_keyframe_dict[video_info['L']][video_info['V']]
        if query_text_2 is None:
            result = search_func(query_text_1, image_path_subset, top_k)
        else:
            result = search_func(query_text_1, query_text_2, image_path_subset, number_frame=50, top_k=top_k)
        return f'<{tag}>{query_text_1}', result

    def image_similarity(self, query_image_path, top_k):
        result = self.search_engine.image_search(
            query_image_path=query_image_path,
            top_k=top_k
        )
        return result

    def _determine_tag(self, query_text_2, image_path_subset, video_info, is_fusion):
        if video_info:
            return 'temp-vid' if query_text_2 else 'video'
        if is_fusion:
            return 'temp-fus' if query_text_2 else 'fusion'
        if image_path_subset:
            return 'temp-mul' if query_text_2 else 'multi'
        return 'temp' if query_text_2 else 'global'


class EmbeddingSpace:
    def __init__(self, use_clip_h14=False, use_clip_h14_xlm=False, use_clip_l14=False, use_blip_vit=False, use_blip_pretrain=False, use_base_beit=False, use_large_beit=False):
        self.model_initialize(use_clip_h14, use_clip_h14_xlm, use_clip_l14, use_blip_vit, use_blip_pretrain, use_base_beit, use_large_beit)
        self.searching_method = SearchingMethod(
            clip_h14_engine=self.clip_h14_engine if use_clip_h14 else None,
            clip_h14_xlm_engine=self.clip_h14_xlm_engine if use_clip_h14_xlm else None,
            clip_l14_engine=self.clip_l14_engine if use_clip_l14 else None, 
            blip_vit_engine=self.blip_vit_engine if use_blip_vit else None,
            blip_pretrain_engine=self.blip_pretrain_engine if use_blip_pretrain else None, 
            beit_base_engine=self.beit_base_engine if use_base_beit else None,
            beit_large_engine=self.beit_large_engine if use_large_beit else None
        )
        self.user_feedback = UserFeedback(
            clip_h14_engine=self.clip_h14_engine if use_clip_h14 else None,
            clip_h14_xlm_engine=self.clip_h14_xlm_engine if use_clip_h14_xlm else None,
            clip_l14_engine=self.clip_l14_engine if use_clip_l14 else None,
            beit_base_engine=self.beit_base_engine if use_base_beit else None,
            beit_large_engine=self.beit_large_engine if use_large_beit else None
        )
        self.search_history = []
        self.result_history = []
        self.use_model = {
            'clip_h14_engine': False,
            'clip_h14_xlm_engine': False,
            'clip_l14_engine': False,
            'blip_vit_engine': False,
            'blip_pretrain_engine': False,
            'beit_base_engine': False,
            'beit_large_engine': False
        }
        self.video_local = None
        self.fusion = False
        self.current_result = {}
        self.update_model(self.use_model)
    
    def model_initialize(self, use_clip_h14=False, use_clip_h14_xlm=False, use_clip_l14=False, use_blip_vit=False, use_blip_pretrain=False, use_base_beit=False, use_large_beit=False):
        if use_clip_h14:
            self.clip_h14_engine = CLIP(
                clip_bin_file=f'{CLIP_DIR}/h14_laion2b.bin',
                clip_id2image_path=f'{CLIP_DIR}/h14_laion2b.json',
                clip_model='ViT-H-14'
            )
        if use_clip_h14_xlm:
            self.clip_h14_xlm_engine = CLIP(
                clip_bin_file=f'{CLIP_DIR}/h14_xlm_laion5b.bin',
                clip_id2image_path=f'{CLIP_DIR}/h14_xlm_laion5b.json',
                clip_model='xlm-roberta-large-ViT-H-14'
            )
        if use_clip_l14:
            self.clip_l14_engine = CLIP(
                clip_bin_file=f'{CLIP_DIR}/l14_laion400m.bin',
                clip_id2image_path=f'{CLIP_DIR}/l14_laion400m.json',
                clip_model='ViT-L-14'
            )
        if use_blip_vit:
            self.blip_vit_engine = BLIP(
                blip_bin_file=f'{BLIP_DIR}/blip_vit.bin',
                blip_id2image_path=f'{BLIP_DIR}/blip_vit.json',
                model_type='pretrain_vitL'
            )
        if use_blip_pretrain:
            self.blip_pretrain_engine = BLIP(
                blip_bin_file=f'{BLIP_DIR}/blip_pretrain.bin',
                blip_id2image_path=f'{BLIP_DIR}/blip_pretrain.json',
                model_type='pretrain'
            )
        if use_base_beit:
            self.beit_base_engine = BEIT(
                model_type='base',
                beit_bin_file=f'{BEIT_DIR}/base_beit.bin',
                beit_id2image_path=f'{BEIT_DIR}/base_beit.json'
            )
        if use_large_beit:
            self.beit_large_engine = BEIT(
                model_type='large',
                beit_bin_file=f'{BEIT_DIR}/large_beit.bin',
                beit_id2image_path=f'{BEIT_DIR}/large_beit.json'
            )
            
    def update_video_local(self, video_local):
        self.video_local = None if all_values_none(video_local) else video_local

    def update_fusion_mode(self, fusion):
        self.fusion = fusion

    def update_model(self, use_model):
        self.use_model = use_model
        self.searching_method.search_engine.update_searching_mode(**use_model)
        self.user_feedback.update_searching_mode(**use_model)

    def delete_history(self):
        self.search_history.clear()
        self.result_history.clear()
        self.current_result = {}

    def back_to_before_result(self, index):
        self.search_history = self.search_history[:index + 1]
        self.result_history = self.result_history[:index + 1]
        self.current_result = self.result_history[-1]

    def feedback(self, query_text, pos_keyframe_subset, neg_keyframe_subset):
        reranked_result = self.user_feedback(
            query_text=query_text,
            eval_keyframe_subset=list(self.current_result.keys()),
            neg_keyframe_subset=neg_keyframe_subset,
            pos_keyframe_subset=pos_keyframe_subset
        )
        return reranked_result

    def search(self, query_text_1, query_text_2=None, metadata_result_subset=None, top_k=10):

        if len(self.search_history) > 0:
            image_path_subset = list(self.current_result.keys())
        else:
            image_path_subset = None

        search_params = {
            'query_text_1': query_text_1,
            'query_text_2': query_text_2,
            'top_k': top_k,
            'image_path_subset': metadata_result_subset if self.fusion else image_path_subset,
            'video_info': self.video_local,
            'is_fusion': self.fusion
        }
        mode = self._determine_search_mode()
        text_query, result = self.searching_method.search(**search_params)

        print(f'-------------------- Using {mode} Search --------------------')
        self.search_history.append(text_query)
        self.result_history.append(result)
        self.current_result = result
        return result

    def image_similarity(self, query_image_path, top_k):
        search_params = {
            'query_image_path': query_image_path,
            'top_k': top_k
        }
        result = self.searching_method.image_similarity(**search_params)
        self.search_history.append('image similarity')
        self.result_history.append(result)
        self.current_result = result
        return result

    def _determine_search_mode(self):
        if self.video_local:
            return 'Local Video'
        if self.fusion:
            return 'Fusion'
        if self.result_history:
            return 'Multistage'
        return 'Global'

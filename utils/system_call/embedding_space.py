from utils.embedding_based_search.embedding_based_search import EmbeddingBasedSearch
from utils.temporal_search.temporal_search import TemporalSearch
from utils.user_feedback.user_feedback import UserFeedback
from utils.system_call.utils import reading_json_file
from utils.system_call.utils import all_values_none


class SearchingMethod:
    def __init__(self, use_clip_h14=False, use_clip_l14=False, use_blip=False, use_beit=False):
        self.search_engine = EmbeddingBasedSearch(
            use_clip_h14=use_clip_h14,
            use_clip_l14=use_clip_l14,
            use_blip=use_blip,
            use_beit=use_beit
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
    def __init__(self, use_clip_h14=False, use_clip_l14=False, use_blip=False, use_beit=False):
        self.searching_method = SearchingMethod(
            use_clip_h14=use_clip_h14,
            use_clip_l14=use_clip_l14,
            use_blip=use_blip,
            use_beit=use_beit
        )
        self.user_feedback = UserFeedback()
        self.search_history = []
        self.result_history = []
        self.use_model = {
            'clip_h14_engine': False,
            'clip_l14_engine': False,
            'blip_engine': False,
            'beit_engine': False
        }
        self.video_local = None
        self.fusion = False
        self.current_result = {}
        self.update_model(self.use_model)

    def update_video_local(self, video_local):
        self.video_local = None if all_values_none(video_local) else video_local

    def update_fusion_mode(self, fusion):
        self.fusion = fusion

    def update_model(self, use_model):
        self.use_model = use_model
        self.searching_method.search_engine.update_searching_mode(**use_model)

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

from utils.user_feedback.wrapper import Wrapper
from utils.combine_module.utils import merge_searching_results_by_addition
from utils.user_feedback.utils import cosine_similarity, find_index_from_image_path


class UserFeedback:
    def __init__(self, clip_h14_engine, clip_h14_xlm_engine, clip_l14_engine, beit_base_engine, beit_large_engine):
        self.clip_h14_feedback = Wrapper(clip_h14_engine)
        self.clip_h14_xlm_feedback = Wrapper(clip_h14_xlm_engine)
        self.clip_l14_feedback = Wrapper(clip_l14_engine)
        self.beit_base_feedback = Wrapper(beit_base_engine)
        self.beit_large_feedback = Wrapper(beit_large_engine)
        self.searching_mode = {
            'clip_h14_engine': False,
            'clip_h14_xlm_engine': False,
            'clip_l14_engine': False,
            'beit_base_engine': False,
            'beit_large_engine': False
        } 

    def update_searching_mode(self, clip_h14_engine=True, clip_h14_xlm_engine=True, clip_l14_engine=True, blip_vit_engine=True, blip_pretrain_engine=True, beit_base_engine=True, beit_large_engine=True):
        self.searching_mode = {
            'clip_h14_engine': clip_h14_engine,
            'clip_h14_xlm_engine': clip_h14_xlm_engine,
            'clip_l14_engine': clip_l14_engine,
            'beit_base_engine': beit_base_engine,
            'beit_large_engine': beit_large_engine
        }

    def __call__(self, query_text, eval_keyframe_subset: list, pos_keyframe_subset: list, neg_keyframe_subset: list):
        list_results = []
        if self.searching_mode['clip_h14_engine']:
            result = self.clip_h14_feedback(
                query_text=query_text,
                eval_keyframe_subset=eval_keyframe_subset,
                pos_keyframe_subset=pos_keyframe_subset,
                neg_keyframe_subset=neg_keyframe_subset
            )
            list_results.append(result)
        
        if self.searching_mode['clip_h14_xlm_engine']:
            result = self.clip_h14_xlm_feedback(
                query_text=query_text,
                eval_keyframe_subset=eval_keyframe_subset,
                pos_keyframe_subset=pos_keyframe_subset,
                neg_keyframe_subset=neg_keyframe_subset
            )
            list_results.append(result)

        if self.searching_mode['clip_l14_engine']:
            result = self.clip_l14_feedback(
                query_text=query_text,
                eval_keyframe_subset=eval_keyframe_subset,
                pos_keyframe_subset=pos_keyframe_subset,
                neg_keyframe_subset=neg_keyframe_subset
            )
            list_results.append(result)

        if self.searching_mode['beit_base_engine']:
            result = self.beit_base_feedback(
                query_text=query_text,
                eval_keyframe_subset=eval_keyframe_subset,
                pos_keyframe_subset=pos_keyframe_subset,
                neg_keyframe_subset=neg_keyframe_subset
            )
            list_results.append(result)
        
        if self.searching_mode['beit_large_engine']:
            result = self.beit_large_feedback(
                query_text=query_text,
                eval_keyframe_subset=eval_keyframe_subset,
                pos_keyframe_subset=pos_keyframe_subset,
                neg_keyframe_subset=neg_keyframe_subset
            )
            list_results.append(result)
        final_result = merge_searching_results_by_addition(list_results)
        return final_result
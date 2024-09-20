from utils.combine_module.utils import merge_searching_results_by_addition


class EmbeddingBasedSearch:

    def __init__(self, clip_h14_engine=None, clip_h14_xlm_engine=None, clip_l14_engine=None, blip_engine=None, beit_base_engine=None, beit_large_engine=None):
        self.clip_h14_engine = clip_h14_engine
        self.clip_h14_xlm_engine = clip_h14_xlm_engine
        self.clip_l14_engine = clip_l14_engine
        self.blip_engine = blip_engine
        self.beit_base_engine = beit_base_engine
        self.beit_large_engine = beit_large_engine 
        self.searching_mode = {
            'clip_h14_engine': True,
            'clip_h14_xlm_engine': True,
            'clip_l14_engine': True,
            'blip_engine': True,
            'beit_base_engine': True,
            'beit_large_engine': True
        }

    def update_searching_mode(self, clip_h14_engine=True, clip_h14_xlm_engine=True, clip_l14_engine=True, blip_engine=True, beit_base_engine=True, beit_large_engine=True):
        self.searching_mode = {
            'clip_h14_engine': clip_h14_engine,
            'clip_h14_xlm_engine': clip_h14_xlm_engine,
            'clip_l14_engine': clip_l14_engine,
            'blip_engine': blip_engine,
            'beit_base_engine': beit_base_engine,
            'beit_large_engine': beit_large_engine
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
            
        if self.searching_mode['clip_h14_xlm_engine']:
            result = self.clip_h14_xlm_engine.image_search(
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

        if self.searching_mode['beit_base_engine']:
            result = self.beit_base_engine.image_search(
                query_image_path=query_image_path,
                image_path_subset=None,
                top_k=top_k
            )
            list_results.append(result)
        
        if self.searching_mode['beit_large_engine']:
            result = self.beit_large_engine.image_search(
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

        if self.searching_mode['clip_h14_xlm_engine']:
            result = self.clip_h14_xlm_engine.text_search(
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

        if self.searching_mode['beit_base_engine']:
            result = self.beit_base_engine.text_search(
                query_text=query_text,
                image_path_subset=image_path_subset,
                top_k=top_k
            )
            list_results.append(result)
        
        if self.searching_mode['beit_large_engine']:
            result = self.beit_large_engine.text_search(
                query_text=query_text,
                image_path_subset=image_path_subset,
                top_k=top_k
            )
            list_results.append(result)

        final_result = merge_searching_results_by_addition(list_results)
        top_k_final_result = dict(list(final_result.items())[:top_k])
        return top_k_final_result

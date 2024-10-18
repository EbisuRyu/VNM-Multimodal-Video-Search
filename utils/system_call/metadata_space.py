from utils.object_color_search.object_color_search import ObjectColorSearch
from utils.filter.tag_recommend import TagRecommender
from utils.filter.filter import Filter
from utils.asr.asr_search import AsrSearch
from utils.combine_module.utils import merge_searching_results_by_addition
from utils.object_color_search.utils import all_values_none


class SearchingMethod:
    def __init__(self):
        self.object_color_search = ObjectColorSearch(
            oclass_search_type='bm25',
            bbox_search_type='bm25'
        )
        self.filter = Filter(
            tag_search_type='bm25'
        )
        self.asr_search = AsrSearch(
            json_folder='./dict/asr',
            stopwords=None
        )

    def search(self, asr_query, ocr_query, tag_query, oclass_queries, bbox_queries, image_path_subset, top_k):
        list_results = []
        if ocr_query is not None or tag_query is not None:
            result = self.filter.search(
                ocr_query=ocr_query,
                tag_query=tag_query,
                image_path_subset=image_path_subset,
                top_k=top_k
            )
            list_results.append(result)
        if not (all_values_none(oclass_queries) and all_values_none(bbox_queries)):
            result = self.object_color_search.metadata_search(
                oclass_queries=oclass_queries,
                bbox_queries=bbox_queries,
                image_path_subset=image_path_subset,
                top_k=top_k
            )
            list_results.append(result)
        if asr_query is not None:
            result = self.asr_search.search(
                input_query=asr_query,
                top_k=top_k
            )
            list_results.append(result)
        final_result = merge_searching_results_by_addition(list_results)
        final_top_k_result = dict(list(final_result.items())[:top_k])
        return final_top_k_result


class MetadataSpace:
    def __init__(self):
        self.searching_method = SearchingMethod()
        self.tag_recommender = TagRecommender()
        self.current_result = {}
        self.tag_recommendation = []
        self.fusion = False

    def tag_recommend(self, text_input):
        self.tag_recommendation = self.tag_recommender(text_input)

    def update_fusion_mode(self, fusion):
        self.fusion = fusion

    def delete_history(self):
        self.current_result = {}

    def search(self, asr_query, ocr_query, tag_query, oclass_queries, bbox_queries, embedding_result_subset, top_k):
        search_params = {
            'asr_query': asr_query,
            'ocr_query': ocr_query,
            'tag_query': tag_query,
            'oclass_queries': oclass_queries,
            'bbox_queries': bbox_queries,
            'image_path_subset': embedding_result_subset if self.fusion else None,
            'top_k': top_k
        }
        result = self.searching_method.search(**search_params)
        self.current_result = result
        return result

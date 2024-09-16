from utils.filter.ocr_elastic_search import ElasticSearch
from utils.filter.tag_search import TagSearch
from utils.combine_module.utils import merge_searching_results_by_addition


class Filter:

    def __init__(self, tag_search_type):
        self.ocr_search = ElasticSearch(
            index_name='ocr_engine',
            user_name='elastic',
            password='123456'
        )
        self.tag_search = TagSearch(
            search_type=tag_search_type
        )

    def search(self, ocr_query, tag_query, image_path_subset, top_k):
        list_results = []
        if ocr_query is not None:
            result = self.ocr_search.search(
                input_query=ocr_query,
                image_path_subset=image_path_subset,
                top_k=top_k
            )
            list_results.append(result)
        if tag_query is not None:
            result = self.tag_search(
                input_query=tag_query,
                image_path_subset=image_path_subset,
                top_k=top_k
            )
            list_results.append(result)
        final_result = merge_searching_results_by_addition(list_results)
        final_top_k_result = dict(list(final_result.items())[:top_k])
        return final_top_k_result

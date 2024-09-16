from utils.object_color_search.bbox_engine import BBoxSearchEngine
from utils.object_color_search.oclass_engine import OClassSearchEngine
from utils.object_color_search.utils import all_values_none
from utils.combine_module.utils import merge_searching_results_by_addition

METADATA_DIR = './dict/metadata'


class ObjectColorSearch:

    def __init__(self, oclass_search_type, bbox_search_type):
        self.oclass_engine = OClassSearchEngine(
            save_path=f'{METADATA_DIR}/{oclass_search_type}',
            id2image_path_dict={
                'object_class': f'{METADATA_DIR}/{oclass_search_type}/object_class/id2image_fps_object_class.json',
                'object_number': f'{METADATA_DIR}/{oclass_search_type}/object_number/id2image_fps_object_number.json',
                'color_class': f'{METADATA_DIR}/{oclass_search_type}/color_class/id2image_fps_color_class.json'
            },
            search_type=oclass_search_type
        )
        self.bbox_engine = BBoxSearchEngine(
            save_path=f'{METADATA_DIR}/{bbox_search_type}',
            id2image_path_dict={
                'object_bbox': f'{METADATA_DIR}/{bbox_search_type}/object_bbox/id2image_fps_object_bbox.json',
                'color_bbox': f'{METADATA_DIR}/{bbox_search_type}/color_bbox/id2image_fps_color_bbox.json'
            },
            search_type=bbox_search_type
        )

    def metadata_search(self, oclass_queries, bbox_queries, image_path_subset, top_k):
        list_results = []
        if not all_values_none(oclass_queries):
            oclass_result = self.oclass_engine(
                input_queries=oclass_queries,
                image_path_subset=image_path_subset,
                top_k=top_k
            )
            list_results.append(oclass_result)

        if not all_values_none(bbox_queries):
            bbox_result = self.bbox_engine(
                input_queries=bbox_queries,
                image_path_subset=image_path_subset,
                top_k=top_k
            )
            list_results.append(bbox_result)

        final_result = merge_searching_results_by_addition(list_results)
        top_k_final_result = dict(list(final_result.items())[:top_k])
        return top_k_final_result

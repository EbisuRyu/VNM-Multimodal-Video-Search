from collections import deque
from utils.temporal_search.utils import Metadata, get_video_frame


class TemporalSearch:

    def __init__(self, search_engine):
        # Có thể truyền nhiều engine khác nhau
        self.search_engine = search_engine

    def engine_search_method(self, query_text, image_path_subset, top_k):
        result = self.search_engine.text_search(
            query_text=query_text,
            image_path_subset=image_path_subset,
            top_k=top_k
        )
        return result

    def merging_result(self, result_1, result_2):
        merged_result_list = []
        for keyframe_path, score in result_1.items():
            video_id, keyframe_id = get_video_frame(keyframe_path)
            merged_result_list.append(Metadata(video_id, keyframe_id, score, 1))
        for keyframe_path, score in result_2.items():
            video_id, keyframe_id = get_video_frame(keyframe_path)
            merged_result_list.append(Metadata(video_id, keyframe_id, score, 2))
        return merged_result_list

    def rerank_method(self, merged_result_list, number_frame):
        deque_storage = deque()
        current_index, next_index = 0, 0

        for current_index in range(len(merged_result_list)):
            while current_index >= next_index:
                next_index += 1

            if merged_result_list[current_index].type == 2:
                continue

            while (len(deque_storage) > 0 and
                   (current_index >= deque_storage[0] or merged_result_list[current_index].video_id != merged_result_list[deque_storage[0]])):
                deque_storage.popleft()

            while (next_index < len(merged_result_list) and
                   merged_result_list[current_index].video_id == merged_result_list[next_index].video_id and
                   merged_result_list[current_index].keyframe_id + number_frame >= merged_result_list[next_index].keyframe_id):

                while (len(deque_storage) > 0 and merged_result_list[next_index].type == 2 and
                       merged_result_list[deque_storage[-1]].score < merged_result_list[next_index].score):
                    deque_storage.pop()

                deque_storage.append(next_index)
                next_index += 1

            if (len(deque_storage) > 0 and merged_result_list[current_index].video_id == merged_result_list[deque_storage[0]].video_id
                    and merged_result_list[current_index].keyframe_id + number_frame >= merged_result_list[deque_storage[0]].keyframe_id):
                merged_result_list[current_index].score += merged_result_list[deque_storage[0]].score
                merged_result_list[current_index].next_keyframe_isok = 1

        deque_storage.clear()
        return merged_result_list

    def result_format(self, merged_result_list, top_k):
        final_result = {}
        for metadata in merged_result_list:
            if metadata.type == 1:
                keyframe_path = '/'.join(['./distilled_keyframe', metadata.video_id, str(metadata.keyframe_id) + '.jpg'])
                final_result[keyframe_path] = metadata.score if metadata.next_keyframe_isok == 1 else metadata.score / 1
        top_k_final_result = dict(sorted(final_result.items(),key=lambda item: item[1],reverse=True)[:top_k])
        return top_k_final_result

    def search(self, query_text_1: str, query_text_2: str, image_path_subset=None, number_frame=50, top_k=100):
        result_1 = self.engine_search_method(
            query_text=query_text_1,
            image_path_subset=image_path_subset,
            top_k=top_k * 10
        )
        result_2 = self.engine_search_method(
            query_text=query_text_2,
            image_path_subset=image_path_subset,
            top_k=top_k * 10
        )
        merged_result_list = self.merging_result(result_1, result_2)
        merged_result_list.sort()
        merged_result_list = self.rerank_method(
            merged_result_list=merged_result_list,
            number_frame=number_frame
        )
        return self.result_format(merged_result_list, top_k)

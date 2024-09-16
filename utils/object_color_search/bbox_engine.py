import numpy as np
import bm25s
from utils.object_color_search.wrapper import TfidfWrapperEngine, Bm25WrapperEngine
from utils.object_color_search.utils import find_index_from_image_path, result_format
from utils.combine_module.utils import merge_searching_results_by_addition


class TfidfBBoxSearchEngine:
    def __init__(self, save_path, id2image_path_dict):
        self.object_bbox = TfidfWrapperEngine(
            save_tfidf_path=save_path,
            id2image_path=id2image_path_dict['object_bbox'],
            data_type='object_bbox'
        )
        self.color_bbox = TfidfWrapperEngine(
            save_tfidf_path=save_path,
            id2image_path=id2image_path_dict['color_bbox'],
            data_type='color_bbox'
        )

    def load_transform(self, transform_type):
        context_matrix, id2image_fps, tfidf_transform = (
            (
                self.object_bbox.context_matrix,
                self.object_bbox.id2image_fps,
                self.object_bbox.tfidf_transform
            )
            if transform_type == 'object_bbox' else
            (
                self.color_bbox.context_matrix,
                self.color_bbox.id2image_fps,
                self.color_bbox.tfidf_transform
            )
        )
        return context_matrix, id2image_fps, tfidf_transform

    def search(self, input_query, image_path_subset, transform_type, top_k):
        context_matrix, id2image_fps, tfidf_transform = self.load_transform(transform_type)
        vectorize = tfidf_transform.transform([input_query])
        if image_path_subset is not None:
            subset_index = find_index_from_image_path(
                id2image_fps=id2image_fps,
                image_path_subset=image_path_subset
            )
            context_matrix = context_matrix[subset_index]
            id2image_fps = {index: id2image_fps[fps_index] for index, fps_index in enumerate(subset_index)}

        scores = vectorize.dot(context_matrix.T).toarray()[0]
        sort_index = np.argsort(scores)[::-1][:top_k]
        scores = scores[sort_index]
        image_paths = list(map(id2image_fps.get, list(sort_index)))
        return image_paths, scores


class Bm25BBoxSearchEngine:
    def __init__(self, save_path, id2image_path_dict):
        self.object_bbox = Bm25WrapperEngine(
            save_bm25_path=save_path,
            id2image_path=id2image_path_dict['object_bbox'],
            data_type='object_bbox'
        )
        self.color_bbox = Bm25WrapperEngine(
            save_bm25_path=save_path,
            id2image_path=id2image_path_dict['color_bbox'],
            data_type='color_bbox'
        )

    def load_transform(self, transform_type):
        retriever, id2image_fps = (
            (
                self.object_bbox.bm25_retriever,
                self.object_bbox.id2image_fps
            )
            if transform_type == 'object_bbox' else
            (
                self.color_bbox.bm25_retriever,
                self.color_bbox.id2image_fps
            )
        )
        return retriever, id2image_fps

    def search(self, input_query, image_path_subset, transform_type, top_k):
        retriever, id2image_fps = self.load_transform(transform_type)
        query_tokens = bm25s.tokenize(input_query)
        if image_path_subset is None:
            results, list_score = retriever.retrieve(query_tokens, k=top_k)
            list_index = [item['id'] for item in results[0]]
        else:
            top_k = top_k if top_k < len(image_path_subset) else len(image_path_subset)
            subset_index = find_index_from_image_path(
                id2image_fps=id2image_fps,
                image_path_subset=image_path_subset
            )
            corpus = np.array(retriever.corpus)
            new_corpus = [item['text'] for item in corpus[subset_index].tolist()]
            id2image_fps = {index: id2image_fps[fps_index] for index, fps_index in enumerate(subset_index)}
            # Tokenize the corpus and only keep the ids (faster and saves memory)
            corpus_tokens = bm25s.tokenize(new_corpus)
            # Create the BM25 model and index the corpus
            retriever = bm25s.BM25(method="lucene")
            retriever.index(corpus_tokens)
            results, list_score = retriever.retrieve(query_tokens, k=top_k)
            list_index = results[0]
        image_paths = list(map(id2image_fps.get, list(list_index)))
        return image_paths, list_score[0]


class BBoxSearchEngine:
    def __init__(self, save_path, id2image_path_dict, search_type):
        if search_type == 'tf-idf':
            self.search_engine = TfidfBBoxSearchEngine(
                save_path=save_path,
                id2image_path_dict=id2image_path_dict
            )
        elif search_type == 'bm25':
            self.search_engine = Bm25BBoxSearchEngine(
                save_path=save_path,
                id2image_path_dict=id2image_path_dict
            )

    def __call__(self, input_queries, image_path_subset, top_k=100):
        list_results = []
        for input_type in ['object_bbox', 'color_bbox']:
            if input_queries[input_type] is not None:
                image_paths, scores = self.search_engine.search(
                    input_query=input_queries[input_type],
                    image_path_subset=image_path_subset,
                    transform_type=input_type,
                    top_k=top_k
                )
                list_results.append(result_format(image_paths, scores))

        final_result = merge_searching_results_by_addition(list_results)
        top_k_final_result = dict(list(final_result.items())[:top_k])
        return top_k_final_result

import os
import json
import scipy
import bm25s
import pickle
import numpy as np
from utils.filter.utils import find_index_from_image_path, result_format


class TfidfTagSearchEngine:
    def __init__(self, save_tfidf_path, id2image_path):
        self.save_tfids_path = save_tfidf_path
        self.id2image_fps = self.load_id2image_file(id2image_path)
        self.context_matrix, self.tfidf_transform = self.load_context()

    def load_id2image_file(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return {int(k): v for k, v in data.items()}

    def load_context(self):
        tfidf_transform_path = os.path.join(
            self.save_tfids_path, 'transform_tag.pkl')
        context_matrix_path = os.path.join(
            self.save_tfids_path, 'sparse_context_matrix_tag.npz')
        context_matrix = scipy.sparse.load_npz(context_matrix_path)
        with open(tfidf_transform_path, 'rb') as f:
            tfidf_transform = pickle.load(f)
        return context_matrix, tfidf_transform

    def search(self, input_query, image_path_subset, top_k):
        context_matrix = self.context_matrix
        id2image_fps = self.id2image_fps
        vectorize = self.tfidf_transform.transform([input_query])
        if image_path_subset is not None:
            subset_index = find_index_from_image_path(
                id2image_fps=self.id2image_fps,
                image_path_subset=image_path_subset
            )
            context_matrix = context_matrix[subset_index]
            id2image_fps = {
                index: id2image_fps[fps_index]
                for index, fps_index in enumerate(subset_index)
            }
        scores = vectorize.dot(context_matrix.T).toarray()[0]
        sort_index = np.argsort(scores)[::-1][:top_k]
        scores = scores[sort_index]
        image_paths = list(map(id2image_fps.get, list(sort_index)))
        return image_paths, scores


class Bm25TagSearchEngine:
    def __init__(self, save_bm25_path, id2image_path):
        self.save_bm25_path = save_bm25_path
        self.id2image_fps = self.load_id2image_file(id2image_path)
        self.bm25_retriever = bm25s.BM25.load(
            self.save_bm25_path, load_corpus=True)

    def load_id2image_file(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return {int(k): v for k, v in data.items()}

    def search(self, input_query, image_path_subset, top_k):
        corpus = self.bm25_retriever.corpus
        query_tokens = bm25s.tokenize(input_query)
        id2image_fps = self.id2image_fps
        if image_path_subset is None:
            results, list_score = self.bm25_retriever.retrieve(
                query_tokens, k=top_k)
            list_index = [item['id'] for item in results[0]]
        else:
            max_k = len(image_path_subset)
            top_k = top_k if top_k < max_k else max_k
            subset_index = find_index_from_image_path(
                id2image_fps=self.id2image_fps,
                image_path_subset=image_path_subset
            )
            corpus = np.array(corpus)
            new_corpus = [item['text']
                          for item in corpus[subset_index].tolist()]
            id2image_fps = {
                index: id2image_fps[fps_index]
                for index, fps_index in enumerate(subset_index)
            }
            # Tokenize the corpus and only keep the ids (faster and saves memory)
            corpus_tokens = bm25s.tokenize(new_corpus)
            # Create the BM25 model and index the corpus
            retriever = bm25s.BM25(method="lucene")
            retriever.index(corpus_tokens)
            results, list_score = retriever.retrieve(query_tokens, k=top_k)
            list_index = results[0]
        image_paths = list(map(id2image_fps.get, list(list_index)))
        return image_paths, list_score[0]


class TagSearch:
    def __init__(self, search_type):
        if search_type == 'tf-idf':
            self.search_engine = TfidfTagSearchEngine(
                save_tfidf_path='./dict/tag/tf-idf',
                id2image_path='./dict/tag/tf-idf/id2image_fps_tag.json'
            )
        elif search_type == 'bm25':
            self.search_engine = Bm25TagSearchEngine(
                save_bm25_path='./dict/tag/bm25',
                id2image_path='./dict/tag/bm25/id2image_fps_tag.json'
            )

    def __call__(self, input_query, image_path_subset, top_k=100):
        image_paths, scores = self.search_engine.search(
            input_query=input_query,
            image_path_subset=image_path_subset,
            top_k=top_k
        )

        result = result_format(image_paths, scores)
        return result

import os
import json
import scipy
import pickle
import bm25s


class TfidfWrapperEngine:
    def __init__(self, save_tfidf_path, id2image_path, data_type):
        self.save_tfids_path = save_tfidf_path
        self.data_type = data_type
        self.id2image_fps = self.load_id2image_file(id2image_path)
        self.context_matrix, self.tfidf_transform = self.load_context()

    def load_id2image_file(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return {int(k): v for k, v in data.items()}

    def load_context(self):
        tfidf_transform_path = os.path.join(self.save_tfids_path, self.data_type, f'transform_{self.data_type}.pkl')
        context_matrix_path = os.path.join(self.save_tfids_path, self.data_type, f'sparse_context_matrix_{self.data_type}.npz')
        context_matrix = scipy.sparse.load_npz(context_matrix_path)
        with open(tfidf_transform_path, 'rb') as f:
            tfidf_transform = pickle.load(f)
        return context_matrix, tfidf_transform


class Bm25WrapperEngine:
    def __init__(self, save_bm25_path, id2image_path, data_type):
        self.save_bm25_path = save_bm25_path + '/' + data_type
        self.data_type = data_type
        self.id2image_fps = self.load_id2image_file(id2image_path)
        self.bm25_retriever = bm25s.BM25.load(
            save_dir=self.save_bm25_path, 
            load_corpus=True
        )

    def load_id2image_file(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return {int(k): v for k, v in data.items()}

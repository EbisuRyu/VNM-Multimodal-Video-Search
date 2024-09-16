import os
import json
import torch
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from googletrans import Translator

DICT_TAG_PATH = "./dict/tag"


class TagRecommender:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        self.translator = Translator()
        self.tags = None
        self.tags_embeddings = None

    def recommend_tags(self, query, max_recommend=30):
        query = self.query_translate(query)
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(query_embedding, torch.tensor(self.tags_embeddings))
        top_results = torch.topk(cosine_scores, k=max_recommend)
        recommended_tags = [self.tags[idx] for idx in top_results.indices[0].cpu().numpy()]
        processed_tags = ['_'.join(tag.split()) for tag in recommended_tags]
        return processed_tags

    def query_translate(self, query):
        try:
            detected_language = detect(query)
            if detected_language != 'en':
                query_translated = self.translator.translate(text=query, src=detected_language, dest='en')
                return query_translated.text
            else:
                return query
        except Exception as e:
            print(f"Error detecting or translating language: {e}")
            return query

    def load_tag_embedding(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.tags = data['tags']
        self.tags_embeddings = data['embeddings']

    def save_tag_embedding(self, tag_list, output_file):
        if not os.path.exists(tag_list):
            raise FileNotFoundError(
                f"File not found: {tag_list}. Please provide a valid file path.")
        with open(tag_list, 'r', encoding='utf-8') as f:
            self.tags = json.load(f)
        tags_embeddings = self.model.encode(self.tags, convert_to_tensor=False)
        data_to_save = {
            'tags': self.tags,
            'embeddings': tags_embeddings.tolist()
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    def __call__(self, query, max_recommend=30):
        tag_list_path = os.path.join(DICT_TAG_PATH, 'tag_list.json')
        tag_embedding_path = os.path.join(DICT_TAG_PATH, 'tag_embedding.json')
        if not os.path.exists(tag_embedding_path):
            self.save_tag_embedding(tag_list_path, tag_embedding_path)
        self.load_tag_embedding(tag_embedding_path)
        return self.recommend_tags(query, max_recommend)

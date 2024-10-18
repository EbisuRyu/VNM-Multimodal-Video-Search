import os
import json
import string
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer  

class AsrSearch:
    def __init__(self, json_folder, stopwords=None):
        self.json_folder = json_folder
        self.stopwords = stopwords if stopwords else set()
        self.documents = []  
        self.metadata = []
        self.bm25 = None     

        self._load_data()
        self._initialize_bm25()

    def _load_data(self):
        """Load JSON files from the folder and preprocess the data."""
        for json_filename in os.listdir(self.json_folder):
            if not json_filename.endswith(".json"):
                continue

            json_file_path = os.path.join(self.json_folder, json_filename)
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                for entry in data:
                    tokens = self._preprocess(entry['text'])
                    if tokens:  
                        self.documents.append(tokens)
                        self.metadata.append({
                            "video_name": entry["video_name"],
                            "start_frame": entry["start_frame"],
                            "end_frame": entry["end_frame"],
                            "text": entry["text"]
                        })

    def _initialize_bm25(self):
        """Initialize the BM25 model with the loaded documents."""
        self.bm25 = BM25Okapi(self.documents)

    def _preprocess(self, text):
        """Preprocess Vietnamese text by segmenting and removing stopwords."""
        text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
        tokens = ViTokenizer.tokenize(text).split()  # Segment Vietnamese text
        tokens = [word for word in tokens if word.lower() not in self.stopwords]  # Remove stopwords
        return tokens

    def search(self, input_query, top_k=5):
        """Search for the most relevant documents using BM25 and return a dictionary of keyframe paths and scores."""
        if self.bm25 is None:
            raise ValueError("BM25 model is not initialized.")

        query_tokens = self._preprocess(input_query)
        scores = self.bm25.get_scores(query_tokens)
        top_n_indices = scores.argsort()[-top_k:][::-1]  # Get indices of top_k highest scores

        results = {}
        for idx in top_n_indices:
            video_name = self.metadata[idx]["video_name"]
            key_frames = self.metadata[idx]["key_frames"]
            score = scores[idx]
            # Construct the path for each key frame
            for key_frame in key_frames:
                image_path = f"./distilled_keyframe/{video_name[:3]}/{video_name[4:]}/{key_frame}"
                # Store the score in the dictionary
                results[image_path] = score

        return results


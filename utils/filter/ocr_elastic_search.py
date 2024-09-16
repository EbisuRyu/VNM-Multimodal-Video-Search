from elasticsearch import Elasticsearch, RequestError
from utils.filter.utils import result_format


class ElasticSearch:
    def __init__(self, index_name, user_name, password, es_host="http://localhost:9200",  request_timeout=60):
        self.index_name = index_name
        self.es = Elasticsearch(
            hosts=es_host, 
            basic_auth=(user_name, password), 
            request_timeout=request_timeout
        )

    def search(self, input_query, image_path_subset=None, top_k=10):
        """Tìm kiếm các document dựa trên một truy vấn và trả về top K kết quả."""
        if image_path_subset is not None:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "TEXT": input_query
                                }
                            }
                        ],
                        "filter": {
                            "ids": {
                                "values": image_path_subset
                            }
                        }
                    }
                }
            }
        else:
            query = {
                "query": {
                    "match": {
                        "TEXT": input_query
                    }
                }
            }

        try:
            response = self.es.search(index=self.index_name, body=query, size=top_k)
            image_paths = [result['_id']for result in response['hits']['hits']]
            scores = [result['_score'] for result in response['hits']['hits']]
            return result_format(image_paths, scores)
        except RequestError as e:
            print(f"Error searching documents: {e.info}")

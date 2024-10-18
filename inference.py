from utils.system_call.embedding_space import EmbeddingSpace
from utils.system_call.metadata_space import MetadataSpace
from utils.query_processing.translator import Translator
from utils.system_call.utils import display_images
from langdetect import detect


class MetadataSpaceInterface:
    def handle_metadata_fusion_mode(self, searching_system, translator):
        fusion_mode = bool(int(input('Enter fusion mode status: ')))
        searching_system.metadata_space.update_fusion_mode(fusion_mode)
        searching_system.current_embedding_result_subset = list(searching_system.embedding_space.current_result.keys()) if fusion_mode else None

    def handle_metadata_delete_history(self, searching_system, translator):
        searching_system.metadata_space.delete_history()

    def handle_metadata_searching(self, searching_system, translator):
        while True:
            result = searching_system.metadata_space.search(
                asr_query=input('ASR query: ') or None,
                ocr_query=input('OCR query: ') or None,
                tag_query=input('Tag query: ') or None,
                oclass_queries={
                    'object_class': input('Object class: ') or None,
                    'color_class': input('Color class: ') or None,
                    'object_number': input('Object number: ') or None
                },
                bbox_queries={
                    'object_bbox': input('Object bounding box: ') or None,
                    'color_bbox': input('Color bounding box: ') or None
                },
                embedding_result_subset=searching_system.current_embedding_result_subset,
                top_k=int(input('Enter the number of results to return: '))
            )
            searching_system.current_metadata_result_subset = list(result.keys())
            display_images(result)
            if int(input('Do you want to end the search process? ')):
                break


class EmbeddingSpaceInterface:
    def handle_embedding_fusion_mode(self, searching_system, translator):
        fusion_mode = bool(int(input('Enter fusion mode status: ')))
        searching_system.embedding_space.update_fusion_mode(fusion_mode)
        searching_system.current_metadata_result_subset = list(searching_system.metadata_space.current_result.keys()) if fusion_mode else None

    def handle_embedding_video_local_update(self, searching_system, translator):
        video_local = {
            'L': input('Enter video part: ') or None,
            'V': input('Enter video ID: ') or None
        }
        searching_system.embedding_space.update_video_local(video_local)

    def handle_embedding_model_update(self, searching_system, translator):
        use_model = {
            'clip_h14_engine': bool(int(input('Use CLIP-H14: '))),
            'clip_h14_xlm_engine': bool(int(input('Use CLIP-H14-xlm: '))),
            'clip_l14_engine': bool(int(input('Use CLIP-L14: '))),
            'blip_vit_engine': bool(int(input('Use ViT BLIP: '))),
            'blip_pretrain_engine': bool(int(input('Use pretrain BLIP: '))),
            'beit_base_engine': bool(int(input('Use Base BEIT: '))),
            'beit_large_engine': bool(int(input('Use Large BEIT: ')))
        }
        searching_system.embedding_space.update_model(use_model)

    def handle_embedding_delete_history(self, searching_system, translator):
        searching_system.embedding_space.delete_history()

    def handle_embedding_back_to_before_result(self, searching_system, translator):
        searching_system.embedding_space.back_to_before_result(int(input('Enter the index of the previous result: ')))

    def handle_embedding_user_feedback(self, searching_system, query_text):
        def collect_keyframes(prompt):
            keyframes = []
            while (user_input := input(prompt)) != "":
                keyframes.append(user_input)
            return keyframes

        pos_keyframe_subset = collect_keyframes("Enter positive keyframe path (or press Enter to finish): ")
        neg_keyframe_subset = collect_keyframes("Enter negative keyframe path (or press Enter to finish): ")

        reranked_result = searching_system.embedding_space.feedback(
            query_text=query_text,
            pos_keyframe_subset=pos_keyframe_subset,
            neg_keyframe_subset=neg_keyframe_subset
        )
        return reranked_result

    def handle_metadata_tag_recommendation(self, searching_system, text_query_1, text_query_2):
        # Tag Recommendation
        text_input = text_query_1 + ' ' + text_query_2 if text_query_2 else text_query_1
        searching_system.metadata_space.tag_recommend(text_input)

    def handle_image_similarity(self, searching_system, translator):
        result = searching_system.embedding_space.image_similarity(
            query_image_path=input('Enter query image path: '),
            top_k=int(input('Enter the number of results to return: '))
        )
        display_images(result)
        searching_system.current_embedding_result_subset = list(result.keys())

    def handle_embedding_searching(self, searching_system, translator):
        while True:
            text_query_1 = input('Enter query 1: ')
            text_query_2 = input('Enter query 2: ') or None
            if detect(text_query_1) == 'vi':
                text_query_1 = translator(text_query_1).lower()
            if text_query_2 and detect(text_query_2) == 'vi':
                text_query_2 = translator(text_query_2).lower()
            result = searching_system.embedding_space.search(
                query_text_1=text_query_1.lower(),
                query_text_2=text_query_2.lower() if text_query_2 else None,
                metadata_result_subset=searching_system.current_metadata_result_subset,
                top_k=int(input('Enter the number of results to return: '))
            )
            display_images(result)
            searching_system.current_embedding_result_subset = list(result.keys())
            # Tag Recommendation
            self.handle_metadata_tag_recommendation(
                searching_system=searching_system,
                text_query_1=text_query_1,
                text_query_2=text_query_2
            )
            if int(input('Do you want to use user feedback? ')):
                reranked_result = self.handle_embedding_user_feedback(
                    searching_system=searching_system,
                    query_text=text_query_1
                )
                display_images(reranked_result)
            if int(input('Do you want to end the search process? ')):
                break


class SearchingSystem:
    def __init__(self):
        self.embedding_space = EmbeddingSpace(
            use_clip_h14=True,
            use_clip_h14_xlm=False,
            use_clip_l14=False,
            use_blip_vit=False,
            use_blip_pretrain=False, 
            use_base_beit=False,
            use_large_beit=False
        )
        self.metadata_space = MetadataSpace()
        self.current_embedding_result_subset = None
        self.current_metadata_result_subset = None

    def show_state(self):
        print('-----------------Embedding Space-----------------')
        print(f'Fusion: {self.embedding_space.fusion}')
        print(f'Video Local: {self.embedding_space.video_local}')
        print(f'Using Model: {self.embedding_space.use_model}')
        print(f'History: {self.embedding_space.search_history}')
        print('-----------------Metadata Space-----------------')
        print(f'Fusion: {self.metadata_space.fusion}')
        print(f'Current Result: {bool(self.metadata_space.current_result)}')
        print(f'Tag Recommendation: {self.metadata_space.tag_recommendation}')


def handle_action(actions, searching_system, translator):
    action = input(get_action_script(actions))
    if action == 'exit':
        return
    actions.get(action, handle_invalid_action)(searching_system, translator)
    searching_system.show_state()


def get_action_script(actions):
    if len(actions) == 3:
        return "Available actions:\n  1. Fusion Mode.\n  2. Delete history.\n  3. Search\n"
    return "Available actions:\n  1. Fusion Mode.\n  2. Update video local.\n  3. Update model.\n  4. Delete history.\n  5. Back to previous result.\n  6. Search\n  7. Image Similarity\n"


def handle_invalid_action(searching_system, translator):
    print("Invalid action. Please select again.")


def main():
    searching_system = SearchingSystem()
    translator = Translator()
    embedding_interface = EmbeddingSpaceInterface()
    metadata_interface = MetadataSpaceInterface()

    embedding_space_actions = {
        '1': embedding_interface.handle_embedding_fusion_mode,
        '2': embedding_interface.handle_embedding_video_local_update,
        '3': embedding_interface.handle_embedding_model_update,
        '4': embedding_interface.handle_embedding_delete_history,
        '5': embedding_interface.handle_embedding_back_to_before_result,
        '6': embedding_interface.handle_embedding_searching,
        '7': embedding_interface.handle_image_similarity
    }

    metadata_space_actions = {
        '1': metadata_interface.handle_metadata_fusion_mode,
        '2': metadata_interface.handle_metadata_delete_history,
        '3': metadata_interface.handle_metadata_searching
    }

    while True:
        space = input(
            "Which space do you want to use:\n  1. Metadata Space.\n  2. Embedding Space.\n")
        if space == '1':
            handle_action(metadata_space_actions, searching_system, translator)
        elif space == '2':
            handle_action(embedding_space_actions, searching_system, translator)
        else:
            print("Invalid selection. Please choose 1 or 2.")


if __name__ == "__main__":
    main()

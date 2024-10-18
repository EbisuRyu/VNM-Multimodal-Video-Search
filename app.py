from utils.system_call.embedding_space import EmbeddingSpace
from utils.system_call.metadata_space import MetadataSpace
from utils.query_processing.translator import Translator
from langdetect import detect
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for
import pandas as pd
import io
import os
import cv2
import time
import json
import re

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

id_export = 0

class SearchingSystem:
    def __init__(self, use_clip_h14=False, use_clip_h14_xlm=False, use_clip_l14=False, use_blip_vit=True, use_blip_pretrain=True, use_base_beit=False, use_large_beit=False):
        self.embedding_space = EmbeddingSpace(
            use_clip_h14=use_clip_h14,
            use_clip_h14_xlm=use_clip_h14_xlm,
            use_clip_l14=use_clip_l14,
            use_blip_vit=use_blip_vit,
            use_blip_pretrain=use_blip_pretrain,
            use_base_beit=use_base_beit,
            use_large_beit=use_large_beit
        )
        self.metadata_space = MetadataSpace()
        self.current_embedding_result_subset = None
        self.current_metadata_result_subset = None

model_list = [False, False, False, False, False, False, False, False]  # cliph14, cliph14xlm, clipl14, blipvit, blippretrain, basebeit, largebeit
searching_system = None
translator = Translator()

okresponse = {
    'status': 'ok'
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/select_model', methods=['POST'])
def select_model():
    global searching_system, model_list
    data = request.get_json()
    model_list = data['query']  # list bool [cliph14, cliph14xlm, clipl14, blipvit, blippretrain, basebeit, largebeit]
    searching_system = SearchingSystem(
        use_clip_h14=model_list[0],
        use_clip_h14_xlm=model_list[1],
        use_clip_l14=model_list[2],
        use_blip_vit=model_list[3],
        use_blip_pretrain=model_list[4],
        use_base_beit=model_list[5],
        use_large_beit=model_list[6]
    )
    return jsonify(okresponse)


@app.route('/search')
def search():
    searching_system.metadata_space.delete_history()
    searching_system.embedding_space.delete_history()

    return render_template('search.html', model_list=model_list)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded!"})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file!"})
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    model_info_str = request.form['model_info']
    model_info = json.loads(model_info_str)
    # image similarity [topk, clip_h14, clip_h14_xlm, clip_l14, blip_vit, blip_pretrain, beit_base, beit_large]
    searching_system.embedding_space.update_model({
        'clip_h14_engine': bool(model_info[1]),
        'clip_h14_xlm_engine': bool(model_info[2]),
        'clip_l14_engine': bool(model_info[3]),
        'blip_vit_engine': bool(model_info[4]),
        'blip_pretrain_engine': bool(model_info[5]),
        'beit_base_engine': bool(model_info[6]),
        'beit_large_engine': bool(model_info[7])
    })
    result = searching_system.embedding_space.image_similarity(
        query_image_path=file_path,
        top_k=100 if model_info[0] == '' else int(model_info[0])
    )
    searching_system.current_embedding_result_subset = list(result.keys())

    result_with_index = {i: [key, float(value)] for i, (key, value) in enumerate(result.items())}  # add index to result_info {0: [key, value], 1: [key, value], ...}
    return jsonify(result_with_index)

@app.route('/process_query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data['query']
    typeSearch = query[0]
    print("Processing query...")
    print(query)
    if typeSearch == 'metadata':
        if query[1] == 1:  # option = 1: delete history
            searching_system.metadata_space.delete_history()
            print('delete meta')
            return jsonify(okresponse)

        if query[1] == 2:  # option = 2: search [type, option, topk, fussion, ocr, tag, object_class, color_class, object_bbox, object_number, color_bbox, asr]
            searching_system.metadata_space.update_fusion_mode(bool(query[3]))
            searching_system.current_embedding_result_subset = list(searching_system.embedding_space.current_result.keys()) if bool(query[3]) else None

            print(bool(query[3]))
            result_info = searching_system.metadata_space.search(
                asr_query=None if query[11] == '' else query[11],
                ocr_query=None if query[4] == '' else query[4],
                tag_query=None if query[5] == '' else query[5],
                oclass_queries={
                    'object_class': None if query[6] == '' else query[6],
                    'color_class': None if query[7] == '' else query[7],
                    'object_number': None if query[9] == '' else query[9]
                },
                bbox_queries={
                    'object_bbox': None if query[8] == '' else query[8],
                    'color_bbox': None if query[10] == '' else query[10]
                },
                embedding_result_subset=searching_system.current_embedding_result_subset,
                top_k=100 if query[2] == '' else int(query[2])
            )

            searching_system.current_metadata_result_subset = list(result_info.keys())

            print(result_info)

            result_with_index = {i: [key, float(value)] for i, (key, value) in enumerate(result_info.items())}  # add index to result_info {0: [key, value], 1: [key, value], ...}
            return jsonify(result_with_index)

        if query[1] == 4:  # option = 4: get tag recommend [type, option, query_current, query_next]
            # Tag Recommendation
            text_input = query[2].lower() + ' ' + \
                query[3].lower() if query[3] else query[2].lower()
            searching_system.metadata_space.tag_recommend(text_input)
            tag_recommend = searching_system.metadata_space.tag_recommendation
            print(tag_recommend)
            return jsonify(tag_recommend)

    elif typeSearch == 'embedding':
        if query[1] == 1:  # option = 1: delete history [type, option]
            searching_system.embedding_space.delete_history()
            print('delete embedding')
            return jsonify(okresponse)

        if query[1] == 2:  # option = 2: search [type, option, topk, fussion, query_text_1, query_text_2, local_L, local_V, clip_h14, clip_h14_xlm, clip_l14, blip_vit, blip_pretrain, beit_base, beit_large]
            startTime = time.time()

            searching_system.embedding_space.update_fusion_mode(bool(query[3]))
            searching_system.current_metadata_result_subset = list(searching_system.metadata_space.current_result.keys()) if bool(query[3]) else None

            searching_system.embedding_space.update_video_local({
                'L': None if query[6] == '' else query[6],
                'V': None if query[7] == '' else query[7]
            })

            searching_system.embedding_space.update_model({
                'clip_h14_engine': bool(query[8]),
                'clip_h14_xlm_engine': bool(query[9]),
                'clip_l14_engine': bool(query[10]),
                'blip_vit_engine': bool(query[11]),
                'blip_pretrain_engine': bool(query[12]),
                'beit_base_engine': bool(query[13]),
                'beit_large_engine': bool(query[14])
            })

            if detect(query[4]) == 'vi':
                query[4] = translator(query[4]).lower()
            if query[5] and detect(query[5]) == 'vi':
                query[5] = translator(query[5]).lower()

            result_info = searching_system.embedding_space.search(
                query_text_1=query[4].lower(),
                query_text_2=query[5].lower() if query[5] else None,
                metadata_result_subset=searching_system.current_metadata_result_subset,
                top_k=100 if query[2] == '' else int(query[2])
            )
            searching_system.current_embedding_result_subset = list(result_info.keys())
            print(result_info)
            result_with_index = {i: [key, float(value)] for i, (key, value) in enumerate(result_info.items())}  # add index to result_info {0: [key, value], 1: [key, value], ...}

            return jsonify(result_with_index)

        if query[1] == 4:  # option = 4: back to previous result [type, option, index]
            searching_system.embedding_space.back_to_before_result(
                int(query[2]))

            result_info = searching_system.embedding_space.current_result
            print('return')
            result_with_index = {i: [key, float(value)] for i, (key, value) in enumerate(result_info.items())}  # add index to result_info {0: [key, value], 1: [key, value], ...}
            return jsonify(result_with_index)

        if query[1] == 5:  # option = 5: user feedback [type, option, query_text_1, pos_keyframe_subset, neg_keyframe_subset]
            reranked_result = searching_system.embedding_space.feedback(
                query_text=query[2],
                pos_keyframe_subset=query[3],
                neg_keyframe_subset=query[4]
            )
            print('feedback')
            searching_system.current_embedding_result_subset = list(reranked_result.keys())
            result_with_index = {i: [key, float(value)] for i, (key, value) in enumerate(reranked_result.items())}  # add index to result_info {0: [key, value], 1: [key, value], ...}
            return jsonify(result_with_index)

        if query[1] == 6:  # option = 6: image similarity [type, option, image_path, topk, clip_h14, clip_h14_xlm, clip_l14, blip_vit, blip_pretrain, beit_base, beit_large]
            searching_system.embedding_space.update_model({
                'clip_h14_engine': bool(query[4]),
                'clip_h14_xlm_engine': bool(query[5]),
                'clip_l14_engine': bool(query[6]),
                'blip_vit_engine': bool(query[7]),
                'blip_pretrain_engine': bool(query[8]),
                'beit_base_engine': bool(query[9]),
                'beit_large_engine': bool(query[10])
            })

            result = searching_system.embedding_space.image_similarity(
                query_image_path=query[2],
                top_k=100 if query[3] == '' else int(query[3])
            )

            searching_system.current_embedding_result_subset = list(result.keys())

            result_with_index = {i: [key, float(value)] for i, (key, value) in enumerate(result.items())}  # add index to result_info {0: [key, value], 1: [key, value], ...}
            return jsonify(result_with_index)

    elif typeSearch == "video": # ['video', 1, L01_V001]
        # get frame rate
        map_keyframe = os.path.join('static', 'map-keyframes', query[2] + '.csv')
        df = pd.read_csv(map_keyframe)
        fps = df['fps'][0]
        print(fps)

        # get video path or youtube link
        video_path = os.path.join('static', 'video', query[2]) + '.mp4'
        if not os.path.exists(video_path): # open on youtube
            with open('static/media-info/' + query[2] + '.json', 'r', encoding='utf-8') as f:
                info_file = json.load(f)
                return jsonify('youtube', fps, info_file['watch_url'])
        
        # open local video
        local_video = '../' + video_path
        return jsonify('local_video', fps, local_video)
    
    elif typeSearch == "keyframe": # ['keyframe', local_L, local_V]
        dir_path = os.path.join('static', 'distilled_keyframe', query[1], query[2])
        keyframe_list = os.listdir(dir_path)
        keyframe_list.sort(key=lambda filename: int(re.search(r'(\d+)', filename).group(0)))

        result = {i: ['/distilled_keyframe/'+query[1]+'/'+query[2]+'/'+jpg, 1] for i, jpg in enumerate(keyframe_list)}

        return jsonify(result)
    return jsonify(okresponse)

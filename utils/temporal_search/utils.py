def get_video_frame(keyframe_path):
    path_list = keyframe_path.split('/')
    video_id = '/'.join([path_list[-3], path_list[-2]])
    frame_id = int(path_list[-1][:-4])
    return video_id, frame_id


class Metadata:
    def __init__(self, video_id: str, keyframe_id: str, score: int, type: int, next_keyframe_isok=0):
        self.video_id = video_id
        self.keyframe_id = keyframe_id
        self.score = score
        self.type = type
        self.next_keyframe_isok = next_keyframe_isok

    def __lt__(self, other_keyframe):
        return ((self.video_id < other_keyframe.video_id) or
                (self.video_id == other_keyframe.video_id and self.keyframe_id < other_keyframe.keyframe_id) or
                (self.video_id == other_keyframe.video_id and self.keyframe_id == other_keyframe.keyframe_id and self.type < other_keyframe.type))

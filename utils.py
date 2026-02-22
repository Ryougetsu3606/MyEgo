from decord import VideoReader, cpu
from PIL import Image

import numpy as np
import logging
import json
import os

def load_video(video, args, start_time=0, end_time=None):
    # FIXME: check video type
    if not video.endswith(".mp4"):
        video = video + ".mp4"

    sampled_fps = args.fps
    frame_num = args.frame_num
    resolution = args.resolution
    video_path = args.video_path
    video = os.path.join(video_path, video)
    # support cut the video to a specific interval
    
    vr = VideoReader(video, ctx=cpu(0))
    meta_fps = vr.get_avg_fps()

    # sample `fps` frames per second
    if frame_num is not None:
        max_frame_num = min(frame_num, len(vr) / meta_fps) # FIXME: hard-coding, sample at most 1 fps
        frame_indices = np.linspace(0, len(vr) - 1, max_frame_num, dtype=int) # uni-sample
        frame_indices = frame_indices.tolist()
    else:
        start_index = int(start_time * meta_fps)
        end_index = int(end_time * meta_fps) if end_time is not None else len(vr)
        frame_indices = range(start_index, end_index, int(meta_fps / sampled_fps))

    frame_list = vr.get_batch(frame_indices).asnumpy()
    resized_frames_list = []

    # resize the short side to `resolution`
    for frame in frame_list:
        h, w, _ = frame.shape
        if resolution is not None:
            if h > w:
                frame = Image.fromarray(frame).resize((int(h * resolution / w), resolution))
            else:
                frame = Image.fromarray(frame).resize((resolution, int(w * resolution / h)))
        resized_frames_list.append(np.array(frame))

    logging.info(f"shape: {resized_frames_list[0].shape}, video length: {len(resized_frames_list)}")
    return resized_frames_list

def save_qa(qas, file_path):
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(qas, ensure_ascii=False) + "\n")

def load_qa(file_path):
    with open(file_path, "r") as f:
        qa_list = json.load(f)
    
    # group the qa by video
    video_qa_dict = {} # Dict{video: List[qa]}
    for qa in qa_list:
        video = qa["video"]
        if video not in video_qa_dict:
            video_qa_dict[video] = []
        video_qa_dict[video].append(qa)
    
    return video_qa_dict



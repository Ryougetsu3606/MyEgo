import numpy as np
import torch
import torch.nn as nn

from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

from PIL import Image

from utils import load_video, load_qa, save_qa
from prompts import ego_qa_prompt_sys

import os
import json
import argparse
import time
import logging
from tqdm import tqdm

def qa_with_frames(question, input_frames, args):
    input_frames = [Image.fromarray(frame) for frame in input_frames]
    messages = [
        {
            "role": "system",
            "content": ego_qa_prompt_sys,
        },
        {
            "role": "user",
            "content": [
                {"video": input_frames, "fps": args.fps},
                {"type": "text", "text": f"Please answer the question: {question}"},
            ]
        }
    ]

    # Copy from Qwen3-VL official
    # TODO: Rewrite into batch process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, image_patch_size=16, return_video_metadata=True)
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
    inputs = inputs.to(args.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
        
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def infer_single_video(
        video,
        qas,
        args,
    ):
    
    ans_qas = []
    window_size, fps = args.window_size, args.fps

    for i, qa in enumerate(qas):
        question, t_q = qa.get("question", None), qa.get("question_moment", None)
        if question is not None and t_q is not None:
            input_frames = load_video(video=video, args=args, start_time=0, end_time=t_q)
            try:
                response = qa_with_frames(question=question, input_frames=input_frames, args=args)
            except Exception as e:
                logging.error(f"Error in video {video}, question {question}, t_q {t_q}: {e}")
                response = ""
            ans_qa = {
                "question": question,
                "question_moment": t_q,
                "answer": qa.get("answer", None),
                "response": response,
            }
            ans_qas.append(ans_qa)

    assert len(ans_qas) == len(qas), f"Only {len(ans_qas)} questions answered, but {len(qas)} questions in total"
    return ans_qas

def infer_dataset(args):
    global model, processor
    
    logging.info(f"Loading base model from {args.mllm_path}...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.mllm_path, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(args.mllm_path, trust_remote_code=True)


    model.eval()

    video_qa_dict = load_qa(file_path=args.dataset_path)

    if args.subset_idx_path is not None:
        _video_qa_dict = {}
        with open(args.subset_idx_path, 'r', encoding='utf-8') as f:
            subset_idx = json.load(f)
        for video, qas in video_qa_dict.items():
            new_qas = [qa for qa in qas if qa.get("index", -1) in subset_idx]
            _video_qa_dict[video] = new_qas
        video_qa_dict = _video_qa_dict

    if os.path.exists(args.output_path):
        existed_videos = []
        save_path = os.path.join(args.output_path, f"{args.mllm_path.split('/')[-1]}_myego_fps{args.fps}_fn{args.frame_num}_res{args.resolution}.json")
        with open(save_path, "r") as f:
            for line in f:
                existed_videos.append(json.loads(line).get("video", None))
        
        video_qa_dict = {video: qas for video, qas in video_qa_dict.items() if video not in existed_videos}
    logging.info(f"Start inference on {len(video_qa_dict)} videos")

    for video, qas in tqdm(video_qa_dict.items()):
        start_time = time.time()
        ans_qas = infer_single_video(video=video, qas=qas, args=args)
        end_time = time.time()

        logging.info(f"video {video} done. {len(qas)} questions answered in {end_time - start_time:.2f} seconds")
        ans_qas = {"video": video, "qa_pairs": ans_qas}
        save_qa(ans_qas, args.output_path)
        logging.info(f"Successfully save {len(ans_qas)} questions to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data/myego.json")
    parser.add_argument("--video_path", type=str, default="./data/videos")
    parser.add_argument("--mllm_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output_path", type=str, default="./result")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=int, default=1) # only support fps=1 for now
    parser.add_argument("--frame_num", type=int, default=16) # uniform sampling
    parser.add_argument("--resolution", type=int, default=336)

    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    infer_dataset(args)
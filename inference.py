#!/usr/bin/env python3

import os
import sys
import argparse
import json
import warnings

import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import SYSTEM_PROMPT, OPEN_ENDED_SYSTEM_PROMPT, build_mc_prompt, build_oe_prompt


def extract(response: str) -> str:
    marker = "</think>\n\n"
    idx = response.find(marker)
    if idx != -1:
        return response[idx + len(marker):].strip()
    return response


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL Evaluation")
    parser.add_argument("--qa_json", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--qa_format", type=str, choices=["mc", "oe"], default="mc")
    return parser.parse_args()


def load_video(video_path: str, timestamp: float, max_frames_num: int = 32) -> list:

    vr = VideoReader(video_path, ctx=cpu(0))
    max_idx = len(vr) - 1
    
    target_end_idx = int(vr.get_avg_fps() * timestamp)
    target_end_idx = min(max_idx, target_end_idx)
    
    num_frames = int(min(max_frames_num, max(1, timestamp * 1)))
    num_frames = min(num_frames, target_end_idx + 1)
    
    uniform_sampled_frames = np.linspace(
        0, target_end_idx, num_frames, dtype=int
    )
    
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    
    resize_frames = []
    for frame in spare_frames:
        resize_frames.append(np.array(Image.fromarray(frame).resize((384, 384))))
        
    return resize_frames


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    warnings.filterwarnings("ignore")
    
    print(f"QA JSON: {args.qa_json}")
    print(f"Video Dir: {args.video_dir}")
    print(f"Model Path: {args.model_path}")
    print(f"QA Format: {args.qa_format}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path)
    save_path = os.path.join(args.save_dir, f"{model_name}_{args.qa_format}.json")
    
    thinking = bool("Thinking" in model_name)
    MAX_NEW_TOKENS = 4096 if thinking is True else 512
    
    print(f"Loading model: {args.model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()
    
    with open(args.qa_json, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    
    results = {}
    print(f"Total to process: {len(qa_pairs)}")
    
    for i, qa_pair in enumerate(qa_pairs):
        qid = str(qa_pair.get("question_id", i))
        
        video_id = qa_pair.get("video_id", qa_pair.get("video_path", ""))
        video_path = os.path.join(args.video_dir, video_id + ".mp4")
        
        question = qa_pair.get("question", "")
        options = qa_pair.get("options", qa_pair.get("choices", []))
        
        try:
            video_frames = load_video(video_path, qa_pair["question_moment"], args.max_frames)
            video_frames = [Image.fromarray(frame) for frame in video_frames]
            
            if args.qa_format == "oe":
                system_prompt = OPEN_ENDED_SYSTEM_PROMPT
                question_prompt = build_oe_prompt(question)
            else:
                system_prompt = SYSTEM_PROMPT
                question_prompt = build_mc_prompt(question, options)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_frames},
                        {"type": "text", "text": question_prompt}
                    ],
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True, image_patch_size=16, return_video_metadata=True
            )
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
            else:
                video_metadatas = None
            
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                video_metadata=video_metadatas, **video_kwargs,
                do_resize=False, return_tensors="pt"
            )
            inputs = inputs.to(model.device)
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            
            if thinking is True:
                response = extract(response)

            if args.qa_format == 'mc':
                results[qid] = response
            
            else:
                results[qid] = {"question": question, "answer": qa_pair.get("answer", ""), "model_response": response}
            
        except Exception as e:
            results[qid] = f"Error: {str(e)}"
        
        print(f"[{i+1}/{len(qa_pairs)}] QID={qid}: {results.get(qid, 'N/A')}")
    
    json.dump(results, open(save_path, "w", encoding="utf-8"), indent=2)
    print(f"\nDone! Results saved to {save_path}")


if __name__ == "__main__":
    main()

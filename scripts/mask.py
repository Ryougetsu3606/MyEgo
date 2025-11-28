import cv2
import numpy as np
import os
import glob
import cv2
import ffmpeg
from tqdm import tqdm
import argparse
def process_video_with_mask(input_path, output_path, center, radius):
    try:
        probe = ffmpeg.probe(input_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps_str = video_stream.get('r_frame_rate', '25/1')
        num, den = map(int, fps_str.split('/'))
        fps = num / den if den != 0 else 25.0

    except ffmpeg.Error as e:
        print(f"skip: ffprobe error {input_path}. Error: {e.stderr.decode('utf-8', errors='ignore')}")
        return

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    input_process = (
        ffmpeg
        .input(input_path)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    
    output_process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', r=fps)
        .output(
            output_path, 
            pix_fmt='yuv420p', 
            vcodec='libx264',
            preset='slow',
            crf=19
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=True)
    )
    
    try:
        while True:
            in_bytes = input_process.stdout.read(width * height * 3)
            if not in_bytes:
                break
            
            if len(in_bytes) != width * height * 3:
                continue

            in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            masked_frame = cv2.bitwise_and(in_frame, in_frame, mask=mask)
            
            output_process.stdin.write(masked_frame.tobytes())

    except BrokenPipeError:
        print("BrokenPipe Error")
    
    finally:
        if output_process.stdin:
            output_process.stdin.close()

        inp_stderr_data = input_process.stderr.read()
        out_stderr_data = output_process.stderr.read()

        input_process.wait()
        output_process.wait()

        if input_process.returncode != 0:
            print(inp_stderr_data.decode('utf-8', errors='ignore'))

        if output_process.returncode != 0:
            print(out_stderr_data.decode('utf-8', errors='ignore'))

def main(args):
    INPUT_FOLDER = args.input_folder
    OUTPUT_FOLDER = args.output_folder
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    all_video_files = glob.glob(os.path.join(INPUT_FOLDER, '*.mp4'))
    
    video_files = [f for f in all_video_files if os.path.basename(f).startswith('02')] # EgoLife prefix

    
    circle_center, circle_radius = (704, 704), 700

    for video_path in tqdm(video_files, desc="Processing"):
        filename = os.path.basename(video_path)
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        process_video_with_mask(video_path, output_path, circle_center, circle_radius)
        
    print("\nDone")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply mask to videos")
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()
    
    main(args)
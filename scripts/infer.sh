export CUDA_VISIBLE_DEVICES=0
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
FPS=1
FRAME_NUM=32
RESOLUTION=336

model_name=Qwen/Qwen3-VL-4B-Instruct

echo "model name: $model_name"
echo "Evaluating with FPS=$FPS, WINDOW_SIZE=$WINDOW_SIZE, RESOLUTION=$RESOLUTION"
python3 -m inference \
    --fps $FPS \
    --frame_num $FRAME_NUM \
    --resolution $RESOLUTION \
    --video_path <path_to_videos> \
    --dataset_path ./data/myego.json \
    --mllm_path $model_name \

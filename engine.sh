CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct --dtype auto \
 --port 10005 --max-model-len 40730 --tensor-parallel-size 4 \
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /mnt/hdd_raid5/gaoch/Llama-3.1-8B-Instruct \
#  --dtype auto --port 10005 --max-model-len 40730 --tensor-parallel-size 4
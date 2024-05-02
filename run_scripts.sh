
# Models
# facebook/opt-125m
# lmsys/vicuna-7b-v1.1


# Quantize model
nohup python -u test_model_quant.py --model_name facebook/opt-125m --b 4 > ./log/out_quant.txt &

# Evaluate model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup accelerate launch -m lm_eval --model hf --model_args pretrained=facebook/opt-125m --tasks mmlu --batch_size 16  > ./log/out_eval_fp.txt &
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup accelerate launch -m lm_eval --model hf --model_args pretrained=./models/quant/vicuna-7b-v1.1 --tasks mmlu --batch_size 16  > ./log/out_eval_q.txt &



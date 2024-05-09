
# Models

cache_path="/mnt/disk5/maktukmak/models"  #/mnt/disk5/maktukmak/models, ./models

#export http_proxy=http://proxy-chain.intel.com:911
#export https_proxy=https://proxy-chain.intel.com:912
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Quantize model
model_path="facebook"   #facebook, lmsys
model="opt-125m"  #opt-125m, vicuna-7b-v1.1
nohup python -u test_model_quant.py --calib --model_name $model_path/$model --cache_path $cache_path --b 4 > ./log/out_quant.txt &
nohup python -u test_model_quant.py --model_name $model_path/$model --cache_path $cache_path --b 4 > ./log/out_quant.txt &

# Evaluate model
model_path=$cache_path/quant 
model="opt-125m"  #opt-125m, vicuna-7b-v1.1
nohup accelerate launch -m lm_eval --model hf --model_args pretrained=$model_path/$model --tasks mmlu --batch_size 16  > ./log/out_eval.txt &
nohup python test_model_eval_visual.py --model_name $model_path/$model > ./log/out_eval.txt &
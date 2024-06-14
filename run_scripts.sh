cache_path="/mnt/disk5/maktukmak/models"  #/mnt/disk5/maktukmak/models, ./models

model_path="lmsys"   #facebook, lmsys
model="vicuna-7b-v1.1"  #opt-125m, vicuna-7b-v1.1
#export CUDA_VISIBLE_DEVICES=0,1

declare -a arr=(
                "uniform snr" 
                "uniform minmax"
                # "uniform iterative"
                # "nonuniform iterative"
                "nonuniform snr" 
                "nonuniform quantile"
                # "float iterative" 
                "float snr" 
                "float minmax"
                )


for i in "${arr[@]}"
do
    set -- $i
    python -u model_quant.py\
                    --model_name $model_path/$model\
                    --cache_path $cache_path\
                    --qtype=$1\
                    --alg=$2\
                    --b 2\
                    --decompose_outlier &&
                    #--block_size 16 &&
                    
                    

    #nohup accelerate launch -m lm_eval --model hf --model_args pretrained=$model_path/$model --tasks mmlu --batch_size 16  > ./log/out_eval.txt &
    #python -m lm_eval --model hf --model_args pretrained=$cache_path/quant/$model --tasks mmlu --batch_size 16  > ./log/out_eval.txt &&
    python -m lm_eval --model hf --model_args pretrained=$cache_path/quant/$model --tasks mmlu --batch_size 16
    # nohup python test_model_eval_visual.py --model_name $model_path/$model > ./log/out_eval.txt &
done
exit
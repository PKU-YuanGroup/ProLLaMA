# the codes are based on Chinese-LLaMA-Alpaca-2
# Read the wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh) carefully before running the script
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="continual pretraining"
lr=5e-5
lora_rank=128
lora_alpha=256
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=GreatCaptainNemo/ProLLaMA_Stage_1 #or your local path
dataset_dir=./pretraining_dataset #your dataset path
data_cache=/remote-home/share/uniref50/cache/
per_device_train_batch_size=4
gradient_accumulation_steps=8
block_size=2048
output_dir=./output_dir/

deepspeed_config_file=ds_zero2_no_offload.json
torchrun --nproc_per_node 8 --rdzv-endpoint=localhost:29500 pretrain.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.1 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --seed 42 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 400 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --load_in_kbits 16 \
    --bf16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --flash_attn \
    --modules_to_save ${modules_to_save} \

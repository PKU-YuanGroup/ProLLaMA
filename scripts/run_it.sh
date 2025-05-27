# the codes are based on Chinese-LLaMA-Alpaca-2
# Read the wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh) carefully before running the script
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="instruction_tuning"
lr=5e-5
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
lora_dropout=0.05

pretrained_model=GreatCaptainNemo/ProLLaMA_Stage_1 # or your local path

# Optionally specify dataset_dir, train_file and validation_file (JSON format)
# If train_file is defined, dataset_dir will be ignored.
dataset_dir=./instruction_tuning_dataset
train_file=./train.json
validation_file=./validation.json

per_device_train_batch_size=144
gradient_accumulation_steps=4
max_seq_length=256
output_dir=save_dir/
deepspeed_config_file=ds_zero2_no_offload.json

# Build additional arguments for optional files and dataset directory
extra_args=""

# If train_file is defined, it takes precedence over dataset_dir
if [ -n "${train_file}" ]; then
    extra_args="${extra_args} --train_file ${train_file}"
elif [ -n "${dataset_dir}" ]; then
    extra_args="${extra_args} --dataset_dir ${dataset_dir}"
fi

if [ -n "${validation_file}" ]; then
    extra_args="${extra_args} --validation_file ${validation_file}"
    extra_args="${extra_args} --do_eval --evaluation_strategy epoch"
fi

torchrun --nproc_per_node 8 instruction_tune.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${pretrained_model} \
    ${extra_args} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --seed 42 \
    --bf16 \
    --num_train_epochs 2 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 2 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 1000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 32 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --load_in_kbits 16 \
    --save_safetensors False \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing \
    --merge_when_finished True
    # When the above parameter is True, the model will be auto-merged after training (the LoRA adapters will be merged into the original LLM).
    # The merged model will be put into {output_dir}_merged
    #--resume_from_checkpoint path_to_checkpoint \
    #--use_flash_attention_2

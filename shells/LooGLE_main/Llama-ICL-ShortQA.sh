python scripts/mp_wrapper.py \
    --script scripts/test_loogle_lift_random_icl_prompt.py \
    --num_process 4 \
    --input_file datasets/loogle/shortdep_qa.jsonl \
    --output_file outputs/LooGLE-ShortQA-Baseline.jsonl \
    --subprocess_args \
    --use_icl True \
    --overwrite True \
    --num_syn_qa 0 \
    --title_option 1 \
    --generator_name_or_path models/Meta-Llama-3-8B-Instruct \
    --model_name_or_path models/Meta-Llama-3-8B-Instruct \
    --model_max_length 7800 \
    --block_size 256 \
    --len_segment 8 \
    --len_offset 3 \
    --use_gated_memory False \
    --load_in_4bit True \
    --use_lora False \
    --use_cot False \
    --gather_batches True \
    --involve_qa_epochs 0 \
    --num_train_epochs 0 \
    --remove_unused_columns True \
    --report_to none \
    --output_dir models/temp \
    --overwrite_output_dir True \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy no \
    --bf16 True \
    --tf32 False \
    --gradient_checkpointing True \
    --lr_scheduler_type constant


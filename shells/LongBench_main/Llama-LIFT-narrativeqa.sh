python scripts/mp_wrapper_longbench.py \
    --script scripts/test_longbench_for_narrativeqa.py \
    --num_process 4 \
    --input_dir datasets/longbench_sampling \
    --output_file outputs/VanGate/narrativeqa-QA30-C3M5-LR2e-3warmup-narrativeqa.jsonl \
    --subprocess_args \
    --subtask_name narrativeqa \
    --overwrite True \
    --num_syn_qa 30 \
    --generator_name_or_path models/Meta-Llama-3-8B-Instruct \
    --use_icl True \
    --model_name_or_path models/Gated-Memory-Llama-3-8B-Instruct \
    --model_max_length 7800 \
    --block_size 256 \
    --len_segment 8 \
    --len_offset 3 \
    --use_gated_memory True \
    --load_in_4bit True \
    --gather_batches True \
    --involve_qa_epochs 5 \
    --num_train_epochs 3 \
    --learning_rate 2e-3 \
    --remove_unused_columns False \
    --report_to none \
    --output_dir models/temp \
    --overwrite_output_dir True \
    --per_device_train_batch_size 1 \
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
    --warmup_steps 4 \
    --lr_scheduler_type constant_with_warmup

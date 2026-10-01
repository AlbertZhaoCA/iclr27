python eval_vllm_accuracy_stats.py \
    --model /mnt/rds/VipinRDS/VipinRDS/users/xxl1337/qwen25/global_step_1755/actor/huggingface/ \
    --model_name ours \
    --tensor_parallel_size 2 \
    --num_runs 10 \
    --temperature 0.6 \
    --out_dir runs/ours/qwen25_7b

# python eval_vllm_accuracy_stats.py \
#     --model /mnt/rds/VipinRDS/VipinRDS/users/xxl1337/qwen25/global_step_1755/actor/huggingface \
#     --model_name Ours \
#     --baseline_model Qwen/Qwen2.5-7B-Instruct \
#     --baseline_name Base \
#     --tensor_parallel_size 2 \
#     --num_runs 1 \
#     --temperature 0.6 \
#     --out_dir runs/ours_vs_base
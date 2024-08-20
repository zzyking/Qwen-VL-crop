checkpoint=YOUR_CHECKPOINT_PATH
for ds in "refcoco_val" "refcoco_testA" "refcoco_testB"
do
    python -m torch.distributed.launch --use-env \
        --nproc_per_node ${NPROC_PER_NODE:-6} \
        --nnodes ${WORLD_SIZE:-1} \
        --node_rank ${RANK:-0} \
        --master_addr ${MASTER_ADDR:-127.0.0.1} \
        --master_port ${MASTER_PORT:-12345} \
        /root/autodl-tmp/Qwen-VL/eval_mm/evaluate_grounding.py \
        --checkpoint $checkpoint \
        --dataset $ds \
        --batch-size 1 \
        --num-workers 6
done
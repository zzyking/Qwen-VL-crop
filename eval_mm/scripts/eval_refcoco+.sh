checkpoint=/root/autodl-tmp/.cache/huggingface/hub/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8
for ds in "refcoco+_val" "refcoco+_testA" "refcoco+_testB"
do
    python -m torch.distributed.launch --use-env \
        --nproc_per_node ${NPROC_PER_NODE:-3} \
        --nnodes ${WORLD_SIZE:-1} \
        --node_rank ${RANK:-0} \
        --master_addr ${MASTER_ADDR:-127.0.0.1} \
        --master_port ${MASTER_PORT:-12345} \
        /root/autodl-tmp/Qwen-VL/eval_mm/evaluate_grounding.py \
        --checkpoint $checkpoint \
        --dataset $ds \
        --batch-size 8 \
        --num-workers 2
done
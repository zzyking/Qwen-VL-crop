ds="textvqa_val"
checkpoint=YOUR_CHECKPOINT_PATH
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-6} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    evaluate_vqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 1 \
    --num-workers 6
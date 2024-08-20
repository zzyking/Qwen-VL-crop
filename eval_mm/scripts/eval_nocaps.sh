ds="nocaps"
checkpoint=/root/autodl-tmp/.cache/huggingface/hub/models--Qwen--Qwen-VL/snapshots/0547ed36a86561e2e42fecec8fd0c4f6953e33c4
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-2} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    evaluate_caption.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2
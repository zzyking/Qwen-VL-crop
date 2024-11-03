# evaluate vqa score on vizwiz val split
ds="vizwiz_val"
checkpoint=/raid_sdd/zzy/19/.cache/huggingface/hub/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-4} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    evaluate_vqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 4 \
    --num-workers 4

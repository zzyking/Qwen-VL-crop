## Evaluation of Qwen-VL with Crop-and-Prompt
***
You can find the download links of datasets at [EVALUATION.md](EVALUATION.md)

All the evaluation scripts has been tested on 6 RTX 4090 D GPUs and you can change the `batch-size`, `num-workers` and `nproc_per_node` accordingly.

Note that you can only use the checkpoint of Qwen-VL, the non-chat version to run these scripts.

Scripts can be found [here](https://github.com/zzyking/Qwen-VL-crop/tree/master/eval_mm/scripts)
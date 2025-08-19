# VLLM FILE CHANGE

### USAGE

```shell
vllm_path=`path_to_vllm_path`
vllm=/opt/conda/lib/python3.11/site-packages/vllm
cp ${vllm_path}/model_executor/layers/quantization/fp8.py        ${vllm}/model_executor/layers/quantization/
cp ${vllm_path}/model_executor/model_loader/utils.py             ${vllm}/model_executor/model_loader/

cp ${vllm_path}/model_executor/models/qwen2.py                   ${vllm}/model_executor/models/
cp ${vllm_path}/model_executor/models/internvl.py                ${vllm}/model_executor/models/
cp ${vllm_path}/model_executor/models/qwen3_moe.py               ${vllm}/model_executor/models/
```

As for logits_processor, please set `+actor_rollout_ref.rollout.invalid_token_ids=[...]` in script instead.

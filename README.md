
代码参考 https://github.com/cambridgeltl/mop 和 https://github.com/wangcunxiang/Can-PLM-Serve-as-KB-for-CBQA/tree/main/simpletransformers/seq2seq 和 https://github.com/shmsw25/bart-closed-book-qa
数据集来自 https://github.com/THU-KEG/KEPLER 和 https://github.com/wangcunxiang/Can-PLM-Serve-as-KB-for-CBQA

```bash
conda create -n qa python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
sudo wg-quick up tw
bash src/relation_prompt/run_pretrain.sh
bash src/evaluation/run_eval_wq.sh
bash src/evaluation/run_eval_nq.sh
bash src/evaluation/run_eval_triq.sh
```

How to implement BartAdapterModel with input_ids=None ? 

## Environment info
<!-- You can run the command `transformers-cli env` and copy-and-paste its output below.
     Remove if your question/ request is not technical. -->
     
- `adapter-transformers` version:3.0.0
- Platform:unbuntu
- Python version: 3.8.0
- PyTorch version (GPU?):  

## Details
        
 # self inherits from BartAdapterModel
  outputs = self.model(
      input_ids=None,
      inputs_embeds=inputs_embeds
  )
Here comes the error:

  File "/home/simon/anaconda3/envs/qa/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/simon/桌面/closed-book-prompt-qa/src/relation_prompt/model_BART.py", line 37, in forward
    outputs = self.model(
  File "/home/simon/anaconda3/envs/qa/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/simon/anaconda3/envs/qa/lib/python3.8/site-packages/transformers/adapters/context.py", line 95, in wrapper_func
    with cls(self, *args, **kwargs):
  File "/home/simon/anaconda3/envs/qa/lib/python3.8/site-packages/transformers/adapters/context.py", line 77, in __init__
    model.forward_context(self, *args, **kwargs)
  File "/home/simon/anaconda3/envs/qa/lib/python3.8/site-packages/transformers/adapters/model_mixin.py", line 618, in forward_context
    input_tensor = args[0]
IndexError: tuple index out of range
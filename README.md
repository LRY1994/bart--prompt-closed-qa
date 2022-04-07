
代码参考 https://github.com/cambridgeltl/mop 和 https://github.com/wangcunxiang/Can-PLM-Serve-as-KB-for-CBQA/tree/main/simpletransformers/seq2seq 和 https://github.com/shmsw25/bart-closed-book-qa
数据集来自 https://github.com/THU-KEG/KEPLER 和 https://github.com/wangcunxiang/Can-PLM-Serve-as-KB-for-CBQA

```bash
reate conda -n env_name python=3.8
conda install pytorch=1.7.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install pytorch_metric_learning
pip install -r requirements.txt
```

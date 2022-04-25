
代码参考 https://github.com/cambridgeltl/mop 和 https://github.com/wangcunxiang/Can-PLM-Serve-as-KB-for-CBQA/tree/main/simpletransformers/seq2seq 和 https://github.com/shmsw25/bart-closed-book-qa
数据集来自 https://github.com/THU-KEG/KEPLER 和 https://github.com/wangcunxiang/Can-PLM-Serve-as-KB-for-CBQA

```bash
conda create -n qa python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
sudo wg-quick up tw
bash src/relation_prompt/train_wiki_roberta.sh
bash src/evaluation/run.sh
```

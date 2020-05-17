# SDM: Sequential Deep Matching Model for Online Large-scale Recommender System
## New Released Code!!!
[doc](https://zhuanlan.zhihu.com/p/141411747)
[code](https://github.com/shenweichen/DeepMatch)
Thanks for the [DeepMatch Group](https://github.com/shenweichen/DeepMatch) Member!

## Demo Code
Code (Python2.7, TF1.4) of the sequential deep matching (SDM) model for recommender system at Taobao.
Current version only contains the core code of our model. The processes of data processing and evaluation are executed on our internal cloud platform [ODPS](https://www.alibabacloud.com/campaign/10-year-anniversary).

## Paper
Here is the arxiv [link](https://arxiv.org/abs/1909.00385) (accepted by CIKM 2019)

Citation:
```
@inproceedings{lv2019sdm,
  title={SDM: Sequential deep matching model for online large-scale recommender system},
  author={Lv, Fuyu and Jin, Taiwei and Yu, Changlong and Sun, Fei and Lin, Quan and Yang, Keping and Ng, Wilfred},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={2635--2643},
  year={2019},
  organization={ACM}
}
```

## Datasets

**JD Dataset:** [raw data](https://drive.google.com/open?id=19PemKrhA8j-RZj0i20_j4ERcnzaxl5JZ), [train and test data](https://drive.google.com/open?id=1pam-_ojsKooRLVeOXEvbh3AwJ6S4IZ7B) in the paper (tfrecord).
The schema of raw data is shown in data/sample_data/.

## Disclaimer
This is an implementation on experiment of offline JD dataset rather than the online official version.
There may be differences between results reported in the paper and the released one,
because the former one is achieved in distribution tensorflow on our internal deep learning platform [PAI](https://data.aliyun.com/product/learn).

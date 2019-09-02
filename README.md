# SDM
Code (Python2.7, TF1.4) for the Sequential Deep Matching model for Recommender System at Taobao.
Current version only contains the core code of our model, and we will continuously complete the remaining parts including data processing, evaluation and running pipeline guidance for the public if possible.

## Datasets
We will release our offline datasets soon after the approval. The schema of raw data is shown in data/sample_data/.

**JD Dataset:** [raw data](https://drive.google.com/open?id=19PemKrhA8j-RZj0i20_j4ERcnzaxl5JZ), [train and test data](https://drive.google.com/open?id=1pam-_ojsKooRLVeOXEvbh3AwJ6S4IZ7B) in the paper (tfrecord)

**Taobao Dataset:** Under internal checking procedure.

## Disclaimer
This is an implementation on experiment of offline JD dataset rather than the online official version.

There may be differences between results reported in the paper and the released one, because the former one is achieved in distribution tensorflow.

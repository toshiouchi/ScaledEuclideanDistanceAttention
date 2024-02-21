# Scaled Euclidean Distance Attention of Transformer

## Check of learning with vision Transformer

To check the proposed attention, I tried learning using Self Attention of Transformer Encoder, an image classification program, and learning progressed using Euclidean distance. For inner product, one epoch takes about 8 minutes, and for Euclidean distance, it takes about 15 minutes. It is not possible to conclude the goodness of the model just by learning for this check. If you use the normal inner product for the similarity between q and k , the book "Learning Image Recognition with Python" in Japanese says that the accuracy rate of the test data is 63.8% using  book's image classification program for a certain problem. In the book's program using the reciprocal of the Euclidean distance instead of the inner product for the same problem, the accuracy rate was 66.7% .

### Chnaging of Loss for image classification

![Loss](https://github.com/toshiouchi/ScaledEuclideanDistanceAttention/assets/121741811/abc71db8-400c-48e2-b328-cf8bef12c8bd)

### Changing of Accuracy

![Accuracy](https://github.com/toshiouchi/ScaledEuclideanDistanceAttention/assets/121741811/1d1d8a85-1c49-4c64-a514-7f3b27370918)

## Check with machine translation program

For Transformer in a machine translator, from a programmatic perspective, it works with Transformer Decoder's Self Attention and Cross Attention. Additionally, in a machine translation program, we checked that learning progressed using attention based on Euclidean distance for the Transformer Encoder and Transformer Decoder.

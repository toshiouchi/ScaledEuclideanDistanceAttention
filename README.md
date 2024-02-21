Scaled Euclidean Distance Attention of Transformer

## Confirmation of learning for vision transformer

To confirm here, I tried learning using Self Attention of Transformer Encoder, an image classification program, and learning progressed using Euclidean distance. For inner product, one epoch takes about 8 minutes, and for Euclidean distance, it takes about 15 minutes. Although it is not possible to conclude the goodness of the model just by learning for this confirmation, if you use the normal inner product for the similarity between q and k in the image classification program in the book ``Learning Image Recognition with Python'', The book says that the accuracy rate of the test data is 63.8%. In the book's program, the learning rate was 66.7% using the reciprocal of the Euclidean distance instead of the inner product.

### Chnaging of Loss for image classification

![Loss](https://github.com/toshiouchi/ScaledEuclideanDistanceAttention/assets/121741811/abc71db8-400c-48e2-b328-cf8bef12c8bd)

### Changing of Accuracy

![Accuracy](https://github.com/toshiouchi/ScaledEuclideanDistanceAttention/assets/121741811/1d1d8a85-1c49-4c64-a514-7f3b27370918)

## Confirmation with machine translation program

For Transformer in a machine translator, from a programmatic perspective, it works with Transformer Decoder's Self Attention and Cross Attention. Additionally, in a machine translation program, we confirmed that learning progressed using attention based on Euclidean distance for the Transformer Encoder and Transformer Decoder.

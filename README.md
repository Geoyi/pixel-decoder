# Pixel Decoder

In computer vision, there are three challenges: image classification, object detection and semantic segmentation.
![computervision](https://user-images.githubusercontent.com/14057932/37719364-3e953da0-2cfb-11e8-8140-f5f12bb806d9.png)
As you see above, semantic segmentation can segment an image into different parts and objects (e.g.grass, cat, tree, sky).

Pixel Decoder is a tool that contains several current available semantic segmentation algorithms. It was built on top of SpaceNet challenge solutions. It includes Standard Unet and its modified versions, Tiramisu and SegNet. SegNet is the algorithm that Skynet was built on. All the algorithms that live inside Pixel Decoder are convolutional neural networks are all in a structure that called encoder-decoder.
![encoder-decoder](https://user-images.githubusercontent.com/14057932/37719742-14b23582-2cfc-11e8-8242-a3773df31bc2.png)
The encoder reads in the image pixels and compresses the information in vector, downsample to save computing memory; and the decoder works on reconstructing the pixels spatial information and output the desired outcome.

You can see all the following algorithms are all included in, they are ordered by the total model parameters. With the same size of dataset and batch size, a heavier algorithm with take a longer time to tune every parameter, and the training process will take longer time.

|Algorithm | Weight (total params)|
| --- | --- |
| Unet_mini | light - 1,962,625 |
|SegNet| light - 5,467,265 |
| Unet | mid -  7,846,657 |
| Unet_Dilated | mid -  7,846,657 |
|Unet_ level_7 | mid -  7,846,657 |
|Dilated_Unet| heavy -14,839,521|
|Unet_ level_8 | heavy -  31,462,273 |
| Tiramisu | very heavy - 70,372,786 |

All these algorithms are built with Tensorflow and Keras. If your training dataset is smaller than 60M, you can try a light- to a mid-weight algorithm, with 16 batch size, each epoch might take about 1- 1.5 minute to run. As long as `tensorflow-gpu` is installed and GPU can be detected by Tensorflow, the training will be speeded up. Each epoch will speed up to between 20 - 30 second for the training dataset and batch size.
Theoretically, I recommend starting with Unet and SegNet for training.

### Start

To set up a [python environment](https://gist.github.com/wronk/a902185f5f8ed018263d828e1027009b) and check what do you need to install in above `requirement.txt`

### Train
Run:
```shell
python pixel-decoder/main_hypo.py data/data_testset_5cl.npz
```
It takes in the training dataset that created from [`Label Maker`](https://github.com/developmentseed/label-maker).

But in real life you might face these two situations for creating training dataset:
- You have color images and the label images are in `JPG`, `TIF` or `PNG`, and are not in numpy `npz` format yet. In this situation you can use the code that I created [`data_from_imgs.py`](I will upload later!). It will create 'npz' numpy array that will read by algorithms.

- You have several `npz` files that you generated and want to train together by your selected algorithm, you can go to [this line](https://github.com/developmentseed/satellite-ml-internal/blob/ff4445e2a77eb9971df6bd3f47f673054915fe8f/pixel-decoder/pixel-decoder/main_hypo.py#L165) and replace `load_data` by`stacked_data`.


### Predict
After the model is trained and you see a trained model weight in your model directory, run:

```shell
python pixel-decoder/predict.py data/data_testset_5cl.npz Unet-20180205-070723.hdf5
```
In this case, my trained model weight is `Unet-20180205-070723.hdf5`.


## About
pixel_decoder was built on top of python-seed that created by [Development Seed](<http://developmentseed.org>)

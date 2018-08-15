# Pixel Decoder

![computervision](https://user-images.githubusercontent.com/14057932/37719364-3e953da0-2cfb-11e8-8140-f5f12bb806d9.png)
In **computer vision**, there are three challenges: image classification, object detection and **semantic segmentation**. As you see above, semantic segmentation can segment an image into different parts and objects (e.g.grass, cat, tree, sky).

Pixel Decoder is a tool that contains several current available semantic segmentation algorithms. **Pixel Decoder** includes Standard Unet and its modified versions, Tiramisu and SegNet. SegNet is the algorithm that Skynet was built on. All the algorithms that live inside Pixel Decoder are convolutional neural networks are all in a structure that called encoder-decoder.
![encoder-decoder](https://user-images.githubusercontent.com/14057932/37719742-14b23582-2cfc-11e8-8242-a3773df31bc2.png)
The encoder reads in the image pixels and compresses the information in vector, downsample to save computing memory; and the decoder works on reconstructing the pixels spatial information and output the desired outcome. Some UNet-like algorithms were adopted from SpaceNet challenge solutions.

All these algorithms are built with Tensorflow and Keras.

### Installation

```bash
git clone https://github.com/Geoyi/pixel-decoder
cd pixel-decoder

```

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

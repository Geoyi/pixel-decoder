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
pip install -e .
```

### Train

```bash
pixel_decoder train --batch_size=4 \
                    --imgs_folder=tiles \
                    --masks_folder=labels \
                    --models_folder=trained_models_out \
                    --model_id=resnet_unet \
                    --origin_shape_no=256 \
                    --border_no=32
```
It takes in the training dataset that created from [`Label Maker`](https://github.com/developmentseed/label-maker).


- `batch_size`: batch size for the training;
- `imgs_folder`: is the directory for RGB images to train;
- `masks_folder`: is the directory for labeled mask to train;
- `model_id`: is the neural net architecture to train with. We have - `resnet_unet`, `inception_unet`, `linknet_unet`, `SegNet`, `Tiramisu` as model_id live in **Pixel Decoder**.
- `origin_shape_no`: 256 is the default image tile shape from [Label Maker](https://github.com/developmentseed/label-maker);
- `border_no`: it's set to 32. It's a additional 32 pixel to add on 256 by 256 image tile to become 320 by 320 to get rid of U-Net's edge distortion.


### Predict
After the model is trained and you see a trained model weight in your model directory, run:


```bash
pixel_decoder predict --imgs_folder=tiles \
                    --test_folder=test_images \
                    --models_folder=trained_models_out \
                    --pred_folder=predictions \
                    --model_id=resnet_unet \
                    --origin_shape_no=256 \  
                    --border_no=32
```

- `imgs_folder`: is the directory for RGB images to train;
- `masks_folder`: is the directory for labeled mask to train. It uses to get the stats, e.g. mean and standard deviation, from training images.
- `test_folder`: is the directory for test images.
- `pred_folder`: a directory that saved all the predicted test image from test_folder;
- `model_id`: is the neural net architecture to train with. We have - `resnet_unet`, `inception_unet`, `linknet_unet`, `SegNet`, `Tiramisu` as model_id live in **Pixel Decoder**.
- `origin_shape_no`: 256 is the default image tile shape from [Label Maker](https://github.com/developmentseed/label-maker);
- `border_no`: it's set to 32. It's a additional 32 pixel to add on 256 by 256 image tile to become 320 by 320 to get rid of U-Net's edge distortion.

## Run Pixel Decoder on AWS Deep Learning AMI instance with **GUPs**

### Install Nvidia-Docker on your instance
- [Docker installation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html) on AWS EC2. Instruction for Nvidia Docker installation [here](https://towardsdatascience.com/using-docker-to-set-up-a-deep-learning-environment-on-aws-6af37a78c551).

- Build provide docker image from the Dockerfile

```bash
git clone https://github.com/Geoyi/pixel-decoder
cd pixel-decoder
nvidia-docker build -t pixel_decoder .
```

- Run nvidia-docker and Pixel Decoder

```bash
nvidia-docker run -v $PWD:/work -it pixel_decoder bash
```

- Install Pixel Decoder and train the model

**Train**
```bash
pixel_decoder train --batch_size=4 \
                    --imgs_folder=tiles \
                    --masks_folder=labels \
                    --models_folder=trained_models_out \
                    --model_id=resnet_unet \
                    --origin_shape_no=256 \
                    --border_no=32
```

**Predict**

```bash
pixel_decoder predict --imgs_folder=tiles \
                    --test_folder=test_images \
                    --models_folder=trained_models_out \
                    --pred_folder=predictions \
                    --model_id=resnet_unet \
                    --origin_shape_no=256 \  
                    --border_no=32
```

## About
To run a neural net, e.g `resnet_unet`, you can create ready-to-train dataset from [Label Maker](https://github.com/developmentseed/label-maker). A detail walkthrough notebook will come soon.
pixel_decoder was built on top of python-seed that created by [Development Seed](<http://developmentseed.org>).

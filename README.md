# Learning a Model-based Painting Agent \\Drawing by Layers


## Abstract

We propose a model-based deep reinforcement learning approach to teach machines to generate sequential stroke-based paintings by drawing in layers, similar to human painters. Our approach incorporates multiple actors, enabling the painting agent to create layers that can be combined for better stylization and increased variation in styles. Experiments show that our approach outperforms existing solutions, resulting in excellent visual effects with fewer training steps and generating stylized images by separating foreground and background strokes.


### Dependencies

Use conda to manage the environment:
```sh
conda create -n paint python=3.9
conda activate paint
# find a suitable version for your gpu, e.g.: cuda 11.7
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c menpo opencv 
conda install -c conda-forge tensorboardx
conda install scipy==1.2.0
```

## Trained models

After we packing and uploading the models, we will put links here.

## Testing
Make sure there are renderer.pkl and actor.pkl before testing.


```
$ wget "https://drive.google.com/uc?export=download&id=1-7dVdjCIZIxh8hHJnGTK-RA1-jL1tor4" -O renderer.pkl
$ wget "https://drive.google.com/uc?export=download&id=1a3vpKgjCVXHON4P7wodqhCgCMPgg1KeR" -O actor.pkl
$ python3 baseline/test.py --max_step=100 --actor=actor.pkl --renderer=renderer.pkl --img=image/test.png --divide=4
$ ffmpeg -r 10 -f image2 -i output/generated%d.png -s 512x512 -c:v libx264 -pix_fmt yuv420p video.mp4 -q:v 0 -q:a 0
(make a painting process video)
```

## Training

### Datasets
~~Download the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and put the aligned images in data/img_align_celeba/\*\*\*\*\*\*.jpg~~
Download img_align_celeba directly from [https://drive.google.com/u/0/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM&export=download](https://drive.google.com/u/0/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM&export=download), then unzip it into `data/`

### Neural Renderer
To create a differentiable painting environment, we need train the neural renderer firstly. 

```
$ python3 baseline/train_renderer.py
$ tensorboard --logdir train_log --port=6006
(The training process will be shown at http://127.0.0.1:6006)
```

### Paint Agent
After the neural renderer looks good enough, we can begin training the agent.
```
$ cd baseline
$ python3 train.py --max_step=40 --debug --batch_size=96
(A step contains 5 strokes in default.)
$ tensorboard --logdir train_log --port=6006
```

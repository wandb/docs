# W&B + fastai 

<a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/fastai/Weights_&_Biases_with_fastai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<img src="https://i.imgur.com/pNKgZgL.png" alt="Fastai2 and Weights & Biases" width="400"/>
<div><img /></div>

<img src="https://i.imgur.com/uEtWSEb.png" width="650" alt="Weights & Biases" />

<div><img /></div>

# 💨 Fastai and 🏋️‍♀️ Weights & Biases

Fastai let us create quickly neural networks architectures using modern best practices in just a few lines of code.

This notebook shows how to use fastai with the[`WandbCallback`](https://docs.wandb.com/library/integrations/fastai) to log and visualize experiments.

## Install libraries

First, install and import `fastai` and `wandb`.


```
!pip install -qU wandb fastai timm
```


```
import wandb

from fastai.vision.all import *
from fastai.callback.wandb import *
```

## Log in to W&B
Log in so your results can stream to a private project in W&B. Here's more info on the [data privacy and export features](https://docs.wandb.com/company/data-and-privacy) you can use so W&B can serve as a reliable system of record for your experiments.

*Note: Login only needs to be done once, and it is automatically called with `wandb.init()`.*


```
wandb.login()
```

## Getting Started

Let's start with a very simple fastai pipeline:


```
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42, bs=32,
    label_func=is_cat, item_tfms=Resize(128))

learn = vision_learner(dls, "convnext_tiny", metrics=error_rate)
learn.fine_tune(1)
```

## How do we add W&B to this pipeline?
You just need to add the `WandbCallback` to the `Learner` (or to the fit method with the `cbs` argument)


```
learn = vision_learner(dls, "convnext_tiny", metrics=error_rate, cbs = WandbCallback())
```

create a run by calling `wandb.init`


```
wandb.init(project="fastai");
```

Train your model as usual


```
learn.fine_tune(1)
```

end the run


```
wandb.finish()
```

you can now click on the run link and enjoy your dashboard ☝️

**TLDR**; it's even shorter if you use the context manager:
```python
with wandb.init(project="fastai"):
    learn = vision_learner(dls, resnet34, metrics=error_rate, cbs = WandbCallback())
    learn.fine_tune(1)
```

## Semantic segmentation on CamVid

In this example, we'll train a U-Net with a ResNet encoder to perform semantic segmentation on the CamVid dataset.

<img src="http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/pr/DBOverview1_1_huff_0000964.jpg" alt="Camvid dataset" width="500"/>

### Download the dataset

Fastai datasets are downloaded from a URL and cached locally.


```
path = untar_data(URLs.CAMVID_TINY)
path
```

This specific dataset contains:
* a folder of input images
* a folder of segmentation masks (same name as images with added suffix `_P`)
* a file listing in order the possible classes
* a file listing which files belong to validation set

### Create DataLoaders
We can create `DataLoaders` in many possible ways: from a `Dataset`, `TfmdList`, `DataBlock` or custom methods such as `ImageDataLoaders` or `SegmentationDataLoaders`.


```
# Get classes
codes = np.loadtxt(path/'codes.txt', dtype=str)
codes
```


```
# Get list of input files path
fnames = get_image_files(path/"images")
fnames[:3]
```


```
# get label path from an input path
def label_func(fn): return path/"labels"/f"{fn.stem}_P{fn.suffix}"
```


```
# create DataLoaders using a function specific to semantic segmentation
dls = SegmentationDataLoaders.from_label_func(path, bs=8, fnames=fnames, label_func=label_func, codes=codes)
```


```
dls.show_batch()
```

### Train a model

We start a new W&B run with wandb.init() which gives us a link to our logged run.


```
wandb.init(project='fastai_camvid');
```

`WandbCallback` can automatically track:
* hyper-parameters
* losses & metrics
* models
* datasets
* code
* computer resources

In addition to logging losses & metrics, we are going to log our dataset and our model, which will be automatically versioned.


```
learn = unet_learner(dls, resnet18, metrics=foreground_acc, cbs=WandbCallback(log_dataset=True, log_model=True))
```


```
learn.fit_one_cycle(2, )
```


```
# optional: mark the run as completed
wandb.finish()
```

That's it! Check out your fastai model training in the live W&B dashboard by clicking on the link printed out above.

# Example W&B dashboard
![](https://i.imgur.com/jef6GjA.png)



# Learn more!
1. [Documentation](https://docs.wandb.com/library/integrations/fastai): Explore the docs to learn what's possible with Weights & Biases visualizations for Fastai models
2. [Slack community](http://wandb.me/slack): Ask questions and share results in our vibrant community of practitioners
3. [Gallery](app.wandb.ai/gallery): See more reproducible research projects from practitioners around the world in the W&B gallery



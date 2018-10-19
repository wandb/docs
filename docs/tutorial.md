---
title: Tutorial
sidebar_label: Tutorial
---

Once you've installed `wandb`, try tracking this quick example CNN.

Download the sample script. We've added a few lines of code to send logs to W&B.
```shell
git clone http://github.com/cvphelps/tutorial
cd tutorial
pip install -r requirements.txt
```
Initialize the training directory.
```shell
wandb init
```
Run the script from the command line.
```shell
python learn.py
```

Go to [app.wandb.ai](https://app.wandb.ai) to see your training runs!
---
description: Examples of W&B Tables
menu:
  default:
    identifier: tables-gallery
    parent: tables
title: Example tables
---

The following sections highlight some of the ways you can use tables:

### View your data

Log metrics and rich media during model training or evaluation, then visualize results in a persistent database synced to the cloud, or to your [hosting instance](/guides/hosting). 

{{< img src="/images/data_vis/tables_see_data.png" alt="Browse examples and verify the counts and distribution of your data" >}}

For example, check out this table that shows a [balanced split of a photos dataset](https://wandb.ai/stacey/mendeleev/artifacts/balanced_data/inat_80-10-10_5K/ab79f01e007113280018/files/data_split.table.json).

### Interactively explore your data

View, sort, filter, group, join, and query tables to understand your data and model performance—no need to browse static files or rerun analysis scripts. 

{{< img src="/images/data_vis/explore_data.png" alt="Listen to original songs and their synthesized versions (with timbre transfer)" >}}

For example, see this report on [style-transferred audio](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM).

### Compare model versions

Quickly compare results across different training epochs, datasets, hyperparameter choices, model architectures etc. 

{{< img src="/images/data_vis/compare_model_versions.png" alt="See granular differences: the left model detects some red sidewalk, the right does not." >}}

For example, see this table that compares [two models on the same test images](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#b6dae62d4f00d31eeebf$eval_Bob).

### Track every detail and see the bigger picture

Zoom in to visualize a specific prediction at a specific step. Zoom out to see the aggregate statistics, identify patterns of errors, and understand opportunities for improvement. This tool works for comparing steps from a single model training, or results across different model versions.

{{< img src="/images/data_vis/track_details.png" alt="" >}}

For example, see this example table that analyzes results [after one and then after five epochs on the MNIST dataset](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec).
## Example Projects with W&B Tables
The following highlight some real W&B Projects that use W&B Tables.

### Image classification

Read [this report](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA), follow [this colab](https://wandb.me/dsviz-nature-colab), or explore this [artifacts context](https://wandb.ai/stacey/mendeleev/artifacts/val_epoch_preds/val_pred_gawf9z8j/2dcee8fa22863317472b/files/val_epoch_res.table.json) to see how a CNN identifies ten types of living things (plants, bird, insects, etc) from [iNaturalist](https://www.inaturalist.org/pages/developers) photos.

{{< img src="/images/data_vis/image_classification.png" alt="Compare the distribution of true labels across two different models' predictions." >}}

### Audio

Interact with audio tables in [this report](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) on timbre transfer. You can compare a recorded whale song with a synthesized rendition of the same melody on an instrument like violin or trumpet. You can also record your own songs and explore their synthesized versions in W&B with [this colab](http://wandb.me/audio-transfer).

{{< img src="/images/data_vis/audio.png" alt="" >}}

### Text

Browse text samples from training data or generated output, dynamically group by relevant fields, and align your evaluation across model variants or experiment settings. Render text as Markdown or use visual diff mode to compare texts. Explore a simple character-based RNN for generating Shakespeare in [this report](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY).

{{< img src="/images/data_vis/shakesamples.png" alt="Doubling the size of the hidden layer yields some more creative prompt completions." >}}

### Video

Browse and aggregate over videos logged during training to understand your models. Here is an early example using the [SafeLife benchmark](https://wandb.ai/safelife/v1dot2/benchmark) for RL agents seeking to [minimize side effects ](https://wandb.ai/stacey/saferlife/artifacts/video/videos_append-spawn/c1f92c6e27fa0725c154/files/video_examples.table.json)

{{< img src="/images/data_vis/video.png" alt="Browse easily through the few successful agents" >}}

### Tabular data

View a report on how to [split and preprocess tabular data](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1) with version control and deduplication.

{{< img src="/images/data_vis/tabs.png" alt="Tables and Artifacts work together to version control, label, and deduplicate your dataset iterations" >}}

### Comparing model variants (semantic segmentation)

An [interactive notebook](https://wandb.me/dsviz-cars-demo) and [live example](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#a57f8e412329727038c2$eval_Ada) of logging Tables for semantic segmentation and comparing different models. Try your own queries [in this Table](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json).

{{< img src="/images/data_vis/comparing_model_variants.png" alt="Find the best predictions across two models on the same test set" >}}

### Analyzing improvement over training time

A detailed report on how to [visualize predictions over time](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk) and the accompanying [interactive notebook](https://wandb.me/dsviz-mnist-colab).
---
slug: /guides/data-vis
description: Iterate on datasets and understand model predictions
displayed_sidebar: default
---

# Visualize your data

Use **W&B Tables** to log, query, and analyze tabular data. Understand your datasets, visualize model predictions, and share insights in a central dashboard. For example, with W&B Tables, you can:

* Compare changes precisely across models, epochs, or individual examples
* Understand higher-level patterns in your data
* Capture and communicate your insights with visual samples


## What are W&B Tables?

A W&B Table (`wandb.Table`) is a two dimensional grid of data where each column has a single type of data—think of this as a more powerful Pandas DataFrame. Tables support primitive and numeric types, as well as nested lists, dictionaries, and rich media types. Log a Table to W&B, then query, compare, and analyze results in the UI.

### View your data

Log metrics and rich media during model training or evaluation, then visualize results in a persistent database synced to the cloud, or to your [hosting instance](https://docs.wandb.ai/guides/hosting). For example, check out this [balanced split of a photos dataset →](https://wandb.ai/stacey/mendeleev/artifacts/balanced\_data/inat\_80-10-10\_5K/ab79f01e007113280018/files/data\_split.table.json)

![Browse actual examples and verify the counts & distribution of your data](/images/data_vis/tables_see_data.png)

### Interactively explore your data

View, sort, filter, group, join, and query Tables to understand your data and model performance—no need to browse static files or rerun analysis scripts. For example, see this project on [style-transferred audio →](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)

![Listen to original songs and their synthesized versions (with timbre transfer)](/images/data_vis/explore_data.png)

### Compare model versions

Quickly compare results across different training epochs, datasets, hyperparameter choices, model architectures etc. For example, take a look at this comparison of [two models on the same test images →](https://wandb.ai/stacey/evalserver\_answers\_2/artifacts/results/eval\_Daenerys/c2290abd3d7274f00ad8/files/eval\_results.table.json#b6dae62d4f00d31eeebf$eval\_Bob)

![See granular differences: the left model detects some red sidewalk, the right does not.](/images/data_vis/compare_model_versions.png)

### Track every detail and see the bigger picture

Zoom in to visualize a specific prediction at a specific step. Zoom out to see the aggregate statistics, identify patterns of errors, and understand opportunities for improvement. This tool works for comparing steps from a single model training, or results across different model versions. Check out this example table analyzing results [after 1 vs 5 epochs on MNIST →](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)

![](/images/data_vis/track_details.png)

## Example Projects with W&B Tables

### Image classification

Read [this report](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA), follow [this colab](https://wandb.me/dsviz-nature-colab), or explore this [artifacts context](https://wandb.ai/stacey/mendeleev/artifacts/val\_epoch\_preds/val\_pred\_gawf9z8j/2dcee8fa22863317472b/files/val\_epoch\_res.table.json) for a CNN identifying 10 types of living things (plants, bird, insects, etc) from [iNaturalist](https://www.inaturalist.org/pages/developers) photos.

![Compare the distribution of true labels across two different models' predictions.](/images/data_vis/image_classification.png)

### Audio

Interact with audio Tables in[ this report](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) on timbre transfer. You can compare a recorded whale song with a synthesized rendition of the same melody on an instrument like violin or trumpet. You can also record your own songs and explore their synthesized versions in W&B via [this colab →](http://wandb.me/audio-transfer)

![](/images/data_vis/audio.png)

### Text

Browse text samples from training data or generated output, dynamically group by relevant fields, and align your evaluation across model variants or experiment settings. Render text as Markdown or use visual diff mode to compare texts. Explore a simple character-based RNN for generating Shakespeare in [this report →](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY)

![Doubling the size of the hidden layer yields some more creative prompt completions.](@site/static/images/data_vis/shakesamples.png)

### Video

Browse and aggregate over videos logged during training to understand your models. Here is an early example using the [SafeLife benchmark](https://wandb.ai/safelife/v1dot2/benchmark) for RL agents seeking to [minimize side effects →](https://wandb.ai/stacey/saferlife/artifacts/video/videos\_append-spawn/c1f92c6e27fa0725c154/files/video\_examples.table.json)

![Browse easily through the few successful agents](/images/data_vis/video.png)

### Tabular data

A report on [splitting and preprocessing tabular data](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1) with version control and deduplication.

![Tables and Artifacts work together to version control, label, and deduplicate your dataset iterations](@site/static/images/data_vis/tabs.png)

### Comparing model variants (semantic segmentation)

An [interactive notebook](https://wandb.me/dsviz-cars-demo) and [live example](https://wandb.ai/stacey/evalserver\_answers\_2/artifacts/results/eval\_Daenerys/c2290abd3d7274f00ad8/files/eval\_results.table.json#a57f8e412329727038c2$eval\_Ada) of logging Tables for semantic segmentation and comparing different models. Try your own queries [in this Table →](https://wandb.ai/stacey/evalserver\_answers\_2/artifacts/results/eval\_Daenerys/c2290abd3d7274f00ad8/files/eval\_results.table.json)

![Find the best predictions across two models on the same test set](/images/data_vis/comparing_model_variants.png)

### Analyzing improvement over training time

A detailed report on [visualizing predictions over time](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk) and the accompanying [interactive notebook →](https://wandb.me/dsviz-mnist-colab)

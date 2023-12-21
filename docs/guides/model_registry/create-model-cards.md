---
description: ''
displayed_sidebar: default
---

# Document machine learning model

Add a description to the model card of your registered model to document aspects of your machine learning model. Some topics worth documenting include:

* **Summary**: A summary of what the model is. The purpose of the model. The machine learning framework that was used, and so forth. 
* **Training data**: Describe which training data was used. What sort of processing was done on the training data set. Where is that data stored.
* **Architecture**: The architecture of the machine learning algorithm. Did the algorithm use transfer learning? And so forth.
* **Deserialize the model**: Provide information on how someone on your team can load the model into memory.


## Add a description to the model card

1. Navigate to the W&B Model Registry app at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Select **View details** next to the name of the registered model you want to create a model card for.
2. Go to the **Model card** section.
![](/images/models/model_card_example.png)
3. Within the **Description** field, provide information about your machine learning model. Format text within a model card with [Markdown markup language](https://www.markdownguide.org/).


For example, the following images shows the model card of a **Credit-card Default Prediction** registered model. Background information about the registered model is provided Within the model card.
![](/images/models/model_card_credit_example.png)
---
description: How to use W&B with the OpenAI API.
menu:
  default:
    identifier: openai-api
    parent: integrations
title: OpenAI API
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://github.com/wandb/examples/blob/master/colabs/openai/OpenAI_API_Autologger_Quickstart.ipynb"></CTAButtons>

Use the W&B OpenAI API integration to log requests, responses, token counts and model metadata for all OpenAI models, including fine-tuned models. 



{{% alert %}}
See the [OpenAI fine-tuning integration](./openai-fine-tuning.md) to learn how to use W&B to track your fine-tuning experiments, models, and datasets and share your results with your colleagues.
{{% /alert %}}

Log your API inputs and outputs you can quickly evaluate the performance of difference prompts, compare different model settings (such as temperature), and track other usage metrics such as token usage.




![](/images/integrations/open_ai_autolog.png)


## Install OpenAI Python API library

The W&B autolog integration works with OpenAI version 0.28.1 and below.

To install OpenAI Python API version 0.28.1, run:
```python
pip install openai==0.28.1
```


### 1. Import autolog and initialise it
First, import `autolog` from `wandb.integration.openai` and initialise it.  

```python
import os
import openai
from wandb.integration.openai import autolog

autolog({"project": "gpt5"})
```

You can optionally pass a dictionary with argument that `wandb.init()` accepts to `autolog`. This includes a project name, team name, entity, and more. For more information about [`wandb.init`](../../../ref/python/init.md), see the API Reference Guide.

### 2. Call the OpenAI API
Each call you make to the OpenAI API is now logged to W&B automatically.

```python
os.environ["OPENAI_API_KEY"] = "XXX"

chat_request_kwargs = dict(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers"},
        {"role": "user", "content": "Where was it played?"},
    ],
)
response = openai.ChatCompletion.create(**chat_request_kwargs)
```

### 3. View your OpenAI API inputs and responses

Click on the W&B [run](../../runs/intro.md) link generated by `autolog` in **step 1**. This redirects you to your project workspace in the W&B App.

Select a run you created to view the trace table, trace timeline and the model architecture of the OpenAI LLM used.

### 4. Turn off autolog
W&B recommends that you call `disable()` to close all W&B processes when you are finished using the OpenAI API.

```python
autolog.disable()
```

Now your inputs and completions will be logged to W&B, ready for analysis or to be shared with colleagues.
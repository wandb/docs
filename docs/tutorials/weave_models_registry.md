---
title: Weave and Models
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/drive/1Uqgel6cNcGdP7AmBXe2pR9u6Dejggsh8?usp=sharing'/>

# Models and Weave - Integration Demo
This notebook shows how W&B Weave can be used together with W&B Models. Specifically, we consider two different teams as part of this example.

* **The Model Team:** the model building team fine-tunes a new Chat Model (Llama 3.2) and saves it to the registry using **W&B Models**.
* **The App Team:** the app development team retrieves the Chat Model to to create and evaluate a new RAG Chatbot using **W&B Weave**.

Find the public workspace for both W&B Models and W&B Weave [here](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/evaluations).

<img src="https://github.com/NiWaRe/agent-dev-collection/blob/master/screenshots/weave_models_workflow.jpg?raw=true"  alt="Weights & Biases" />

We'll cover the following steps as part of the workflow:

1. Create a new `ChatModel` object on Weave based on the new fine-tuned Chat Model in the registry.
2. Retrieve the `RagModel` object on Weave - with old `ChatModel`.
3. Create a new `RagModel`with the new `ChatModel` and publish to Weave.
4. Get existing Evaluation from Weave and evaluate new `RagModel` and save the results to W&B Weave and to W&B Models.
5. Save the new `RagModel` on the registry to be shared. 

# 1. Setup
We first have to install `weave` and `wandb` and login. We also set a couple of API keys that we might need.

```python
# pip install weave wandb

import wandb
import weave
import pandas as pd

PROJECT = "weave-cookboook-demo"
ENTITY = "wandb-smle"

wandb.login()
weave.init(ENTITY + "/" + PROJECT)
```

# 2. Make `ChatModel` based on Artifact
We retrieve the fine-tuned chat model from the Registry and create a `weave.Model` out of it that we can directly plug in to the [RagModel](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2FcqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo%3F%26) in the next step. It takes in the same parameters as the existing [ChatModel](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-rag-experiments%2Fobjects%2FChatModelRag%2Fversions%2F2mhdPb667uoFlXStXtZ0MuYoxPaiAXj3KyLS1kYRi84%3F%26) just the `init` and `predict` change.

```python
pip install unsloth
# Also get the latest nightly Unsloth!
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

The model team fine-tuned different Llama-3.2 models using the `unsloth` library to make it faster. Hence we'll use the special `unsloth.FastLanguageModel` or `peft.AutoPeftModelForCausalLM` models with adapters to load in the model once we download it from the Registry. The loading code in the `model_post_init` can be simply copy & pasted from the "Use" tab in the Registry.

```python
import weave
from pydantic import PrivateAttr
from typing import Any, List, Dict, Optional
from unsloth import FastLanguageModel
import torch


class UnslothLoRAChatModel(weave.Model):
    """
    We define an extra ChatModel class to be able store and version more parameters than just the model name.
    Especially, relevant if we consider fine-tuning (locally or aaS) because of specific parameters.
    """

    chat_model: str
    cm_temperature: float
    cm_max_new_tokens: int
    cm_quantize: bool
    inference_batch_size: int
    dtype: Any
    device: str
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def model_post_init(self, __context):
        # we can simply paste this from the "Use" tab from the registry
        run = wandb.init(project=PROJECT, job_type="model_download")
        artifact = run.use_artifact(f"{self.chat_model}")
        model_path = artifact.download()

        # unsloth version (enable native 2x faster inference)
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.cm_max_new_tokens,
            dtype=self.dtype,
            load_in_4bit=self.cm_quantize,
        )
        FastLanguageModel.for_inference(self._model)

    @weave.op()
    async def predict(self, query: List[str]) -> dict:
        # add_generation_prompt = true - Must add for generation
        input_ids = self._tokenizer.apply_chat_template(
            query,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        output_ids = self._model.generate(
            input_ids=input_ids,
            max_new_tokens=64,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )

        decoded_outputs = self._tokenizer.batch_decode(
            output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
        )

        return "".join(decoded_outputs).strip()
```

Now we create a new model with a specific link from our registry:

```python
MODEL_REG_URL = "wandb32/wandb-registry-RAG Chat Models/Finetuned Llama-3.2:v3"

max_seq_length = 2048
dtype = None
load_in_4bit = True

new_chat_model = UnslothLoRAChatModel(
    name="UnslothLoRAChatModelRag",
    chat_model=MODEL_REG_URL,
    cm_temperature=1.0,
    cm_max_new_tokens=max_seq_length,
    cm_quantize=load_in_4bit,
    inference_batch_size=max_seq_length,
    dtype=dtype,
    device="auto",
)
```

 And finally run the evaluation in async mode:

 ```python
 await new_chat_model.predict(
     [{"role": "user", "content": "What is the capital of Germany?"}]
 )
 ```

 # 3. Integrate new `ChatModel` version into `RagModel`
Now we retrieve the [RagModel](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2FcqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo%3F%26) (you can fetch the weave ref for the current RagModel from the use tab as shown in the image below) from our existing Weave project and exchange the `ChatModel` to the new one. We don't need to change or re-create any of the other components (VDB, prompts, etc.)!

<img src="https://github.com/wandb/docodile/blob/weave-models-tutorial/static/images/tutorials/weave-ref-1.png?raw=true"  alt="Weights & Biases" />

```python
# pip install litellm faiss-gpu

RagModel = weave.ref(
    "weave:///wandb-smle/weave-cookboook-demo/object/RagModel:cqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo"
).get()
# MAGIC: exchange chat_model and publish new version (no need to worry about other RAG components)
RagModel.chat_model = new_chat_model
# first publish new version so that in prediction we reference new version
PUB_REFERENCE = weave.publish(RagModel, "RagModel")
await RagModel.predict("When was the first conference on climate change?")
```

# 4. Run new `weave.Evaluation` connecting to the existing models run
Finally, we can evaluate our new `RagModel` on the existing `weave.Evaluation`. To make the integration as easy as possible we include the following changes. 

From a Models perspective:
- We log the summary result of the weave Evaluation to the run used to download the fine-tuned chat model as part of the summary variable and as graphs in a new [workspace view](https://wandb.ai/wandb-smle/weave-cookboook-demo/workspace?nw=eglm8z7o9)
- We add the trace ID (with current eval ID) to the run config so that the model team can simply click on the link to go to the corresponding Weave page

From a Weave perspective:
- We save the artifact / registry link as input to the `ChatModel` (i.e. RagModel)
- We save the run.id as extra column in the traces with `weave.attributes`

```python
# MAGIC: we can simply get an evaluation with a eval dataset and scorers and use them
WEAVE_EVAL = "weave:///wandb-smle/weave-cookboook-demo/object/climate_rag_eval:ntRX6qn3Tx6w3UEVZXdhIh1BWGh7uXcQpOQnIuvnSgo"
climate_rag_eval = weave.ref(WEAVE_EVAL).get()

with weave.attributes({"wandb-run-id": wandb.run.id}):
    # use .call attribute to retrieve both the result and the call in order to save eval trace to Models
    summary, call = await climate_rag_eval.evaluate.call(climate_rag_eval, RagModel)
```

# 5. Save the new RAG Model on the Registry
In order to effectively share the new RAG Model we push it to the Regsitry as a reference artifact adding in the weave version as an alias.

```python
MODELS_OBJECT_VERSION = PUB_REFERENCE.digest  # weave object version
MODELS_OBJECT_NAME = PUB_REFERENCE.name  # weave object name

models_url = f"https://wandb.ai/{ENTITY}/{PROJECT}/weave/objects/{MODELS_OBJECT_NAME}/versions/{MODELS_OBJECT_VERSION}"
models_link = (
    f"weave:///{ENTITY}/{PROJECT}/object/{MODELS_OBJECT_NAME}:{MODELS_OBJECT_VERSION}"
)

with wandb.init(project=PROJECT, entity=ENTITY) as run:
    # create new Artifact
    artifact_model = wandb.Artifact(
        name="RagModel",
        type="model",
        description="Models Link from RagModel in Weave",
        metadata={"url": models_url},
    )
    artifact_model.add_reference(models_link, name="model", checksum=False)

    # log new artifact
    run.log_artifact(artifact_model, aliases=[MODELS_OBJECT_VERSION])

    # link to registry
    run.link_artifact(
        artifact_model, target_path="wandb32/wandb-registry-RAG Models/RAG Model"
    )
```

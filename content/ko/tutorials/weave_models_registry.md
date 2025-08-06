---
title: Weave 및 Models 인테그레이션 데모
menu:
  tutorials:
    identifier: ko-tutorials-weave_models_registry
    parent: weave-and-models-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1Uqgel6cNcGdP7AmBXe2pR9u6Dejggsh8?usp=sharing" >}}

이 노트북에서는 W&B Weave 와 W&B Models 를 함께 사용하는 방법을 보여줍니다. 이 예제에서는 두 개의 서로 다른 팀을 가정하고 있습니다.

* **Model Team:** 모델을 구축하는 팀이 새로운 Chat Model (Llama 3.2)을 파인튜닝하고, **W&B Models** 를 사용해 레지스트리에 저장합니다.
* **App Team:** 앱 개발 팀은 이 Chat Model 을 불러와 새로운 RAG 챗봇을 만들고 평가하며, 이때 **W&B Weave** 를 사용합니다.

W&B Models 와 W&B Weave 를 위한 공개 워크스페이스는 [여기](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/evaluations)에서 확인할 수 있습니다.

{{< img src="/images/tutorials/weave_models_workflow.jpg"  alt="W&B" >}}

워크플로우는 아래 과정을 포함합니다:

1. RAG 앱 코드를 W&B Weave 로 계측하기
2. LLM (예: Llama 3.2, 다른 LLM 으로 대체 가능) 을 파인튜닝하고 W&B Models 로 추적하기
3. 파인튜닝된 모델을 [W&B Registry](https://docs.wandb.ai/guides/core/registry)에 저장하기
4. 파인튜닝된 새 모델로 RAG 앱을 구현하고, W&B Weave 를 사용해 앱 평가하기
5. 결과가 만족스러우면, 업데이트된 Rag 앱을 W&B Registry 에 참조로 저장하기

**참고:**

아래에서 언급하는 `RagModel` 은 상위 레벨의 `weave.Model` 로, 완전한 RAG 앱이라고 볼 수 있습니다. 이 오브젝트는 `ChatModel`, 벡터 데이터베이스, 프롬프트를 포함합니다. `ChatModel` 역시 또 다른 `weave.Model` 이며, 여기에는 W&B Registry 로부터 artifact 를 다운로드하는 코드가 담겨 있습니다. 이 부분을 교체하여 다른 챗 모델로도 사용할 수 있습니다. 자세한 내용은 [Weave 에서 전체 모델](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/evaluations?peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2Fx7MzcgHDrGXYHHDQ9BA8N89qDwcGkdSdpxH30ubm8ZM%3F%26) 을 참고하세요.

## 1. 환경 설정
먼저, `weave` 와 `wandb`를 설치한 후 API 키로 로그인하세요. API 키 생성 및 확인은 https://wandb.ai/settings 에서 가능합니다.

```bash
pip install weave wandb
```

```python
import wandb
import weave
import pandas as pd

PROJECT = "weave-cookboook-demo"
ENTITY = "wandb-smle"

wandb.login()
weave.init(ENTITY + "/" + PROJECT)
```

## 2. 아티팩트를 기반으로 `ChatModel` 만들기

Registry 에 저장된 파인튜닝한 챗 모델을 가져와서, 이를 바탕으로 새로운 `weave.Model` 을 만들어 다음 단계에서 [`RagModel`](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2FcqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo%3F%26) 에 바로 연결합니다. 기존 [ChatModel](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-rag-experiments%2Fobjects%2FChatModelRag%2Fversions%2F2mhdPb667uoFlXStXtZ0MuYoxPaiAXj3KyLS1kYRi84%3F%26) 과 동일한 파라미터를 받되, `init` 과 `predict` 만 변경됩니다.

```bash
pip install unsloth
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

모델 팀에서는 `unsloth` 라이브러리를 활용해 다양한 Llama 3.2 모델을 더 빠르게 파인튜닝 했습니다. 따라서 Registry 에서 모델을 받아올 땐, 어댑터와 함께 `unsloth.FastLanguageModel` 또는 `peft.AutoPeftModelForCausalLM` 을 사용합니다. Registry 의 "Use" 탭에서 제공하는 로딩 코드를 복사해 `model_post_init` 에 붙여넣으세요.

```python
import weave
from pydantic import PrivateAttr
from typing import Any, List, Dict, Optional
from unsloth import FastLanguageModel
import torch


class UnslothLoRAChatModel(weave.Model):
    """
    ChatModel 클래스를 추가로 정의하여, 단순히 모델 이름뿐만 아니라 여러 파라미터도 저장하고 버전 관리합니다.
    이를 통해 특정 파라미터로 세밀하게 파인튜닝할 수 있습니다.
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
        run = wandb.init(project=PROJECT, job_type="model_download")
        artifact_ref = self.chat_model.replace("wandb-artifact:///", "")
        artifact = run.use_artifact(artifact_ref)
        model_path = artifact.download()

        # unsloth 버전 (최대 2배 빠른 추론 지원)
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.cm_max_new_tokens,
            dtype=self.dtype,
            load_in_4bit=self.cm_quantize,
        )
        FastLanguageModel.for_inference(self._model)

    @weave.op()
    async def predict(self, query: List[str]) -> dict:
        # add_generation_prompt = true - 반드시 생성 프롬프트 추가해야 함
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

이제 Registry 에서 특정 링크를 지정해 새 모델을 생성해 봅시다:

```python
ORG_ENTITY = "wandb32"  # 여기에 본인 조직 이름 입력
artifact_name = "Finetuned Llama-3.2" # 본인 아티팩트 이름으로 교체
MODEL_REG_URL = f"wandb-artifact:///{ORG_ENTITY}/wandb-registry-RAG Chat Models/{artifact_name}:v3"

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

마지막으로 아래와 같이 비동기로 평가를 실행합니다:

 ```python
 await new_chat_model.predict(
     [{"role": "user", "content": "What is the capital of Germany?"}]
 )
 ```

## 3. 새로운 `ChatModel` 버전을 `RagModel` 에 통합하기
파인튜닝된 챗 모델로 RAG 앱을 만드는 것은 특히 대화 AI 시스템의 성능과 다양성을 높이는 데 큰 장점이 있습니다.

이제 기존 Weave 프로젝트에서 [`RagModel`](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2FcqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo%3F%26) (이미지처럼 use 탭에서 현재 RagModel 의 weave ref 를 확인 가능)을 가져오고, `ChatModel` 만 방금 만든 것으로 교체합니다. 나머지 구성요소(VDB, 프롬프트 등)는 변경하거나 다시 만들 필요가 없습니다!

{{< img src="/images/tutorials/weave-ref-1.png" alt="Weave UI 'Use' tab with reference code" >}}

```bash
pip install litellm faiss-gpu
```

```python
RagModel = weave.ref(
    "weave:///wandb-smle/weave-cookboook-demo/object/RagModel:cqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo"
).get()
# MAGIC: chat_model 만 교체해 새 버전 배포 (기존 RAG 컴포넌트는 변경 걱정 불필요)
RagModel.chat_model = new_chat_model
# 예측시 참조될 수 있게 먼저 새 버전을 publish
PUB_REFERENCE = weave.publish(RagModel, "RagModel")
await RagModel.predict("When was the first conference on climate change?")
```

## 4. 기존 모델 run 과 연결되는 새로운 `weave.Evaluation` 실행
이제 새로 만든 `RagModel` 을 기존 `weave.Evaluation` 으로 평가해봅니다. 통합이 쉽도록 아래 변경사항을 참고하세요.

Models 관점:
- Registry 에서 모델을 받아오면 새로운 `run` 오브젝트가 만들어지며, 이는 챗 모델 E2E 계보의 일부가 됩니다
- run 설정에 Trace ID(현재 eval ID)를 추가하여, 모델 팀이 Weave 페이지로 바로 이동할 수 있게 합니다

Weave 관점:
- artifact / registry 링크를 `ChatModel` (즉, `RagModel`) 의 입력값으로 저장합니다
- `weave.attributes` 를 사용해 traces 의 컬럼에 run.id 를 추가로 저장합니다

```python
# MAGIC: 평가 데이터셋과 스코어러가 포함된 evaluation 을 불러와 사용
WEAVE_EVAL = "weave:///wandb-smle/weave-cookboook-demo/object/climate_rag_eval:ntRX6qn3Tx6w3UEVZXdhIh1BWGh7uXcQpOQnIuvnSgo"
climate_rag_eval = weave.ref(WEAVE_EVAL).get()

run = wandb.init()

with weave.attributes({"wandb-run-id": run.id}):
    # .call 속성을 통해 결과와 호출 모두 얻어서, eval trace 를 Models 에 저장
    summary, call = await climate_rag_eval.evaluate.call(climate_rag_eval, ` RagModel `)
```

## 5. 새 RAG 모델을 Registry 에 저장하기
새 RAG Model 을 효과적으로 공유하려면, 레퍼런스 아티팩트로 Registry 에 push 하고 weave 버전을 에일리어스로 추가하세요.

```python
MODELS_OBJECT_VERSION = PUB_REFERENCE.digest  # weave 오브젝트 버전
MODELS_OBJECT_NAME = PUB_REFERENCE.name  # weave 오브젝트 이름

models_url = f"https://wandb.ai/{ENTITY}/{PROJECT}/weave/objects/{MODELS_OBJECT_NAME}/versions/{MODELS_OBJECT_VERSION}"
models_link = (
    f"weave:///{ENTITY}/{PROJECT}/object/{MODELS_OBJECT_NAME}:{MODELS_OBJECT_VERSION}"
)

with wandb.init(project=PROJECT, entity=ENTITY) as run:
    # 새 Artifact 생성
    artifact_model = wandb.Artifact(
        name="RagModel",
        type="model",
        description="Weave 에서 생성된 RagModel 링크",
        metadata={"url": models_url},
    )
    artifact_model.add_reference(models_link, name="model", checksum=False)

    # 새 artifact 로그 남기기
    run.log_artifact(artifact_model, aliases=[MODELS_OBJECT_VERSION])

    # registry 에 연결
    run.link_artifact(
        artifact_model, target_path="wandb32/wandb-registry-RAG Models/RAG Model"
    )
```
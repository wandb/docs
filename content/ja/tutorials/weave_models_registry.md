---
title: Weave and Models integration demo
menu:
  tutorials:
    identifier: ja-tutorials-weave_models_registry
    parent: weave-and-models-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1Uqgel6cNcGdP7AmBXe2pR9u6Dejggsh8?usp=sharing" >}}

このノートブックでは、W&B Weave を W&B Models と共に使用する方法を紹介します。具体的には、この例では 2 つの異なる Teams を検討します。

* **The Model Team:** モデル構築 Team は、新しい Chat Model (Llama 3.2) を fine-tune し、**W&B Models** を使用して Registry に保存します。
* **The App Team:** アプリ開発 Team は、Chat Model を取得して **W&B Weave** を使用して新しい RAG chatbot を作成および評価します。

W&B Models と W&B Weave の両方のパブリック Workspace は[こちら](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/evaluations)にあります。

{{< img src="/images/tutorials/weave_models_workflow.jpg"  alt="Weights & Biases" >}}

この ワークフロー では、次のステップを実行します。

1. W&B Weave で RAG アプリの code をインストルメント化します。
2. LLM (Llama 3.2 など。他の LLM に置き換えることも可能) を fine-tune し、W&B Models で追跡します。
3. fine-tune された model を[W&B Registry](https://docs.wandb.ai/guides/registry)に ログ します。
4. 新しい fine-tune された model で RAG アプリを実装し、W&B Weave でアプリを評価します。
5. 結果に満足したら、更新された Rag アプリへの参照を W&B Registry に保存します。

**Note:**

下記で参照されている `RagModel` は、完全な RAG アプリと見なすことができるトップレベルの `weave.Model` です。これには、`ChatModel`、Vector database、および Prompt が含まれています。`ChatModel` は別の `weave.Model` でもあります。これには、W&B Registry から Artifact をダウンロードする code が含まれており、`RagModel` の一部として他の Chat Model をサポートするように変更できます。詳細については、[Weave の完全な model ](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/evaluations?peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2Fx7MzcgHDrGXYHHDQ9BA8N89qDwcGkdSdpxH30ubm8ZM%3F%26)を参照してください。

## 1. Setup
まず、`weave` と `wandb` をインストールし、次に APIキー で ログイン します。API キー は、https://wandb.ai/settings で作成および表示できます。

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

## 2. Artifact に基づいて `ChatModel` を作成する

Registry から fine-tune された chat model を取得し、それから `weave.Model` を作成して、次のステップで [`RagModel`](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2FcqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo%3F%26)に直接接続します。既存の[ChatModel](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-rag-experiments%2Fobjects%2FChatModelRag%2Fversions%2F2mhdPb667uoFlXStXtZ0MuYoxPaiAXj3KyLS1kYRi84%3F%26)と同じ パラメータ を受け取ります。`init` と `predict` のみが変更されます。

```bash
pip install unsloth
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

Model Team は、`unsloth` ライブラリを使用して、さまざまな Llama-3.2 models を fine-tune し、高速化しました。したがって、Registry からダウンロードした model をロードするために、adapter を使用して特別な `unsloth.FastLanguageModel` または `peft.AutoPeftModelForCausalLM` models を使用します。「Use」タブからローディング code をコピーして、`model_post_init` に貼り付けます。

```python
import weave
from pydantic import PrivateAttr
from typing import Any, List, Dict, Optional
from unsloth import FastLanguageModel
import torch


class UnslothLoRAChatModel(weave.Model):
    """
    model 名だけでなく、より多くの パラメータ を保存および バージョン 管理するための追加の ChatModel クラスを定義します。
    これにより、特定の パラメータ での fine-tuning が可能になります。
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
        # これを Registry の [Use] タブから貼り付けます
        run = wandb.init(project=PROJECT, job_type="model_download")
        artifact = run.use_artifact(f"{self.chat_model}")
        model_path = artifact.download()

        # unsloth バージョン (ネイティブ 2 倍高速推論を有効にします)
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.cm_max_new_tokens,
            dtype=self.dtype,
            load_in_4bit=self.cm_quantize,
        )
        FastLanguageModel.for_inference(self._model)

    @weave.op()
    async def predict(self, query: List[str]) -> dict:
        # add_generation_prompt = true - 生成には必ず追加してください
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

次に、Registry からの特定の リンク を使用して新しい model を作成します。

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

最後に、評価を非同期的に実行します。

 ```python
 await new_chat_model.predict(
     [{"role": "user", "content": "What is the capital of Germany?"}]
 )
 ```

## 3. 新しい `ChatModel` バージョンを `RagModel` に統合する
fine-tune された chat model から RAG アプリを構築すると、特に会話型 AI システムの パフォーマンス と汎用性を向上させる上で、いくつかのメリットが得られます。

次に、既存の Weave プロジェクトから [`RagModel`](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2FcqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo%3F%26) を取得し (下の画像に示すように、[Use] タブから現在の `RagModel` の Weave 参照を取得できます)、`ChatModel` を新しいものに交換します。他のコンポーネント (VDB、prompts など) を変更または再作成する必要はありません。

<img src="/images/tutorials/weave-ref-1.png"  alt="Weights & Biases" />

```bash
pip install litellm faiss-gpu
```

```python
RagModel = weave.ref(
    "weave:///wandb-smle/weave-cookboook-demo/object/RagModel:cqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo"
).get()
# MAGIC: chat_model を交換して新しい バージョン を公開します (他の RAG コンポーネントを気にする必要はありません)
RagModel.chat_model = new_chat_model
# 予測中に参照されるように、最初に新しい バージョン を公開します
PUB_REFERENCE = weave.publish(RagModel, "RagModel")
await RagModel.predict("When was the first conference on climate change?")
```

## 4. 既存の models run に接続する新しい `weave.Evaluation` を実行する
最後に、既存の `weave.Evaluation` で新しい `RagModel` を評価します。統合をできるだけ簡単にするために、次の変更を含めます。

Models の観点から:
- Registry から model を取得すると、chat model の E2E リネージ の一部である新しい `wandb.run` が作成されます
- 対応する Weave ページに移動するために、Trace ID (現在の eval ID を使用) を run config に追加して、Model Team が リンク をクリックできるようにします

Weave の観点から:
- Artifact / Registry リンク を `ChatModel` (つまり `RagModel`) への入力として保存します
- `weave.attributes` を使用して、run.id を Trace の追加の列として保存します

```python
# MAGIC: 評価 データセット と scorer を使用して評価を取得し、それらを使用します
WEAVE_EVAL = "weave:///wandb-smle/weave-cookboook-demo/object/climate_rag_eval:ntRX6qn3Tx6w3UEVZXdhIh1BWGh7uXcQpOQnIuvnSgo"
climate_rag_eval = weave.ref(WEAVE_EVAL).get()

with weave.attributes({"wandb-run-id": wandb.run.id}):
    # Models に eval Trace を保存するために、結果と 呼び出し の両方を取得するには、.call 属性を使用します
    summary, call = await climate_rag_eval.evaluate.call(climate_rag_eval, ` RagModel `)
```

## 5. Registry に新しい RAG model を保存する
新しい RAG Model を効果的に共有するために、Weave バージョン を エイリアス として追加して、参照 Artifact として Registry に push します。

```python
MODELS_OBJECT_VERSION = PUB_REFERENCE.digest  # weave object バージョン
MODELS_OBJECT_NAME = PUB_REFERENCE.name  # weave object 名

models_url = f"https://wandb.ai/{ENTITY}/{PROJECT}/weave/objects/{MODELS_OBJECT_NAME}/versions/{MODELS_OBJECT_VERSION}"
models_link = (
    f"weave:///{ENTITY}/{PROJECT}/object/{MODELS_OBJECT_NAME}:{MODELS_OBJECT_VERSION}"
)

with wandb.init(project=PROJECT, entity=ENTITY) as run:
    # 新しい Artifact を作成する
    artifact_model = wandb.Artifact(
        name="RagModel",
        type="model",
        description="Weave の RagModel からの Models リンク",
        metadata={"url": models_url},
    )
    artifact_model.add_reference(models_link, name="model", checksum=False)

    # 新しい Artifact を ログ します
    run.log_artifact(artifact_model, aliases=[MODELS_OBJECT_VERSION])

    # Registry への リンク
    run.link_artifact(
        artifact_model, target_path="wandb32/wandb-registry-RAG Models/RAG Model"
    )
```
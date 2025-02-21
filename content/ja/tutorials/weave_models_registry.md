---
title: Weave and Models integration demo
menu:
  tutorials:
    identifier: ja-tutorials-weave_models_registry
    parent: weave-and-models-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1Uqgel6cNcGdP7AmBXe2pR9u6Dejggsh8?usp=sharing" >}}

このノートブックでは、W&B Modelsと一緒にW&B Weaveを使用する方法を示します。具体的には、2つの異なるチームに焦点を当てた例です。

* **The Model Team:** モデル構築チームは、新しいChat Model (Llama 3.2)をファインチューンし、**W&B Models**を使用してレジストリに保存します。
* **The App Team:** アプリ開発チームは、**W&B Weave**を使用してChat Modelを取得し、新しいRAGチャットボットを作成し、評価します。

W&B ModelsとW&B Weaveの公共ワークスペースは[こちら](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/evaluations)です。

{{< img src="/images/tutorials/weave_models_workflow.jpg"  alt="Weights & Biases" >}}

ワークフローは以下のステップをカバーします:

1. RAGアプリのコードをW&B Weaveでインストゥルメント化
2. LLM（Llama 3.2など）をファインチューンし、それをW&B Modelsでトラッキング
3. ファインチューンしたモデルを[W&B Registry](https://docs.wandb.ai/guides/registry)にログ
4. 新しいファインチューンしたモデルでRAGアプリを実装し、W&B Weaveでアプリを評価
5. 結果に満足したら、W&B Registryに更新されたRagアプリの参照を保存

**注意:**

以下で参照されている`RagModel`は、基本となる`weave.Model`であり、完全なRAGアプリとみなすことができます。`ChatModel`、ベクトルデータベース、プロンプトを含みます。`ChatModel`もまた、W&B Registryからアーティファクトをダウンロードするコードを含む`weave.Model`であり、`RagModel`の一部として他のチャットモデルをサポートするために変更できます。詳細は[Weaveでの完全なモデル](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/evaluations?peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2Fx7MzcgHDrGXYHHDQ9BA8N89qDwcGkdSdpxH30ubm8ZM%3F%26)を参照してください。

## 1. セットアップ
まず、`weave`と`wandb`をインストールし、APIキーでログインします。APIキーは https://wandb.ai/settings で作成したり確認できます。

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

## 2. アーティファクトに基づいた `ChatModel` を作成

ファインチューンされたチャットモデルをRegistryから取得し、次のステップで[`RagModel`](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2FcqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo%3F%26)に直接プラグインする`weave.Model`を作成します。これは、既存の[ChatModel](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-rag-experiments%2Fobjects%2FChatModelRag%2Fversions%2F2mhdPb667uoFlXStXtZ0MuYoxPaiAXj3KyLS1kYRi84%3F%26)と同じパラメータを取りますが、`init`と`predict`が変更されます。

```bash
pip install unsloth
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

モデルチームは、`unsloth`ライブラリを使用して異なるLlama-3.2モデルをファインチューンし、高速化しました。したがって、`unsloth.FastLanguageModel`またはアダプターを使用した`peft.AutoPeftModelForCausalLM`モデルを使用し、Registryからダウンロードされたモデルをロードします。"Use"タブからロードコードをコピーし、`model_post_init`に貼り付けてください。

```python
import weave
from pydantic import PrivateAttr
from typing import Any, List, Dict, Optional
from unsloth import FastLanguageModel
import torch


class UnslothLoRAChatModel(weave.Model):
    """
    モデル名だけでなく、より多くのパラメータを保存およびバージョン化するために、追加のChatModelクラスを定義します。
    これにより、特定のパラメータに対するファインチューンが可能になります。
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
        # レジストリの"Use"タブからこのコードを貼り付けます
        run = wandb.init(project=PROJECT, job_type="model_download")
        artifact = run.use_artifact(f"{self.chat_model}")
        model_path = artifact.download()

        # unslothバージョン（ネイティブで2倍速い推論を可能に）
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.cm_max_new_tokens,
            dtype=self.dtype,
            load_in_4bit=self.cm_quantize,
        )
        FastLanguageModel.for_inference(self._model)

    @weave.op()
    async def predict(self, query: List[str]) -> dict:
        # add_generation_prompt = true - 生成するために追加する必要があります
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

レジストリから特定のリンクで新しいモデルを作成します：

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

そして最後に評価を非同期で実行します：

```python
await new_chat_model.predict(
    [{"role": "user", "content": "What is the capital of Germany?"}]
)
```

## 3. `RagModel` に新しい `ChatModel` バージョンを統合
ファインチューンされたチャットモデルからRAGアプリを構築することは、特に会話型AIシステムの性能と多様性を向上させることにおいて、いくつかの利点を提供します。

次に、既存のWeaveプロジェクトから[`RagModel`](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2FcqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo%3F%26)を取得し、新しいものと`ChatModel`を交換します。他のコンポーネント（VDB、プロンプトなど）を変更したり再作成したりする必要はありません！

<img src="/images/tutorials/weave-ref-1.png"  alt="Weights & Biases" />

```bash
pip install litellm faiss-gpu
```

```python
RagModel = weave.ref(
    "weave:///wandb-smle/weave-cookboook-demo/object/RagModel:cqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo"
).get()
# MAGIC: chat_modelを交換して新しいバージョンを公開（他のRAGコンポーネントについて心配する必要はありません）
RagModel.chat_model = new_chat_model
# 予測中に参照されるように 最初に新しいバージョンを公開します
PUB_REFERENCE = weave.publish(RagModel, "RagModel")
await RagModel.predict("When was the first conference on climate change?")
```

## 4. 既存のモデルrunに接続して新しい `weave.Evaluation` を実行
最後に、既存の`weave.Evaluation`で新しい`RagModel`を評価します。インテグレーションを可能な限り簡単にするために、次の変更を含めます。

Modelsの観点から：
- レジストリからモデルを取得すると、chat modelのE2Eリネージの一部である新しい`wandb.run`が作成されます
- Run configにTrace ID（現在の評価ID付き）を追加し、モデルチームがリンクをクリックして対応するWeaveページに移動できるようにします

Weaveの観点から：
- アーティファクト / レジストリリンクを`ChatModel`の入力として保存する（つまり`RagModel`）
- `weave.attributes`でtraceにrun.idをエクストラカラムとして保存

```python
# MAGIC: 評価データセットとスコアラーで評価を取得して使用
WEAVE_EVAL = "weave:///wandb-smle/weave-cookboook-demo/object/climate_rag_eval:ntRX6qn3Tx6w3UEVZXdhIh1BWGh7uXcQpOQnIuvnSgo"
climate_rag_eval = weave.ref(WEAVE_EVAL).get()

with weave.attributes({"wandb-run-id": wandb.run.id}):
    # .call属性を使用して結果とcallの両方を取得し、Evaluation TraceをModelsに保存
    summary, call = await climate_rag_eval.evaluate.call(climate_rag_eval, ` RagModel `)
```

## 5. 新しいRAGモデルをRegistryに保存
新しいRAGモデルを効果的に共有するために、weaveバージョンをエイリアスとして追加して、参照アーティファクトとしてRegistryにプッシュします。

```python
MODELS_OBJECT_VERSION = PUB_REFERENCE.digest  # weave object version
MODELS_OBJECT_NAME = PUB_REFERENCE.name  # weave object name

models_url = f"https://wandb.ai/{ENTITY}/{PROJECT}/weave/objects/{MODELS_OBJECT_NAME}/versions/{MODELS_OBJECT_VERSION}"
models_link = (
    f"weave:///{ENTITY}/{PROJECT}/object/{MODELS_OBJECT_NAME}:{MODELS_OBJECT_VERSION}"
)

with wandb.init(project=PROJECT, entity=ENTITY) as run:
    # 新しいアーティファクトを作成
    artifact_model = wandb.Artifact(
        name="RagModel",
        type="model",
        description="WeaveのRagModelからのModelsリンク",
        metadata={"url": models_url},
    )
    artifact_model.add_reference(models_link, name="model", checksum=False)

    # 新しいアーティファクトをログ
    run.log_artifact(artifact_model, aliases=[MODELS_OBJECT_VERSION])

    # レジストリにリンク
    run.link_artifact(
        artifact_model, target_path="wandb32/wandb-registry-RAG Models/RAG Model"
    )
```
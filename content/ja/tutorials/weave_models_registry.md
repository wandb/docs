---
title: Weave と Models インテグレーションのデモ
menu:
  tutorials:
    identifier: weave_models_registry
    parent: weave-and-models-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1Uqgel6cNcGdP7AmBXe2pR9u6Dejggsh8?usp=sharing" >}}

このノートブックでは、W&B Weave と W&B Models を組み合わせて使う方法を紹介します。具体的には、2つの異なるチームを例に説明します。

* **Model Team：** モデル構築チームは、新しい Chat Model（Llama 3.2）をファインチューンし、**W&B Models** を使ってレジストリに保存します。
* **App Team：** アプリ開発チームは、Chat Model を取得し、新しい RAG チャットボットを作成・評価する際に **W&B Weave** を活用します。

W&B Models および W&B Weave の公開ワークスペースは[こちら](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/evaluations)から閲覧できます。

{{< img src="/images/tutorials/weave_models_workflow.jpg"  alt="W&B" >}}

このワークフローでは、以下のステップをカバーしています。

1. RAG アプリのコードを W&B Weave でインスツルメント化する
2. LLM（例：Llama 3.2。ただし他の LLM でも可）をファインチューンし、W&B Models で追跡する
3. ファインチューン済みモデルを [W&B Registry](https://docs.wandb.ai/guides/core/registry) へログする
4. 新しいファインチューンモデルを使って RAG アプリを実装し、W&B Weave でアプリを評価する
5. 結果に満足したら、更新済み Rag アプリの参照を W&B Registry に保存する

**補足：**

以下で参照されている `RagModel` は、トップレベルの `weave.Model` です。これは RAG アプリ全体を指し、中に `ChatModel`、ベクターデータベース、プロンプトが含まれます。`ChatModel` も別の `weave.Model` で、W&B Registry からアーティファクトをダウンロードするコードを含み、`RagModel` の一部として他の任意のチャットモデルもサポート可能です。詳しくは [Weave 上の完全なモデル](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/evaluations?peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2Fx7MzcgHDrGXYHHDQ9BA8N89qDwcGkdSdpxH30ubm8ZM%3F%26) をご参照ください。

## 1. セットアップ
まず、`weave` と `wandb` をインストールし、APIキーでログインします。APIキーの発行・確認は https://wandb.ai/settings から行えます。

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

## 2. Artifact ベースの `ChatModel` を作成

Registry からファインチューン済みチャットモデルを取得し、それを元に `weave.Model` を作成します。次のステップで [`RagModel`](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2FcqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo%3F%26) へ直接組み込みます。既存 [ChatModel](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-rag-experiments%2Fobjects%2FChatModelRag%2Fversions%2F2mhdPb667uoFlXStXtZ0MuYoxPaiAXj3KyLS1kYRi84%3F%26) と同じパラメータを取り、`init` と `predict` で変更があります。

```bash
pip install unsloth
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

モデルチームは `unsloth` ライブラリを使って Llama-3.2 モデルを高速にファインチューンしました。そのため、Registry からダウンロードしたモデルは専用の `unsloth.FastLanguageModel` や、アダプター付きの `peft.AutoPeftModelForCausalLM` で読み込んでいます。Registry の "Use" タブから読み込み用コードを `model_post_init` に貼り付けてください。

```python
import weave
from pydantic import PrivateAttr
from typing import Any, List, Dict, Optional
from unsloth import FastLanguageModel
import torch

# 追加の ChatModel クラスを定義し、モデル名以外のパラメータも保存・バージョン管理
# これにより特定パラメータでのファインチューンが可能になります
class UnslothLoRAChatModel(weave.Model):
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

        # unsloth バージョン（高速推論対応）
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.cm_max_new_tokens,
            dtype=self.dtype,
            load_in_4bit=self.cm_quantize,
        )
        FastLanguageModel.for_inference(self._model)

    @weave.op()
    async def predict(self, query: List[str]) -> dict:
        # add_generation_prompt = true - 生成用プロンプト追加
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

続いて、レジストリから特定リンクのモデルを作成します：

```python
ORG_ENTITY = "wandb32"  # ご自身の組織名に置き換えてください
artifact_name = "Finetuned Llama-3.2" # ご自身のアーティファクト名に置き換えてください
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

最後に非同期で評価を実行します：

```python
await new_chat_model.predict(
    [{"role": "user", "content": "What is the capital of Germany?"}]
)
```

## 3. 新しい `ChatModel` バージョンを `RagModel` に統合
ファインチューン済みチャットモデルから RAG アプリを構築することで、会話型 AI システムの性能や多様性を大幅に向上させることができます。

既存の Weave プロジェクトから [`RagModel`](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2FcqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo%3F%26) を取得し、`ChatModel` を新しいものに差し替えます。（下図のように Weave UI の "Use" タブから weave ref を取得可能です）。他のコンポーネント（VDB, prompts など）は変更不要です！

{{< img src="/images/tutorials/weave-ref-1.png" alt="Weave UI 『Use』タブの参照コード" >}}

```bash
pip install litellm faiss-gpu
```

```python
RagModel = weave.ref(
    "weave:///wandb-smle/weave-cookboook-demo/object/RagModel:cqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo"
).get()
# MAGIC: chat_model を入れ替えて新バージョンを公開（他のRAGコンポーネントは変更不要）
RagModel.chat_model = new_chat_model
# 新しいバージョンを公開し、予測時に参照されるようにします
PUB_REFERENCE = weave.publish(RagModel, "RagModel")
await RagModel.predict("When was the first conference on climate change?")
```

## 4. 既存 models run に接続した新しい `weave.Evaluation` を実行
最後に、新しい `RagModel` を既存の `weave.Evaluation` で評価します。できるだけ簡単に統合できるよう以下の調整を行います。

Models 側の観点：
- レジストリからモデルを取得すると、新しい `run` オブジェクトが生成され、チャットモデルの E2E リネージに含まれます
- Trace ID（現 eval ID）を run の config に追加して、モデルチームが Weave ページへ直接アクセスできるようにします

Weave 側の観点：
- `ChatModel`（= `RagModel`）にアーティファクト／レジストリリンクを入力として保存
- traces に run.id を `weave.attributes` で追加カラムとして保存

```python
# MAGIC: eval データセットとスコアラーを含んだ評価を取得し利用
WEAVE_EVAL = "weave:///wandb-smle/weave-cookboook-demo/object/climate_rag_eval:ntRX6qn3Tx6w3UEVZXdhIh1BWGh7uXcQpOQnIuvnSgo"
climate_rag_eval = weave.ref(WEAVE_EVAL).get()

run = wandb.init()

with weave.attributes({"wandb-run-id": run.id}):
    # .call 属性を使い、結果と call の両方を取得。評価トレースを Models に保存
    summary, call = await climate_rag_eval.evaluate.call(climate_rag_eval, ` RagModel `)
```

## 5. 新しい RAG モデルを Registry に保存
新しい RAG Model を効果的に共有するため、weave バージョンをエイリアスとして追加し、参照アーティファクトとして Registry にプッシュします。

```python
MODELS_OBJECT_VERSION = PUB_REFERENCE.digest  # weave オブジェクトバージョン
MODELS_OBJECT_NAME = PUB_REFERENCE.name  # weave オブジェクト名

models_url = f"https://wandb.ai/{ENTITY}/{PROJECT}/weave/objects/{MODELS_OBJECT_NAME}/versions/{MODELS_OBJECT_VERSION}"
models_link = (
    f"weave:///{ENTITY}/{PROJECT}/object/{MODELS_OBJECT_NAME}:{MODELS_OBJECT_VERSION}"
)

with wandb.init(project=PROJECT, entity=ENTITY) as run:
    # 新しい Artifact を作成
    artifact_model = wandb.Artifact(
        name="RagModel",
        type="model",
        description="Models Link from RagModel in Weave",
        metadata={"url": models_url},
    )
    artifact_model.add_reference(models_link, name="model", checksum=False)

    # 新しい artifact をログ
    run.log_artifact(artifact_model, aliases=[MODELS_OBJECT_VERSION])

    # registry へリンク
    run.link_artifact(
        artifact_model, target_path="wandb32/wandb-registry-RAG Models/RAG Model"
    )
```
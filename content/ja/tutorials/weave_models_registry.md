---
title: Weave と Models インテグレーションのデモ
menu:
  tutorials:
    identifier: ja-tutorials-weave_models_registry
    parent: weave-and-models-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1Uqgel6cNcGdP7AmBXe2pR9u6Dejggsh8?usp=sharing" >}}

このノートブックでは、W&B Weave と W&B Models を組み合わせて使う方法を紹介します。特に、この例では 2 つの異なるチームによる活用を想定しています。

* **モデルチーム:** モデル開発チームが Chat モデル（Llama 3.2）をファインチューンし、**W&B Models** を使ってレジストリに保存します。
* **アプリチーム:** アプリ開発チームは、その Chat モデルを取得して新しい RAG チャットボットを作成・評価します（**W&B Weave** を利用）。

W&B Models と W&B Weave の両方のパブリックワークスペースは [こちら](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/evaluations) から参照できます。

{{< img src="/images/tutorials/weave_models_workflow.jpg"  alt="W&B" >}}

このワークフローでは、以下の手順をカバーします：

1. RAG アプリのコードに W&B Weave を組み込みます
2. LLM（Llama 3.2 など。お好きな LLM でも可）をファインチューンし、W&B Models で管理します
3. ファインチューン済みモデルを [W&B Registry](https://docs.wandb.ai/guides/core/registry) に保存します
4. 新しいファインチューン済みモデルで RAG アプリを作成し、W&B Weave でアプリの評価を行います
5. 結果に満足したら、更新された Rag アプリの参照を W&B Registry に保存します

**補足：**

この後で登場する `RagModel` は、最上位の `weave.Model` で、RAG アプリ全体と考えて構いません。`RagModel` には `ChatModel`、ベクトルデータベース、そしてプロンプトが含まれています。`ChatModel` もまた別の `weave.Model` で、W&B Registry から artifact をダウンロードするコードを保持しています。`RagModel` の一部として、他のチャットモデルにも差し替えられる設計です。詳細は [Weave 上の完全なモデル](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/evaluations?peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2Fx7MzcgHDrGXYHHDQ9BA8N89qDwcGkdSdpxH30ubm8ZM%3F%26) をご覧ください。

## 1. セットアップ

まず `weave` と `wandb` をインストールし、APIキーでログインします。APIキーの作成・確認は https://wandb.ai/settings でできます。

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

## 2. Artifact ベースで `ChatModel` を作成

Registry からファインチューン済みチャットモデルを取得し、それを `weave.Model` に変換して、次のステップで直接 [`RagModel`](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2FcqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo%3F%26) に接続します。既存の [ChatModel](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-rag-experiments%2Fobjects%2FChatModelRag%2Fversions%2F2mhdPb667uoFlXStXtZ0MuYoxPaiAXj3KyLS1kYRi84%3F%26) と同じパラメータを受け取り、`init` と `predict` のみ変更します。

```bash
pip install unsloth
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

モデルチームは `unsloth` ライブラリを使い、複数の Llama-3.2 モデルをより高速にファインチューンしています。そのため、ダウンロードしたモデルを Registry から読み込む時は、`unsloth.FastLanguageModel` もしくはアダプター付きの `peft.AutoPeftModelForCausalLM` モデルを使ってください。Registry の「Use」タブからロード用のコードをコピーし、`model_post_init` に貼り付けてください。

```python
import weave
from pydantic import PrivateAttr
from typing import Any, List, Dict, Optional
from unsloth import FastLanguageModel
import torch


class UnslothLoRAChatModel(weave.Model):
    """
    モデル名だけでなく、追加のパラメータも保存・バージョン管理できる ChatModel クラスを定義します。
    これにより特定のパラメータでファインチューンした内容も管理できます。
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

        # unsloth バージョン（2倍のスピードで推論可能）
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.cm_max_new_tokens,
            dtype=self.dtype,
            load_in_4bit=self.cm_quantize,
        )
        FastLanguageModel.for_inference(self._model)

    @weave.op()
    async def predict(self, query: List[str]) -> dict:
        # add_generation_prompt = true - 生成のために必須
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

次に、Registry から指定したリンクを使って新しいモデルを作成します。

```python
ORG_ENTITY = "wandb32"  # ここを所属組織名に置き換えてください
artifact_name = "Finetuned Llama-3.2" # ここを artifact 名に置き換えてください
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

最後に、評価を非同期で実行します：

 ```python
 await new_chat_model.predict(
     [{"role": "user", "content": "What is the capital of Germany?"}]
 )
 ```

## 3. 新しい `ChatModel` バージョンを `RagModel` に統合

ファインチューン済みチャットモデルを基盤に RAG アプリを構築することで、会話型 AI システムの性能と柔軟性を大きく高めることができます。

既存の Weave プロジェクトから [`RagModel`](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2FcqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo%3F%26) を取得し（下の画像のように「Use」タブから weave ref を取得可能）、`ChatModel` を新しいものに交換します。他のコンポーネント（VDB、プロンプトなど）の変更や再作成は不要です！

{{< img src="/images/tutorials/weave-ref-1.png" alt="Weave UI 'Use' tab with reference code" >}}

```bash
pip install litellm faiss-gpu
```

```python
RagModel = weave.ref(
    "weave:///wandb-smle/weave-cookboook-demo/object/RagModel:cqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo"
).get()
# MAGIC: chat_model を入れ替えて新バージョンを公開（他の RAG コンポーネントはそのままなので変更の必要なし）
RagModel.chat_model = new_chat_model
# 最初に新バージョンを publish し、予測時にはこの参照を使用します
PUB_REFERENCE = weave.publish(RagModel, "RagModel")
await RagModel.predict("When was the first conference on climate change?")
```

## 4. 既存モデル run につないで新しい `weave.Evaluation` を実行

最後に、新しい `RagModel` を既存の `weave.Evaluation` で評価します。統合を簡単にするため、以下の変更を推奨します。

Models 側の観点：

- レジストリからモデルを取得すると、新たな `run` オブジェクト（チャットモデルの E2E リネージの一部）が生成されます
- Trace ID（現在の eval ID）を run config に加えることで、モデルチームが Weave の該当ページへすぐアクセスできます

Weave 側の観点：

- アーティファクト／registry のリンクを `ChatModel`（すなわち `RagModel`）の入力として保存
- run.id を trace の追加カラムとして `weave.attributes` で記録

```python
# MAGIC: eval dataset・scorer付きで評価を入手し、実行する
WEAVE_EVAL = "weave:///wandb-smle/weave-cookboook-demo/object/climate_rag_eval:ntRX6qn3Tx6w3UEVZXdhIh1BWGh7uXcQpOQnIuvnSgo"
climate_rag_eval = weave.ref(WEAVE_EVAL).get()

run = wandb.init()

with weave.attributes({"wandb-run-id": run.id}):
    # .call 属性を使い、結果と呼び出し情報の両方を取得（eval trace を Models 側に保存可能）
    summary, call = await climate_rag_eval.evaluate.call(climate_rag_eval, ` RagModel `)
```

## 5. 新しい RAG モデルを Registry に保存

新しい RAG モデルを効果的に共有するには、参照 artifact の形で Registry に push し、Weave のバージョンをエイリアスとして追加します。

```python
MODELS_OBJECT_VERSION = PUB_REFERENCE.digest  # weave object version
MODELS_OBJECT_NAME = PUB_REFERENCE.name  # weave object name

models_url = f"https://wandb.ai/{ENTITY}/{PROJECT}/weave/objects/{MODELS_OBJECT_NAME}/versions/{MODELS_OBJECT_VERSION}"
models_link = (
    f"weave:///{ENTITY}/{PROJECT}/object/{MODELS_OBJECT_NAME}:{MODELS_OBJECT_VERSION}"
)

with wandb.init(project=PROJECT, entity=ENTITY) as run:
    # 新規 Artifact 作成
    artifact_model = wandb.Artifact(
        name="RagModel",
        type="model",
        description="Models Link from RagModel in Weave",
        metadata={"url": models_url},
    )
    artifact_model.add_reference(models_link, name="model", checksum=False)

    # artifact を新規ログ
    run.log_artifact(artifact_model, aliases=[MODELS_OBJECT_VERSION])

    # registry にリンク
    run.link_artifact(
        artifact_model, target_path="wandb32/wandb-registry-RAG Models/RAG Model"
    )
```
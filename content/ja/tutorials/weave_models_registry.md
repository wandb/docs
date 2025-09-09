---
title: Weave と Models の インテグレーション デモ
menu:
  tutorials:
    identifier: ja-tutorials-weave_models_registry
    parent: weave-and-models-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1Uqgel6cNcGdP7AmBXe2pR9u6Dejggsh8?usp=sharing" >}}

このノートブックでは、W&B Weave と W&B Models をあわせて使う方法を紹介します。ここでは 2 つのチームを想定します。

* **The Model Team:** モデル開発チームが新しいチャットモデル（Llama 3.2）をファインチューンし、**W&B Models** を使って Registry に保存します。
* **The App Team:** アプリ開発チームがそのチャットモデルを取得し、**W&B Weave** を使って新しい RAG チャットボットを作成・評価します。

W&B Models と W&B Weave の両方の public workspace は [こちら](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/evaluations) です。

{{< img src="/images/tutorials/weave_models_workflow.jpg"  alt="W&B" >}}

このワークフローでは次の手順を扱います:

1. RAG アプリのコードを W&B Weave で計測する
2. LLM（例: Llama 3.2。任意の LLM に置き換え可能）をファインチューンし、W&B Models でトラッキングする
3. ファインチューンしたモデルを [W&B Registry](https://docs.wandb.ai/guides/core/registry) にログする
4. 新しいファインチューン済みモデルで RAG アプリを実装し、W&B Weave でアプリを評価する
5. 結果に満足したら、更新した RAG アプリの参照を W&B Registry に保存する

注意:
以下で参照している `RagModel` は、RAG アプリ全体とみなせるトップレベルの `weave.Model` です。これは `ChatModel`、ベクターデータベース、およびプロンプトを含みます。`ChatModel` も別の `weave.Model` で、W&B Registry から artifact をダウンロードするコードを含み、`RagModel` の一部として他の任意のチャットモデルに対応できるように変更できます。詳細は [Weave 上の完全なモデル](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/evaluations?peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2Fx7MzcgHDrGXYHHDQ9BA8N89qDwcGkdSdpxH30ubm8ZM%3F%26) を参照してください。

## 1. Setup
まず `weave` と `wandb` をインストールし、APIキー でログインします。APIキー の作成と確認は https://wandb.ai/settings から行えます。

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

## 2. Artifact ベースの `ChatModel` を作成する

Registry からファインチューン済みのチャットモデルを取得し、それをもとに `weave.Model` を作って、次のステップで [`RagModel`](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2FcqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo%3F%26) にそのまま差し替えられるようにします。既存の [ChatModel](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-rag-experiments%2Fobjects%2FChatModelRag%2Fversions%2F2mhdPb667uoFlXStXtZ0MuYoxPaiAXj3KyLS1kYRi84%3F%26) と同じパラメータを受け取り、`init` と `predict` だけが変わります。

```bash
pip install unsloth
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

モデルチームは推論を高速化するために `unsloth` ライブラリを使って複数の Llama-3.2 モデルをファインチューンしました。そのため、Registry からダウンロード後の読み込みには、アダプタ付きの特別な `unsloth.FastLanguageModel` または `peft.AutoPeftModelForCausalLM` を使用します。Registry の「Use」タブからロード用コードをコピーし、`model_post_init` に貼り付けてください。

```python
import weave
from pydantic import PrivateAttr
from typing import Any, List, Dict, Optional
from unsloth import FastLanguageModel
import torch


class UnslothLoRAChatModel(weave.Model):
    """
    モデル名だけでなく、より多くのパラメータを保存・バージョニングするための追加の ChatModel クラスを定義します。
    これにより特定のパラメータに対するファインチューニングが可能になります。
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

        # unsloth バージョン（ネイティブで 2x 高速な推論を有効化）
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.cm_max_new_tokens,
            dtype=self.dtype,
            load_in_4bit=self.cm_quantize,
        )
        FastLanguageModel.for_inference(self._model)

    @weave.op()
    async def predict(self, query: List[str]) -> dict:
        # add_generation_prompt = true - 生成には必須
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

次に、Registry からの特定のリンクで新しいモデルを作成します:

```python
ORG_ENTITY = "wandb32"  # 自分の組織名に置き換えてください
artifact_name = "Finetuned Llama-3.2" # 自分の artifact 名に置き換えてください
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

最後に、評価を非同期で実行します:

```python
await new_chat_model.predict(
    [{"role": "user", "content": "What is the capital of Germany?"}]
)
```

## 3. 新しい `ChatModel` バージョンを `RagModel` に統合する
ファインチューン済みチャットモデルから RAG アプリを構築すると、会話型 AI システムの性能と汎用性を高められるなど、いくつかの利点があります。

既存の Weave project から [`RagModel`](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/object-versions?filter=%7B%22objectName%22%3A%22RagModel%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fobjects%2FRagModel%2Fversions%2FcqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo%3F%26) を取得し（下の画像のように Use タブから現在の `RagModel` の weave ref を取得できます）、`ChatModel` を新しいものに差し替えます。他のコンポーネント（VDB、プロンプトなど）を変更・再作成する必要はありません。

{{< img src="/images/tutorials/weave-ref-1.png" alt="Weave の UI にある「Use」タブの参照コード" >}}

```bash
pip install litellm faiss-gpu
```

```python
RagModel = weave.ref(
    "weave:///wandb-smle/weave-cookboook-demo/object/RagModel:cqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo"
).get()
# MAGIC: chat_model を入れ替えて新しいバージョンを公開（他の RAG コンポーネントは気にしなくて OK）
RagModel.chat_model = new_chat_model
# まず新しいバージョンを公開して、予測時にこの参照が使われるようにする
PUB_REFERENCE = weave.publish(RagModel, "RagModel")
await RagModel.predict("When was the first conference on climate change?")
```

## 4. 既存の Models の run に接続して新しい `weave.Evaluation` を実行
最後に、新しい `RagModel` を既存の `weave.Evaluation` で評価します。統合をできるだけ簡単にするため、以下の変更を加えます。

Models の観点:
- Registry からモデルを取得すると、そのチャットモデルのエンドツーエンドのリネージの一部となる新しい `run` オブジェクトが作成されます
- run の config に現在の eval ID を含む Trace ID を追加し、モデルチームが対応する Weave ページへのリンクをクリックできるようにします

Weave の観点:
- `ChatModel`（つまり `RagModel`）への入力として artifact / Registry のリンクを保存します
- `weave.attributes` を使って、トレースに run.id を追加のカラムとして保存します

```python
# MAGIC: 評価データセットとスコアラーを持つ evaluation を取得して、それらを使用
WEAVE_EVAL = "weave:///wandb-smle/weave-cookboook-demo/object/climate_rag_eval:ntRX6qn3Tx6w3UEVZXdhIh1BWGh7uXcQpOQnIuvnSgo"
climate_rag_eval = weave.ref(WEAVE_EVAL).get()

run = wandb.init()

with weave.attributes({"wandb-run-id": run.id}):
    # 結果と call の両方を取得して、Models に評価トレースを保存できるようにするため .call 属性を使用
    summary, call = await climate_rag_eval.evaluate.call(climate_rag_eval, ` RagModel `)
```

## 5. 新しい RAG モデルを Registry に保存
新しい RAG モデルを効果的に共有するために、weave のバージョンをエイリアス として付けて、reference artifact として Registry にプッシュします。

```python
MODELS_OBJECT_VERSION = PUB_REFERENCE.digest  # weave のオブジェクト バージョン
MODELS_OBJECT_NAME = PUB_REFERENCE.name  # weave のオブジェクト名

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

    # Registry にリンク
    run.link_artifact(
        artifact_model, target_path="wandb32/wandb-registry-RAG Models/RAG Model"
    )
```
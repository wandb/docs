---
title: モデルを評価する
menu:
  default:
    identifier: ja-guides-models-evaluate-models
    parent: w-b-models
---

## Weave でモデルを評価する

[W&B Weave](https://weave-docs.wandb.ai/) は、LLM と GenAI アプリケーション の評価に特化した ツールキット です。スコアラー、ジャッジ、詳細な トレース などの包括的な評価機能を備え、モデル の パフォーマンス を理解・改善するのに役立ちます。Weave は W&B Models と連携し、モデルレジストリ に保存された モデル を評価できます。

{{< img src="/images/weave/evals.png" alt="Weave の評価ダッシュボード。モデルのパフォーマンス メトリクスとトレースを表示" >}}

### モデルの評価 の主な機能

* スコアラー と ジャッジ: 正確性、関連性、一貫性 などの 事前構築済み・カスタムの 評価メトリクス
* 評価用 データセット: 系統的な評価のための、正解 付きの 構造化 テストセット
* モデル の バージョン管理: 異なる バージョン を追跡・比較
* 詳細な トレース: 入出力 の完全な トレース で モデル の 振る舞い をデバッグ
* コスト トラッキング: 評価 全体での API コスト と トークン使用量 を監視

### はじめに: W&B Registry から モデル を評価する

W&B Models Registry から モデル をダウンロードし、Weave で評価します:

```python
import weave
import wandb
from typing import Any

# Weave を初期化
weave.init("your-entity/your-project")

# W&B Models Registry から読み込む ChatModel を定義
class ChatModel(weave.Model):
    model_name: str
    
    def model_post_init(self, __context):
        # W&B Models Registry からモデルをダウンロード
        run = wandb.init(project="your-project", job_type="model_download")
        artifact = run.use_artifact(self.model_name)
        self.model_path = artifact.download()
        # ここであなたのモデルを初期化
    
    @weave.op()
    async def predict(self, query: str) -> str:
        # モデル の 推論 ロジック
        return self.model.generate(query)

# 評価用 データセット を作成
dataset = weave.Dataset(name="eval_dataset", rows=[
    {"input": "What is the capital of France?", "expected": "Paris"},
    {"input": "What is 2+2?", "expected": "4"},
])

# スコアラー を定義
@weave.op()
def exact_match_scorer(expected: str, output: str) -> dict:
    return {"correct": expected.lower() == output.lower()}

# 評価 を実行
model = ChatModel(model_name="wandb-entity/registry-name/model:version")
evaluation = weave.Evaluation(
    dataset=dataset,
    scorers=[exact_match_scorer]
)
results = await evaluation.evaluate(model)
```

### Weave の評価を W&B Models と連携

[Models と Weave の インテグレーション デモ](https://weave-docs.wandb.ai/reference/gen_notebooks/Models_and_Weave_Integration_Demo) では、以下の完全な ワークフロー を紹介しています:

1. 既存の Registry から モデル を読み込む: W&B Models Registry に保存された ファインチューン 済み モデル をダウンロード
2. 評価 パイプライン を作成: カスタム スコアラー を使って 包括的な 評価 を構築
3. 結果 を W&B に ログ: 評価メトリクス を モデル の run に接続
4. 評価 済み モデル を バージョン管理: 改善した モデル を Registry に保存

評価 結果 を Weave と W&B Models の両方に ログします:

```python
# W&B のトラッキング付きで 評価 を実行
with weave.attributes({"wandb-run-id": wandb.run.id}):
    summary, call = await evaluation.evaluate.call(evaluation, model)

# メトリクス を W&B Models に ログ
wandb.run.log(summary)
wandb.run.config.update({
    "weave_eval_url": f"https://wandb.ai/{entity}/{project}/r/call/{call.id}"
})
```

### Weave の高度な機能

#### カスタム スコアラー と ジャッジ
ユースケース に合わせた 高度な 評価メトリクス を作成:

```python
@weave.op()
def llm_judge_scorer(expected: str, output: str, judge_model) -> dict:
    prompt = f"Is this answer correct? Expected: {expected}, Got: {output}"
    judgment = await judge_model.predict(prompt)
    return {"judge_score": judgment}
```

#### バッチ 評価
複数の モデル バージョン や 設定 を評価:

```python
models = [
    ChatModel(model_name="model:v1"),
    ChatModel(model_name="model:v2"),
]

for model in models:
    results = await evaluation.evaluate(model)
    print(f"{model.model_name}: {results}")
```

### 次のステップ

* [Weave 評価チュートリアル（完全版）](https://weave-docs.wandb.ai/tutorial-eval/)
* [Models と Weave の インテグレーション 例](https://weave-docs.wandb.ai/reference/gen_notebooks/Models_and_Weave_Integration_Demo)



## W&B Tables でモデルを評価する

W&B Tables を使ってできること:
* モデル の 予測 を比較: 異なる モデル が同じ テストセット で どうパフォーマンス するかを 横並び で表示
* 予測 の変化を追跡: トレーニング の エポック や モデル バージョン をまたいだ 予測 の推移を監視
* エラー を分析: フィルター や クエリ で、よく誤分類する 例 や エラー パターン を発見
* リッチ メディア を可視化: 画像、音声、テキスト などを 予測 や メトリクス と並べて表示

![モデルの出力と正解ラベルを並べて表示する予測テーブルの例](/images/data_vis/tables_sample_predictions.png)

### 基本例: 評価結果をログする

```python
import wandb

# run を初期化
run = wandb.init(project="model-evaluation")

# 評価 結果 の テーブル を作成
columns = ["id", "input", "ground_truth", "prediction", "confidence", "correct"]
eval_table = wandb.Table(columns=columns)

# 評価 データ を追加
for idx, (input_data, label) in enumerate(test_dataset):
    prediction = model(input_data)
    confidence = prediction.max()
    predicted_class = prediction.argmax()
    
    eval_table.add_data(
        idx,
        wandb.Image(input_data),  # 画像 などの メディア をログ
        label,
        predicted_class,
        confidence,
        label == predicted_class
    )

# テーブル を ログ
run.log({"evaluation_results": eval_table})
```

### 高度な テーブル ワークフロー

#### 複数の モデル を比較
異なる モデル の 評価 テーブル を 同じ キー に ログ して 直接 比較:

```python
# モデル A の評価
with wandb.init(project="model-comparison", name="model_a") as run:
    eval_table_a = create_eval_table(model_a, test_data)
    run.log({"test_predictions": eval_table_a})

# モデル B の評価
with wandb.init(project="model-comparison", name="model_b") as run:
    eval_table_b = create_eval_table(model_b, test_data)
    run.log({"test_predictions": eval_table_b})
```

![トレーニング エポック をまたぐ モデル 予測 の横並び比較](/images/data_vis/table_comparison.png)

#### 時間とともに 予測 を追跡
異なる トレーニング エポック で テーブル をログして 改善 を可視化:

```python
for epoch in range(num_epochs):
    train_model(model, train_data)
    
    # この エポック の 予測 を評価・ログ
    eval_table = wandb.Table(columns=["image", "truth", "prediction"])
    for image, label in test_subset:
        pred = model(image)
        eval_table.add_data(wandb.Image(image), label, pred.argmax())
    
    wandb.log({f"predictions_epoch_{epoch}": eval_table})
```

### W&B UI での インタラクティブな 分析

ログ したら、次が可能です:
1. 結果 をフィルター: 列 ヘッダー をクリックして、予測 精度、信頼度 のしきい値、特定の クラス で絞り込み
2. テーブル を比較: 複数の テーブル バージョン を選択して 横並び 比較
3. データ をクエリ: クエリ バー を使ってパターンを検索（例: `"correct" = false AND "confidence" > 0.8`）
4. グループ化 と 集計: 予測 クラス ごとにグループ化して、クラス別の 精度 メトリクス を確認

![W&B Tables における評価結果のフィルタリングとクエリ操作](/images/data_vis/wandb_demo_filter_on_a_table.png)

### 例: 拡張 テーブル による エラー 分析

```python
# 後から 分析 用の 列 を追加できる ミュータブル な テーブル を作成
eval_table = wandb.Table(
    columns=["id", "image", "label", "prediction"],
    log_mode="MUTABLE"  # 後から 列 を追加可能
)

# 最初の 予測
for idx, (img, label) in enumerate(test_data):
    pred = model(img)
    eval_table.add_data(idx, wandb.Image(img), label, pred.argmax())

run.log({"eval_analysis": eval_table})

# エラー 分析 用に 確信度 スコア を追加
confidences = [model(img).max() for img, _ in test_data]
eval_table.add_column("confidence", confidences)

# エラー タイプ を追加
error_types = classify_errors(eval_table.get_column("label"), 
                            eval_table.get_column("prediction"))
eval_table.add_column("error_type", error_types)

run.log({"eval_analysis": eval_table})
```
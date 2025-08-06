---
title: 利用可能な Models
description: W&B Inference で利用可能な foundation models を閲覧する
weight: 10
---

W&B Inference では、複数のオープンソースファウンデーションモデルへのアクセスが可能です。それぞれのモデルには異なる強みとユースケースがあります。

## モデル比較

| Model | モデル ID（API 利用時） | タイプ | コンテキストウィンドウ | パラメータ数 | 説明 |
|-------|--------------------------|------|----------------|------------|-------------|
| OpenAI GPT OSS 120B	| `openai/gpt-oss-120b` | テキスト | 131,000 | 5.1B-117B（アクティブ-合計） | 高度な推論・エージェント的・汎用ユースケース向けに設計された効率的な Mixture-of-Experts 型モデルです。 |
| OpenAI GPT OSS 20B | `openai/gpt-oss-20b` | テキスト | 131,000 | 3.6B-20B（アクティブ-合計） | OpenAI の Harmony レスポンスフォーマットで訓練された、推論を得意とする低レイテンシな Mixture-of-Experts 型モデル。 |
| Qwen3 235B A22B Thinking-2507 | `Qwen/Qwen3-235B-A22B-Thinking-2507` | テキスト | 262K | 22B-235B（アクティブ-合計） | 構造的な推論・数学・長文生成に最適化された高性能 Mixture-of-Experts モデル。|
| Qwen3 235B A22B-2507 | `Qwen/Qwen3-235B-A22B-Instruct-2507` | テキスト | 262K | 22B-235B（アクティブ-合計） | 効率的な多言語対応、Mixture-of-Experts 構造、インストラクションチューニング済みで論理的推論に最適化。|
| Qwen3 Coder 480B A35B | `Qwen/Qwen3-Coder-480B-A35B-Instruct` | テキスト | 262K | 35B-480B（アクティブ-合計） | 関数呼び出しやツール利用、長文コンテキストでの推論など、コーディングタスクに特化した Mixture-of-Experts モデル。|
| MoonshotAI Kimi K2 | `moonshotai/Kimi-K2-Instruct` | テキスト | 128K | 32B-1T（アクティブ-合計） | 複雑なツール利用・推論・コード生成に特化した Mixture-of-Experts モデル。|
| DeepSeek R1-0528 | `deepseek-ai/DeepSeek-R1-0528` | テキスト | 161K | 37B-680B（アクティブ-合計） | 複雑なコーディング、数学、構造化ドキュメント分析など正確な推論タスクに最適化。|
| DeepSeek V3-0324 | `deepseek-ai/DeepSeek-V3-0324` | テキスト | 161K | 37B-680B（アクティブ-合計） | 高度な言語プロセッシングや幅広いドキュメント分析に適した堅牢な Mixture-of-Experts 型モデル。|
| Meta Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | テキスト | 128K | 8B（合計） | 応答性の高い多言語チャットボット対話に最適化された効率的な会話モデル。|
| Meta Llama 3.3 70B | `meta-llama/Llama-3.3-70B-Instruct` | テキスト | 128K | 70B（合計） | 会話タスク・詳細な指示実行・コーディングに優れた多言語モデル。|
| Meta Llama 4 Scout | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | テキスト・ビジョン | 64K | 17B-109B（アクティブ-合計） | テキストと画像の理解を統合したマルチモーダルモデル。視覚タスクや組み合わせた分析に最適。|
| Microsoft Phi 4 Mini 3.8B | `microsoft/Phi-4-mini-instruct` | テキスト | 128K | 3.8B（アクティブ-合計） | 資源制約環境での高速応答に適したコンパクトで効率的なモデル。|

## モデル ID の利用

API を利用する際は、上記テーブルのモデル ID を指定してください。例えば:

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[...]
)
```

## 次のステップ

- 各モデルの[利用制限と価格]({{< relref "usage-limits" >}})を確認する
- [API リファレンス]({{< relref "api-reference" >}})でモデルの使い方を参照する
- [W&B Playground]({{< relref "ui-guide" >}}) でモデルを試してみる
---
title: 利用可能な Models
description: W&B Inference で利用可能な foundation models を閲覧する
menu:
  default:
    identifier: ja-guides-inference-models
weight: 10
---

W&B Inference では、複数のオープンソース基盤モデルへの アクセス を提供しています。各モデルには強みやユースケースが異なります。

## モデル比較

| モデル | API利用時のモデルID | タイプ | コンテキストウィンドウ | パラメータ | 説明 |
|-------|--------------------------|------|----------------|------------|-------------|
| OpenAI GPT OSS 120B	| `openai/gpt-oss-120b` | テキスト | 131,000 | 5.1B-117B（アクティブ-総数） | 論理的推論、高度なエージェントや汎用ユースケース向けに設計された効率的な Mixture-of-Experts モデル。 |
| OpenAI GPT OSS 20B | `openai/gpt-oss-20b` | テキスト | 131,000 | 3.6B-20B（アクティブ-総数） | 低レイテンシの Mixture-of-Experts モデルで、OpenAI の Harmony 応答形式で学習し推論機能を備えています。 |
| Qwen3 235B A22B Thinking-2507 | `Qwen/Qwen3-235B-A22B-Thinking-2507` | テキスト | 262K | 22B-235B（アクティブ-総数） | 構造化推論、数学、長文生成に最適化された高性能 Mixture-of-Experts モデル |
| Qwen3 235B A22B-2507 | `Qwen/Qwen3-235B-A22B-Instruct-2507` | テキスト | 262K | 22B-235B（アクティブ-総数） | 効率的な多言語対応 Mixture-of-Experts 指示特化型モデルで論理的推論に最適化 |
| Qwen3 Coder 480B A35B | `Qwen/Qwen3-Coder-480B-A35B-Instruct` | テキスト | 262K | 35B-480B（アクティブ-総数） | コーディング（関数呼び出し、ツール利用、長文推論など）に最適化された Mixture-of-Experts モデル |
| MoonshotAI Kimi K2 | `moonshotai/Kimi-K2-Instruct` | テキスト | 128K | 32B-1T（アクティブ-総数） | 複雑なツール利用、推論、 コード合成 に最適化された Mixture-of-Experts モデル |
| DeepSeek R1-0528 | `deepseek-ai/DeepSeek-R1-0528` | テキスト | 161K | 37B-680B（アクティブ-総数） | コーディング、数学、構造化ドキュメント分析などの精密な推論タスクに最適化 |
| DeepSeek V3-0324 | `deepseek-ai/DeepSeek-V3-0324` | テキスト | 161K | 37B-680B（アクティブ-総数） | 高度な言語 プロセッシング と包括的なドキュメント 分析 に対応した堅牢な Mixture-of-Experts モデル |
| Meta Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | テキスト | 128K | 8B（総数） | 応答性の高い多言語 チャットボット 対話用に最適化された効率的な会話モデル |
| Meta Llama 3.3 70B | `meta-llama/Llama-3.3-70B-Instruct` | テキスト | 128K | 70B（総数） | 会話、詳細な指示フォロー、コーディングに優れた多言語モデル |
| Meta Llama 4 Scout | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | テキスト、ビジョン | 64K | 17B-109B（アクティブ-総数） | テキストと画像理解を統合したマルチモーダルモデルでビジュアルタスクや複合 分析 に最適 |
| Microsoft Phi 4 Mini 3.8B | `microsoft/Phi-4-mini-instruct` | テキスト | 128K | 3.8B（アクティブ-総数） | リソース制約のある環境で高速応答に適したコンパクトで効率的なモデル |

## モデルIDの使い方

API 利用時は、上記テーブル内のモデルIDを指定してください。例：

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[...]
)
```

## 次のステップ

- 各モデルの[使用制限と料金]({{< relref path="usage-limits" lang="ja" >}})を確認する
- これらのモデルの使い方は[APIリファレンス]({{< relref path="api-reference" lang="ja" >}})を見る
- [W&B Playground]({{< relref path="ui-guide" lang="ja" >}})でモデルを試してみる
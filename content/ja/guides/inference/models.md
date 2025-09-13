---
title: 利用可能な モデル
description: W&B Inference で利用可能な基盤モデルを閲覧する
menu:
  default:
    identifier: ja-guides-inference-models
weight: 10
---

W&B Inference は、複数のオープンソースの基盤 モデル への アクセス を提供します。各 モデル には異なる強みと ユースケース があります。

## モデルカタログ

| モデル | モデル ID（API 利用時） | タイプ | コンテキスト ウィンドウ | パラメータ数 | 説明 |
|-------|--------------------------|------|----------------|------------|-------------|
| DeepSeek R1-0528 | `deepseek-ai/DeepSeek-R1-0528` | テキスト | 161K | 37B-680B (Active-Total) | 複雑なコーディング、数学、構造化ドキュメントの 分析 など、精緻な推論タスク向けに最適化 |
| DeepSeek V3-0324 | `deepseek-ai/DeepSeek-V3-0324` | テキスト | 161K | 37B-680B (Active-Total) | 高複雑度の言語プロセッシングと包括的なドキュメント分析に特化した堅牢な Mixture-of-Experts モデル |
| DeepSeek V3.1 | `deepseek-ai/DeepSeek-V3.1` | テキスト | 128K | 37B-671B (Active-Total) | プロンプト テンプレートにより思考/非思考の両モードをサポートする大規模ハイブリッド モデル |
| Meta Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | テキスト | 128K | 8B (Total) | 応答性の高い多言語 チャットボット との対話に最適化された高効率な会話 モデル |
| Meta Llama 3.3 70B | `meta-llama/Llama-3.3-70B-Instruct` | テキスト | 128K | 70B (Total) | 会話タスク、詳細な指示追従、コーディングに優れた多言語 モデル |
| Meta Llama 4 Scout | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | テキスト、ビジョン | 64K | 17B-109B (Active-Total) | テキストと画像理解を統合したマルチモーダル モデルで、視覚タスクや統合的な 分析 に最適 |
| Microsoft Phi 4 Mini 3.8B | `microsoft/Phi-4-mini-instruct` | テキスト | 128K | 3.8B (Active-Total) | リソース制約の厳しい 環境 での高速応答に最適な、コンパクトで高効率な モデル |
| MoonshotAI Kimi K2 | `moonshotai/Kimi-K2-Instruct` | テキスト | 128K | 32B-1T (Active-Total) | 複雑な ツール 利用、推論、コード合成に最適化された Mixture-of-Experts モデル |
| OpenAI GPT OSS 20B | `openai/gpt-oss-20b` | テキスト | 131K | 3.6B-20B (Active-Total) | OpenAI の Harmony レスポンス形式で学習された、推論能力を備える低レイテンシな Mixture-of-Experts モデル。 |
| OpenAI GPT OSS 120B	| `openai/gpt-oss-120b` | テキスト | 131K | 5.1B-117B (Active-Total) | 高度な推論、エージェント的および汎用的な ユースケース を想定して設計された、高効率な Mixture-of-Experts モデル。 |
| Qwen3 235B A22B Thinking-2507 | `Qwen/Qwen3-235B-A22B-Thinking-2507` | テキスト | 262K | 22B-235B (Active-Total) | 構造化推論、数学、長文生成に最適化された高性能な Mixture-of-Experts モデル |
| Qwen3 235B A22B-2507 | `Qwen/Qwen3-235B-A22B-Instruct-2507` | テキスト | 262K | 22B-235B (Active-Total) | 論理的推論に最適化された、多言語対応・Mixture-of-Experts・指示チューニング済みの高効率 モデル |
| Qwen3 Coder 480B A35B | `Qwen/Qwen3-Coder-480B-A35B-Instruct` | テキスト | 262K | 35B-480B (Active-Total) | 関数呼び出し、ツールの利用、長いコンテキストでの推論などのコーディング タスクに最適化された Mixture-of-Experts モデル |
| Z.AI GLM 4.5 | `zai-org/GLM-4.5` | テキスト | 131K | 32B-355B (Active-Total) | 推論、コード、エージェントのために、ユーザーが制御可能な思考/非思考 モードを備えた Mixture-of-Experts モデル |

## モデル ID の使用

API を使用する際は、上の表にある ID で モデル を指定します。例:

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[...]
)
```

## 次のステップ

- 各 モデル の [使用制限と料金]({{< relref path="usage-limits" lang="ja" >}}) を確認する
- これらの モデル の使い方は [API リファレンス]({{< relref path="api-reference" lang="ja" >}}) を参照
- [W&B Playground]({{< relref path="ui-guide" lang="ja" >}}) で モデル を試す
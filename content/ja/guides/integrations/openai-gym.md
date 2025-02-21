---
title: OpenAI Gym
description: W&B と OpenAI Gym の統合方法
menu:
  default:
    identifier: ja-guides-integrations-openai-gym
    parent: integrations
weight: 260
---

{{% alert %}}
「2021 年から Gym をメンテナンスしてきたチームは、今後のすべての開発を [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) に移行しました。Gymnasium は Gym のドロップイン代替品 (import gymnasium as gym) であり、Gym は今後更新されません。」( [出典](https://github.com/openai/gym#the-team-that-has-been-maintaining-gym-since-2021-has-moved-all-future-development-to-gymnasium-a-drop-in-replacement-for-gym-import-gymnasium-as-gym-and-gym-will-not-be-receiving-any-future-updates-please-switch-over-to-gymnasium-as-soon-as-youre-able-to-do-so-if-youd-like-to-read-more-about-the-story-behind-this-switch-please-check-out-this-blog-post) )

Gym はもはや活発にメンテナンスされている プロジェクト ではないため、Gymnasium との インテグレーション をお試しください。
{{% /alert %}}

[OpenAI Gym](https://github.com/openai/gym) を使用している場合、Weights & Biases は `gym.wrappers.Monitor` によって生成された 環境 のビデオを自動的に ログ 記録します。`monitor_gym` キーワード 引数 を [`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) に `True` に設定するか、`wandb.gym.monitor()` を呼び出すだけです。

Weights & Biases の gym インテグレーション は非常に軽量です。`gym` から ログ 記録されている [ビデオ ファイルの名前を確認](https://github.com/wandb/wandb/blob/master/wandb/integration/gym/__init__.py#L15) し、その名前に従って名前を付けるか、一致するものが見つからない場合は `"videos"` にフォールバックします。より詳細な制御が必要な場合は、いつでも手動で [ビデオを ログ 記録]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}}) できます。

[CleanRL](https://github.com/vwxyzjn/cleanrl) による [OpenRL Benchmark](http://wandb.me/openrl-benchmark-report) では、この インテグレーション が OpenAI Gym の例で使用されています。ソース コード ( [特定の Runs に使用される特定の コード](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang) を含む) は、gym の使用方法を示しています。

{{< img src="/images/integrations/open_ai_report_example.png" alt="詳細はこちら: http://wandb.me/openrl-benchmark-report" >}}

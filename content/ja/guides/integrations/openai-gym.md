---
title: OpenAI Gym
description: W&B を OpenAI Gym と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-openai-gym
    parent: integrations
weight: 260
---

{{% alert %}}
2021年以降 Gym をメンテナンスしてきたチームは、今後の開発をすべて [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) に移行しました。Gymnasium は Gym のドロップイン代替品 (import gymnasium as gym) であり、Gym は今後アップデートを受け取ることはありません。([出典](https://github.com/openai/gym#the-team-that-has-been-maintaining-gym-since-2021-has-moved-all-future-development-to-gymnasium-a-drop-in-replacement-for-gym-import-gymnasium-as-gym-and-gym-will-not-be-receiving-any-future-updates-please-switch-over-to-gymnasium-as-soon-as-youre-able-to-do-so-if-youd-like-to-read-more-about-the-story-behind-this-switch-please-check-out-this-blog-post))

Gym はもはや活発にメンテナンスされているプロジェクトではないため、Gymnasium との インテグレーション をお試しください。
{{% /alert %}}

[OpenAI Gym](https://github.com/openai/gym) を使用している場合、Weights & Biases は `gym.wrappers.Monitor` によって生成された 環境 の動画を自動的に ログ します。[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) の `monitor_gym` キーワード 引数 を `True` に設定するか、`wandb.gym.monitor()` を呼び出すだけです。

当社の gym インテグレーション は非常に軽量です。`gym` から ログ されている動画ファイルの名前を [確認](https://github.com/wandb/wandb/blob/master/wandb/integration/gym/__init__.py#L15) し、それにちなんで名前を付けるか、一致するものが見つからない場合は `"videos"` にフォールバックします。より詳細な制御が必要な場合は、いつでも手動で [動画を ログ ]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}}) できます。

[CleanRL](https://github.com/vwxyzjn/cleanrl) による [OpenRL Benchmark](http://wandb.me/openrl-benchmark-report) では、OpenAI Gym の例でこの インテグレーション を使用しています。gym での使用方法を示すソース コード ([特定の Runs に使用される特定のコード](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang) を含む) を見つけることができます。

{{< img src="/images/integrations/open_ai_report_example.png" alt="詳細はこちら: http://wandb.me/openrl-benchmark-report" >}}

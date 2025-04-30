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
「2021 年から Gym を維持しているチームは、将来のすべての開発を [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) に移しました。Gym の代替である Gymnasium (import gymnasium as gym) を利用できるようにするため、Gym は今後更新を受けることはありません。」（[出典](https://github.com/openai/gym#the-team-that-has-been-maintaining-gym-since-2021-has-moved-all-future-development-to-gymnasium-a-drop-in-replacement-for-gym-import-gymnasium-as-gym-and-gym-will-not-be-receiving-any-future-updates-please-switch-over-to-gymnasium-as-soon-as-youre-able-to-do-so-if-youd-like-to-read-more-about-the-story-behind-this-switch-please-check-out-this-blog-post))

Gym はもはや積極的に維持されているプロジェクトではないため、Gymnasium とのインテグレーションを試してみてください。
{{% /alert %}}

もし [OpenAI Gym](https://github.com/openai/gym) を使用している場合、Weights & Biases は自動的に `gym.wrappers.Monitor` によって生成された環境のビデオをログします。ただし、[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) の `monitor_gym` キーワード引数を `True` に設定するか、`wandb.gym.monitor()` を呼び出してください。

私たちの gym インテグレーションは非常に軽量です。単に `gym` からログされるビデオファイルの[名前を見て](https://github.com/wandb/wandb/blob/master/wandb/integration/gym/__init__.py#L15)、それにちなんで名前を付けるか、一致しない場合は「videos」にフォールバックします。より細かい制御をしたい場合は、いつでも手動で[ビデオをログする]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}})ことができます。

[OpenRL ベンチマーク](http://wandb.me/openrl-benchmark-report) は、[CleanRL](https://github.com/vwxyzjn/cleanrl) によって、OpenAI Gym の例でこのインテグレーションを使用しています。gym を使用する方法を示すソースコード（[特定の run に使用された特定のコード](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang)を含む）を見つけることができます。

{{< img src="/images/integrations/open_ai_report_example.png" alt="詳細はこちら: http://wandb.me/openrl-benchmark-report" >}}
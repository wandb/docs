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
"2021 年以来 Gym を維持してきたチームは、今後の開発をすべて [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) に移行しました。これは Gym の代替となるもので (import gymnasium as gym)、Gym は今後の更新を受けません。" ([出典](https://github.com/openai/gym#the-team-that-has-been-maintaining-gym-since-2021-has-moved-all-future-development-to-gymnasium-a-drop-in-replacement-for-gym-import-gymnasium-as-gym-and-gym-will-not-be-receiving-any-future-updates-please-switch-over-to-gymnasium-as-soon-as-youre-able-to-do-so-if-youd-like-to-read-more-about-the-story-behind-this-switch-please-check-out-this-blog-post))

Gym はもはや積極的にメンテナンスされているプロジェクトではないため、私たちの Gymnasium とのインテグレーションを試してみてください。
{{% /alert %}}

[OpenAI Gym](https://github.com/openai/gym) を使用している場合、Weights & Biases は `gym.wrappers.Monitor` によって生成された環境のビデオを自動的にログします。`monitor_gym` キーワード引数を [`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) に `True` として設定するか、`wandb.gym.monitor()` を呼び出してください。

私たちの gym インテグレーションは非常に軽量です。ただ単に [ログされているビデオファイルの名前を確認して](https://github.com/wandb/wandb/blob/master/wandb/integration/gym/__init__.py#L15)、それにちなんで名前を付けるか、一致するものがない場合は "videos" にフォールバックします。より多くの制御を行いたい場合は、いつでも手動で[ビデオをログ]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}})することができます。

[CleanRL](https://github.com/vwxyzjn/cleanrl) の [OpenRL Benchmark](http://wandb.me/openrl-benchmark-report) は、OpenAI Gym の例にこのインテグレーションを使用しています。ジムを使った方法を示すソースコード（[特定の runs に使用された特定のコード](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang) を含む）を見つけることができます。

{{< img src="/images/integrations/open_ai_report_example.png" alt="詳細はこちら: http://wandb.me/openrl-benchmark-report" >}}
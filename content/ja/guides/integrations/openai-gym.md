---
title: OpenAI Gym
description: W&B を OpenAI Gym と統合する方法
menu:
  default:
    identifier: openai-gym
    parent: integrations
weight: 260
---

{{% alert %}}
「2021年以降 Gym をメンテナンスしてきたチームは、今後の開発を [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) に移行しました（`import gymnasium as gym` のようにドロップインで置き換え可能）。Gym は今後アップデートされません。」（[出典](https://github.com/openai/gym#the-team-that-has-been-maintaining-gym-since-2021-has-moved-all-future-development-to-gymnasium-a-drop-in-replacement-for-gym-import-gymnasium-as-gym-and-gym-will-not-be-receiving-any-future-updates-please-switch-over-to-gymnasium-as-soon-as-youre-able-to-do-so-if-youd-like-to-read-more-about-the-story-behind-this-switch-please-check-out-this-blog-post)）

Gym は現在アクティブにメンテナンスされていないプロジェクトのため、ぜひ Gymnasium とのインテグレーションをお試しください。
{{% /alert %}}

[OpenAI Gym](https://github.com/openai/gym) を使用している場合、W&B は `gym.wrappers.Monitor` によって生成された環境の動画を自動でログします。[`wandb.init`]({{< relref "/ref/python/sdk/functions/init.md" >}}) で `monitor_gym` キーワード引数を `True` に設定するか、`wandb.gym.monitor()` を呼び出してください。

gym とのインテグレーションは非常にシンプルです。[動画ファイルの名前を確認](https://github.com/wandb/wandb/blob/master/wandb/integration/gym/__init__.py#L15)し、その名前に合わせてログしたり、該当するものがなければ `"videos"` という名前で保存します。より細かくコントロールしたい場合は、[動画を手動でログ]({{< relref "/guides/models/track/log/media.md" >}})することも可能です。

[OpenRL Benchmark](https://wandb.me/openrl-benchmark-report)（[CleanRL](https://github.com/vwxyzjn/cleanrl) 提供）では、OpenAI Gym のサンプルでこのインテグレーションが使われています。gym を使った方法を実演しているソースコード（[特定 run 用のコード例](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang) を含む）も公開されています。

{{< img src="/images/integrations/open_ai_report_example.png" alt="OpenAI Gym ダッシュボード" >}}
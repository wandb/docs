---
title: OpenAI Gym
description: W&B を OpenAI Gym と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-openai-gym
    parent: integrations
weight: 260
---

{{% alert %}}
「2021 年以降 Gym をメンテナンスしてきたチームは、今後の開発をすべて [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) に移行しました（import gymnasium as gym で Gym の代替となります）。Gym には今後アップデートが行われません。」([出典](https://github.com/openai/gym#the-team-that-has-been-maintaining-gym-since-2021-has-moved-all-future-development-to-gymnasium-a-drop-in-replacement-for-gym-import-gymnasium-as-gym-and-gym-will-not-be-receiving-any-future-updates-please-switch-over-to-gymnasium-as-soon-as-youre-able-to-do-so-if-youd-like-to-read-more-about-the-story-behind-this-switch-please-check-out-this-blog-post))

Gym は現在メンテナンスされていないプロジェクトのため、ぜひ Gymnasium とのインテグレーションをお試しください。
{{% /alert %}}

[OpenAI Gym](https://github.com/openai/gym) を使っている場合、W&B は `gym.wrappers.Monitor` で生成された環境の動画を自動的にログします。`monitor_gym` キーワード引数を [`wandb.init`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) に `True` で指定するか、`wandb.gym.monitor()` を呼び出すだけです。

私たちの gym インテグレーションは非常にシンプルです。[動画ファイルの名前を確認](https://github.com/wandb/wandb/blob/master/wandb/integration/gym/__init__.py#L15) し、`gym` からログされた場合はその名前を使い、一致しなければ `"videos"` をデフォルトとしています。より細かい制御が必要な場合は、手動で[動画をログする]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}})こともできます。

[OpenRL Benchmark](https://wandb.me/openrl-benchmark-report) では、[CleanRL](https://github.com/vwxyzjn/cleanrl) により、このインテグレーションが OpenAI Gym の例で使われています。使い方を示すソースコード（[特定の run で使われているコード](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang)を含む）も参照できます。

{{< img src="/images/integrations/open_ai_report_example.png" alt="OpenAI Gym ダッシュボード" >}}
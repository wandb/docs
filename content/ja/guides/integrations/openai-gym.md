---
title: OpenAI Gym
description: W&B と OpenAI Gym を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-openai-gym
    parent: integrations
weight: 260
---

{{% alert %}}
「2021 年以降 Gym をメンテナンスしてきたチームは、今後の開発をすべて [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)（Gym の代替。import gymnasium as gym）に移行し、Gym は今後アップデートを受け取りません。」（[出典](https://github.com/openai/gym#the-team-that-has-been-maintaining-gym-since-2021-has-moved-all-future-development-to-gymnasium-a-drop-in-replacement-for-gym-import-gymnasium-as-gym-and-gym-will-not-be-receiving-any-future-updates-please-switch-over-to-gymnasium-as-soon-as-youre-able-to-do-so-if-youd-like-to-read-more-about-the-story-behind-this-switch-please-check-out-this-blog-post)）

Gym はすでに積極的にメンテナンスされていないため、Gymnasium とのインテグレーションをお試しください。
{{% /alert %}}

[OpenAI Gym](https://github.com/openai/gym) を使用している場合、W&B は `gym.wrappers.Monitor` によって生成される 環境 の動画を自動で ログ します。`monitor_gym` キーワード引数を [`wandb.init`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) に `True` として渡すか、`wandb.gym.monitor()` を呼び出してください。

この gym インテグレーションはとても軽量です。`gym` から ログ される動画ファイル名を参照してその名前を付け、該当がない場合は `"videos"` をデフォルトとして使用します。より細かく制御したい場合は、手動で [動画を ログ する]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}}) こともできます。

[CleanRL](https://github.com/vwxyzjn/cleanrl) による [OpenRL Benchmark](https://wandb.me/openrl-benchmark-report) は、OpenAI Gym のサンプルにこのインテグレーションを使用しています。gym の使い方を示すソースコード（[特定の Runs で使用された具体的なコード](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang) を含む）を参照できます。

{{< img src="/images/integrations/open_ai_report_example.png" alt="OpenAI Gym ダッシュボード" >}}
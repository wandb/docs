---
title: '`Est. Runs` カラムとは何ですか？'
menu:
  support:
    identifier: ja-support-kb-articles-est_runs_column
support:
- sweep
- ハイパーパラメーター
toc_hide: true
type: docs
url: /ja/support/:filename
---

W&B は、離散的な探索空間を持つ W&B Sweep を作成する際に生成される Run の推定数を提供します。この合計値は、探索空間のデカルト積を反映しています。

たとえば、次の探索空間を考えてみましょう：

{{< img src="/images/sweeps/sweeps_faq_whatisestruns_1.png" alt="" >}}

この場合、デカルト積は 9 になります。W&B は、アプリケーション UI にこの値を推定された run の数 (**Est. Runs**) として表示します：

{{< img src="/images/sweeps/spaces_sweeps_faq_whatisestruns_2.webp" alt="" >}}

プログラムから推定 Run 数を取得するには、W&B SDK 内の Sweep オブジェクトの `expected_run_count` 属性を使用してください。

```python
sweep_id = wandb.sweep(
    sweep_configs, project="your_project_name", entity="your_entity_name"
)
api = wandb.Api()
sweep = api.sweep(f"your_entity_name/your_project_name/sweeps/{sweep_id}")
print(f"EXPECTED RUN COUNT = {sweep.expected_run_count}")
```
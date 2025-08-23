---
title: '`Est. Runs` 列とは何ですか？'
menu:
  support:
    identifier: ja-support-kb-articles-est_runs_column
support:
- スイープ
- ハイパーパラメーター
toc_hide: true
type: docs
url: /support/:filename
---

W&B では、離散的な探索空間を持つ W&B Sweep を作成する際に、生成される Run の推定数を表示します。この合計は探索空間のデカルト積を反映しています。

例えば、次のような探索空間を考えてみましょう。

{{< img src="/images/sweeps/sweeps_faq_whatisestruns_1.png" alt="推定 Run 列" >}}

この場合、デカルト積は 9 となります。W&B はこの値を App UI 上で推定 Run 数（**Est. Runs**）として表示します。

{{< img src="/images/sweeps/spaces_sweeps_faq_whatisestruns_2.webp" alt="Sweep の Run 推定" >}}

推定 Run 数をプログラム上で取得したい場合は、W&B SDK の Sweep オブジェクトの `expected_run_count` 属性を利用します。

```python
sweep_id = wandb.sweep(
    sweep_configs, project="your_project_name", entity="your_entity_name"
)
api = wandb.Api()
sweep = api.sweep(f"your_entity_name/your_project_name/sweeps/{sweep_id}")
print(f"EXPECTED RUN COUNT = {sweep.expected_run_count}")
```

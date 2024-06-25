---
displayed_sidebar: default
---


# Runを管理する

### Runをチームに移動する

プロジェクトページで:

1. テーブルタブをクリックしてRunテーブルを展開します
2. チェックボックスをクリックしてすべてのRunを選択します
3. **Move**をクリックします: 移動先のプロジェクトは、個人アカウントやあなたがメンバーであるチームのいずれかにできます。

![](/images/app_ui/demo_move_runs.gif)

### 新しいRunをチームに送る

スクリプトで、entityをチームに設定します。"Entity"はユーザー名またはチーム名を意味します。Runを送信する前に、Webアプリでentity（個人アカウントまたはチームアカウント）を作成します。

```python
wandb.init(entity="example-team")
```

**デフォルトのentity**は、チームに参加すると更新されます。つまり、[設定ページ](https://app.wandb.ai/settings)では、新しいプロジェクトを作成するデフォルトの場所が、あなたが参加したチームに変更されていることがわかります。以下はその[設定ページ](https://app.wandb.ai/settings)セクションの例です:

![](/images/app_ui/send_new_runs_to_team.png)
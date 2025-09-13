---
title: 実験を再現する
menu:
  default:
    identifier: ja-guides-models-track-reproduce_experiments
    parent: track
weight: 7
---

チームメンバー が 自身の 結果 を 検証・確認するために作成した experiment を 再現します。

experiment を 再現する前に、次の内容を控えてください:

* run が ログされた project の 名前
* 再現したい run の 名前

experiment を 再現するには:

1. run が ログされた project に 移動します。
2. 左側 の サイドバー で **Workspace** タブ を 選択します。
3. run の 一覧 から、再現したい run を 選択します。
4. **Overview** を クリックします。

続行するには、指定された ハッシュ の時点における experiment の コード を ダウンロードするか、experiment の リポジトリ 全体を クローンします。

{{< tabpane text=true >}}
{{% tab "Python スクリプト または ノートブック をダウンロード" %}}

experiment の Python スクリプト または ノートブック を ダウンロードします:

1. **Command** フィールドで、experiment を 作成した スクリプト の 名前 を メモします。
2. 左 の ナビゲーションバー で **Code** タブ を 選択します。
3. スクリプト または ノートブック に 対応する ファイル の 横にある **Download** を クリックします。


{{% /tab %}}
{{% tab "GitHub" %}}

experiment を 作成する際に チームメンバー が 使った GitHub リポジトリ を クローンします。手順:

1. 必要に応じて、チームメンバー が experiment を 作成するために使用した GitHub リポジトリ への アクセス を 取得します。
2. GitHub リポジトリ の URL が 含まれる **Git repository** フィールド を コピーします。
3. リポジトリ を クローンします:
    ```bash
    git clone https://github.com/your-repo.git && cd your-repo
    ```
4. **Git state** フィールド を ターミナル に コピー＆ペーストします。Git state は、チームメンバー が experiment を 作成した際に使った まさにその コミット を チェックアウトするための Git コマンド 群 です。続く コードスニペット にある 値 を 自分の ものに 置き換えてください:
    ```bash
    git checkout -b "<run-name>" 0123456789012345678901234567890123456789
    ```



{{% /tab %}}
{{< /tabpane >}}

5. 左 の ナビゲーションバー で **Files** を 選択します。
6. `requirements.txt` ファイル を ダウンロードし、作業用 ディレクトリー に 保存します。この ディレクトリー には、クローンした GitHub リポジトリ か、ダウンロードした Python スクリプト または ノートブック の いずれか が 含まれている 必要があります。
7. （推奨）Python の 仮想 環境 を 作成します。
8. `requirements.txt` に 記載された 依存関係 を インストールします。
    ```bash
    pip install -r requirements.txt
    ```

9. コード と 依存関係 が そろったので、スクリプト または ノートブック を 実行して experiment を 再現できます。リポジトリ を クローンした 場合は、スクリプト または ノートブック が 置かれている ディレクトリー に 移動する 必要が あるかも しれません。そうでなければ、作業用 ディレクトリー から そのまま スクリプト または ノートブック を 実行できます。

{{< tabpane text=true >}}
{{% tab "Python ノートブック" %}}

Python ノートブック を ダウンロードした 場合は、ノートブック を ダウンロードした ディレクトリー に 移動し、ターミナル で 次の コマンド を 実行します:
```bash
jupyter notebook
```

{{% /tab %}}
{{% tab "Python スクリプト" %}}

Python スクリプト を ダウンロードした 場合は、スクリプト を ダウンロードした ディレクトリー に 移動し、ターミナル で 次の コマンド を 実行します。`<>` で 囲まれた 値 は 自分の ものに 置き換えてください:

```bash
python <your-script-name>.py
```


{{% /tab %}}
{{< /tabpane >}}
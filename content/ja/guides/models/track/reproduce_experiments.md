---
title: Reproduce experiments
menu:
  default:
    identifier: ja-guides-models-track-reproduce_experiments
    parent: track
weight: 7
---

チームメンバーが作成した実験を再現して、その結果を検証します。

実験を再現する前に、以下をメモする必要があります。

* run が記録された project の名前
* 再現したい run の名前

実験を再現するには:

1. run が記録されている project に移動します。
2. 左側のサイドバーで [**Workspace**] タブを選択します。
3. run のリストから、再現する run を選択します。
4. [**概要**] をクリックします。

次に、特定のハッシュで実験の コード をダウンロードするか、実験の リポジトリ 全体をクローンします。

{{< tabpane text=true >}}
{{% tab "Python スクリプトまたは notebook のダウンロード" %}}

実験の Python スクリプトまたは notebook をダウンロードします。

1. [**Command**] フィールドで、実験を作成したスクリプトの名前をメモします。
2. 左側の ナビゲーションバー で [**Code**] タブを選択します。
3. スクリプトまたは notebook に対応するファイルの横にある [**ダウンロード**] をクリックします。


{{% /tab %}}
{{% tab "GitHub" %}}

チームメイトが実験の作成に使用した GitHub リポジトリをクローンします。これを行うには:

1. 必要に応じて、チームメイトが実験の作成に使用した GitHub リポジトリへの アクセス権 を取得します。
2. [**Git リポジトリ**] フィールドをコピーします。ここには GitHub リポジトリの URL が含まれています。
3. リポジトリをクローンします。
    ```bash
    git clone https://github.com/your-repo.git && cd your-repo
    ```
4. [**Git state**] フィールドをコピーして ターミナル に貼り付けます。Git state は、チームメイトが実験の作成に使用した正確なコミットをチェックアウトする一連の Git コマンドです。上記の コードスニペット で指定された 値 を独自の値に置き換えます。
    ```bash
    git checkout -b "<run-name>" 0123456789012345678901234567890123456789
    ```



{{% /tab %}}
{{< /tabpane >}}

5. 左側の ナビゲーションバー で [**ファイル**] を選択します。
6. `requirements.txt` ファイルをダウンロードして、作業 ディレクトリー に保存します。この ディレクトリー には、クローンされた GitHub リポジトリ、またはダウンロードされた Python スクリプトまたは notebook のいずれかが含まれている必要があります。
7. （推奨）Python 仮想 環境 を作成します。
8. `requirements.txt` ファイルで指定された 要件 をインストールします。
    ```bash
    pip install -r requirements.txt
    ```

9. コード と依存関係が揃ったので、スクリプトまたは notebook を実行して実験を再現できます。リポジトリをクローンした場合は、スクリプトまたは notebook がある ディレクトリー に移動する必要がある場合があります。それ以外の場合は、作業 ディレクトリー からスクリプトまたは notebook を実行できます。

{{< tabpane text=true >}}
{{% tab "Python notebook" %}}

Python notebook をダウンロードした場合は、notebook をダウンロードした ディレクトリー に移動し、ターミナル で次の コマンド を実行します。
```bash
jupyter notebook
```

{{% /tab %}}
{{% tab "Python スクリプト" %}}

Python スクリプトをダウンロードした場合は、スクリプトをダウンロードした ディレクトリー に移動し、ターミナル で次の コマンド を実行します。`<>` で囲まれた 値 は独自の値に置き換えてください。

```bash
python <your-script-name>.py
```


{{% /tab %}}
{{< /tabpane >}}

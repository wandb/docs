---
title: 実験を再現する
menu:
  default:
    identifier: ja-guides-models-track-reproduce_experiments
    parent: track
weight: 7
---

チームメンバーが作成した実験を再現して、その結果を検証・確認しましょう。

実験を再現する前に、以下をメモしておく必要があります。

* run が記録された Project の名前
* 再現したい run の名前

実験を再現する手順:

1. run が記録された Project にアクセスします。
2. 左サイドバーの **Workspace** タブを選択します。
3. runs 一覧から、再現したい run を選択します。
4. **Overview** をクリックします。

続けて、指定されたハッシュで実験のコードをダウンロードするか、実験全体のリポジトリをクローンします。

{{< tabpane text=true >}}
{{% tab "Download Python script or notebook" %}}

実験の Python スクリプトまたはノートブックをダウンロードする手順です。

1. **Command** フィールドで、その実験を作成したスクリプト名をメモしておきます。
2. 左ナビゲーションバーの **Code** タブを選択します。
3. スクリプトまたはノートブックに該当するファイルの横にある **Download** をクリックします。

{{% /tab %}}
{{% tab "GitHub" %}}

チームメイトが実験を作成した際に利用した GitHub リポジトリをクローンします。手順は以下の通りです。

1. 必要であれば、チームメイトが利用した GitHub リポジトリへのアクセス権を取得します。
2. **Git repository** フィールド（GitHub リポジトリの URL）をコピーします。
3. リポジトリをクローンします:
    ```bash
    git clone https://github.com/your-repo.git && cd your-repo
    ```
4. **Git state** フィールドをコピーしてターミナルに貼り付けます。Git state とは、チームメイトが実験の作成に使ったまさにそのコミットをチェックアウトする Git コマンドです。後続のコードスニペットで示す値は、自身のものに置き換えてください。
    ```bash
    git checkout -b "<run-name>" 0123456789012345678901234567890123456789
    ```

{{% /tab %}}
{{< /tabpane >}}

5. 左ナビゲーションバーで **Files** を選択します。
6. `requirements.txt` ファイルをダウンロードし、作業ディレクトリーに保存します。このディレクトリーには、クローンした GitHub リポジトリまたはダウンロードした Python スクリプト／ノートブックが含まれている必要があります。
7. （推奨）Python 仮想環境を作成します。
8. `requirements.txt` ファイルで指定されている依存パッケージをインストールします。
    ```bash
    pip install -r requirements.txt
    ```

9. コードと依存関係が揃ったら、スクリプトまたはノートブックを実行して実験を再現できます。リポジトリをクローンした場合は、スクリプトやノートブックが格納されているディレクトリーに移動してください。それ以外の場合は、作業ディレクトリーからスクリプトまたはノートブックを実行できます。

{{< tabpane text=true >}}
{{% tab "Python notebook" %}}

Python ノートブックをダウンロードした場合は、ノートブックをダウンロードしたディレクトリーに移動し、ターミナルで次のコマンドを実行してください。
```bash
jupyter notebook
```

{{% /tab %}}
{{% tab "Python script" %}}

Python スクリプトをダウンロードした場合は、スクリプトをダウンロードしたディレクトリーに移動し、ターミナルで次のコマンドを実行します。`<>` で囲まれている部分にはご自身の値を入力してください。

```bash
python <your-script-name>.py
```

{{% /tab %}}
{{< /tabpane >}}
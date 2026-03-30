---
title: ライセンス通知
---

<div id="license-notice-for-reference-documentation-generation">
  # リファレンスドキュメント生成に関するライセンス通知
</div>

<div id="overview">
  ## 概要
</div>

このディレクトリ内のスクリプトは、開発/CI プロセス中にリファレンスドキュメントを生成するためだけに使用されます。Weave ライブラリと共に配布されたり、本番コードに含まれたりすることはありません。

<div id="dependencies-and-their-licenses">
  ## 依存関係とそのライセンス
</div>

<div id="direct-dependencies">
  ### 直接依存関係
</div>

* **requests** (Apache-2.0): HTTPリクエストに使用する
* **lazydocs** (MIT): W&amp;Bが管理するドキュメントジェネレーター

<div id="transitive-dependencies-via-lazydocs">
  ### 推移的依存関係 (lazydocs 経由)
</div>

* **setuptools** (MIT、LGPL-3.0 コンポーネントを含む) : ビルドシステム
* その他、混在ライセンスを持つ各種依存関係

<div id="important-notes">
  ## 重要なメモ
</div>

1. **開発専用**: これらの依存関係は、CI/GitHub Actions でのドキュメント生成中にのみ一時的にインストールされます。配布される Weave パッケージには含まれません。

2. **配布なし**: 生成されたドキュメントは、実行可能なコードや依存関係を含まない MDX/Markdown ファイルのみで構成されています。

3. **隔離された実行**: GitHub Action は、使用後に破棄される隔離された仮想環境でこれらのスクリプトを実行します。

4. **ライセンスへの準拠**: これらのツールは Weave と一緒に配布されないため、setuptools のベンダー依存関係に含まれる LGPL-3.0 コンポーネントは、Weave ユーザーにライセンス上の義務を生じさせません。

<div id="for-organizations-with-strict-license-policies">
  ## 厳格なライセンスポリシーを持つ組織向け
</div>

組織に開発ツールへの LGPL コードを禁止するポリシーがある場合:

1. GitHub Action を使用してクラウドでドキュメントを生成する (推奨) 
2. lazydocs を使用しない最小限の Python ジェネレーターを使用する
3. Docker コンテナーでドキュメントを生成する
4. 開発専用ツールの例外を申請する

<div id="socket-security">
  ## Socket Security
</div>

リポジトリルートにある `.socketignore` ファイルは、これらのスクリプトをセキュリティスキャンの対象から除外します。これらは本番コードではなく、開発用ツールであるためです。

<div id="known-socket-security-warnings">
  ### 既知の Socket Security の警告
</div>

* **ホイール内のネイティブコード**: `wheel` パッケージにはネイティブコードが含まれていますが、これは Python パッケージングツールとして正常な動作です
* **ライセンス違反**: 一部の推移的な依存関係には LGPL またはその他のライセンスが含まれており、ポリシー警告が発生する場合があります

これらの警告が許容される理由は以下のとおりです。

1. これらのツールはドキュメント生成時にのみ使用されます
2. 隔離された CI 環境で実行されます
3. Weave と一緒に配布されることはありません
4. 生成されたドキュメントには実行可能なコードが含まれていません
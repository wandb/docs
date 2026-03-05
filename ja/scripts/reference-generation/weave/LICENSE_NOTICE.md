---
title: "ライセンスに関する通知"
---

<div id="license-notice-for-reference-documentation-generation">
  # リファレンスドキュメント生成に関するライセンス通知
</div>

<div id="overview">
  ## 概要
</div>

このディレクトリ内のスクリプトは、開発／CI プロセス中にリファレンス ドキュメントを生成する目的にのみ使用されます。これらは Weave ライブラリに同梱されたり、本番コードに組み込まれたりすることはありません。

<div id="dependencies-and-their-licenses">
  ## 依存関係およびそのライセンス
</div>

<div id="direct-dependencies">
  ### 直接依存関係
</div>

- **requests** (Apache-2.0): HTTP リクエストに使用
- **lazydocs** (MIT): W&B がメンテナンスしているドキュメント生成ツール

<div id="transitive-dependencies-via-lazydocs">
  ### 間接的な依存関係（lazydocs 経由）
</div>

- **setuptools**（MIT（同梱の LGPL-3.0 コンポーネントを含む））: ビルドシステム
- その他、ライセンスが混在するさまざまな依存関係

<div id="important-notes">
  ## 重要な注意事項
</div>

1. **開発用途のみ**: これらの依存関係は、CI / GitHub Actions でのドキュメント生成時にのみ一時的にインストールされます。配布される Weave パッケージには一切含まれません。

2. **配布対象外**: 生成されるドキュメントは、実行可能なコードや依存関係を含まない MDX / Markdown ファイルのみで構成されています。

3. **分離された実行環境**: GitHub Actions は、これらのスクリプトを使用後に破棄される分離された仮想環境内で実行します。

4. **ライセンス遵守**: これらのツールは Weave と一緒に配布されないため、setuptools の同梱依存関係に含まれる LGPL-3.0 コンポーネントは、Weave ユーザーにライセンス上の義務を生じさせません。

<div id="for-organizations-with-strict-license-policies">
  ## 厳格なライセンスポリシーを持つ組織向け
</div>

所属組織が、開発ツールでの LGPL コードの使用を一切認めないポリシーを持つ場合:

1. GitHub Action を使ってクラウド上でドキュメントを生成する（推奨）
2. `lazydocs` を使用しない最小限の Python ジェネレーターを使う
3. Docker コンテナ内でドキュメントを生成する
4. 開発専用ツールとしての例外許可を申請する

<div id="socket-security">
  ## ソケットセキュリティ
</div>

リポジトリのルートにある `.socketignore` ファイルによって、これらのスクリプトは本番コードではなく開発用ツールであるため、セキュリティスキャンの対象から除外されています。

<div id="known-socket-security-warnings">
  ### 既知のソケットセキュリティ警告
</div>

- **wheel 内のネイティブコード**: `wheel` パッケージにはネイティブコードが含まれていますが、これは Python のパッケージングツールとしては通常の挙動です
- **ライセンス違反**: 一部の推移的依存関係には LGPL などのライセンスが含まれており、ポリシー警告が発生する場合があります

これらの警告を許容できる理由は次のとおりです。

1. これらのツールはドキュメント生成時にのみ使用される
2. それらは分離された CI 環境内で実行される
3. それらが Weave と一緒に配布されることは一切ない
4. 生成されたドキュメントには実行可能なコードは含まれない
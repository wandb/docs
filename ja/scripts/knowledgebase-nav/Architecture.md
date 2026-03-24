<div id="knowledgebase-nav-generator-architecture">
  # Knowledgebase Nav ジェネレーターのアーキテクチャ
</div>

このドキュメントでは、`wandb-docs` リポジトリ内の **Knowledgebase Nav** システムについて説明します。何を生成するのか、どのファイルと関数によって動作しているのか、そしてオートメーションによってそれらがどのように連携しているのかを扱います。作成者向けの手順とローカルでの setup については、[README.md](./README.md) を参照してください。

<div id="purpose">
  ## 目的
</div>

このジェネレーターは、サポート (ナレッジベース) のナビゲーションを記事の内容と整合するように保ちます。設定されたプロダクト (たとえば models、weave、inference) を対象に走査し、`support/<product>/articles/` 配下の MDX 記事を読み取り、生成された MDX ページ、ルートの `support.mdx` の件数、および `docs.json` 内の英語のサポートタブを更新します。

<div id="high-level-context">
  ## 全体像
</div>

このシステムは完全に `wandb-docs` 内で動作します。外部 API は呼び出しません。リポジトリのワーキングツリー内のファイルを読み書きします。

```mermaid
flowchart LR
  subgraph repo["wandb-docs repository"]
    CFG["config.yaml"]
    TPL["templates/*.j2"]
    ART["support/*/articles/*.mdx"]
    GEN["generate_tags.py"]
    OUT1["support/*/tags/*.mdx"]
    OUT2["support/<product>.mdx"]
    DJ["docs.json"]
    SM["support.mdx"]
  end
  CFG --> GEN
  TPL --> GEN
  ART --> GEN
  GEN --> OUT1
  GEN --> OUT2
  GEN --> DJ
  GEN --> SM
  GEN --> ART
```

**articles** に戻る矢印は、フェーズ 4 で MDX コメントマーカーに囲まれた `/support/<product>/tags/` 配下のタグページを指す `<Badge>` リンクだけが更新対象であることを示します。その他のコンテンツ（`---`、他の `<Badge>`、マーカー外のテキストを含む）は書き換えられません。

<div id="automation-workflow">
  ## オートメーション ワークフロー
</div>

`support/**` または `scripts/knowledgebase-nav/**` 配下のファイルが変更されると（オープン中の PR への新しい push を含む）、プルリクエストによって **Knowledgebase Nav** ワークフローがトリガーされます。このワークフローでは Python の依存関係をインストールし、ジェネレーターを実行して、差分がある場合は一致するパスをコミットします。**forks** からのプルリクエストでは fork の head commit を checkout し、ジェネレーターも実行されますが、デフォルトの token では fork に push できないため、自動コミットの step はスキップされます。

```mermaid
flowchart TD
  A[PR or manual workflow_dispatch] --> B[Checkout ref]
  B --> C[Python 3.11 + pip install requirements.txt]
  C --> D["generate_tags.py --repo-root ."]
  D --> E{Files changed?}
  E -->|yes| F[git-auto-commit selected paths]
  E -->|no| G[No commit]
```

コミット対象のパスパターンには、`support.mdx`、`support/*/articles/*.mdx`、`support/*/tags/*.mdx`、`support/*.mdx`（プロダクト インデックス）、および `docs.json` が含まれます。

<div id="pipeline-orchestration">
  ## パイプライン オーケストレーション
</div>

`run_pipeline(repo_root, config_path)` は、CLI と tests で使用する単一のエントリ ポイントです。`config.yaml` を読み込み、すべてのプロダクト向けに 1 つの Jinja2 環境を構築したあと、各プロダクトを順に処理します。ループ終了後、`docs.json` を 1 回だけ更新し、`support.mdx` も 1 回だけ更新します。

```mermaid
flowchart TD
  START([run_pipeline]) --> LOAD[load_config]
  LOAD --> JINJA[create_template_env]
  JINJA --> LOOP{For each product in config}
  LOOP --> P1[crawl_articles]
  P1 --> P2[build_tag_index]
  P2 --> P3[render_tag_pages]
  P3 --> P3b[cleanup_stale_tag_pages]
  P3b --> P4[render_product_index]
  P4 --> P5[sync_all_support_article_footers]
  P5 --> P6[Record product_stats]
  P6 --> LOOP
  LOOP -->|done| P7[update_docs_json]
  P7 --> P8[update_support_index]
  P8 --> P9[update_support_featured]
  P9 --> DONE([Done])
```

<div id="per-product-data-flow">
  ## プロダクトごとのデータフロー
</div>

1 つのプロダクト内では、データは生のファイルからインメモリ構造へ移動し、その後、後続の step で使用するために MDX と集約済みの構造へ戻されます。

```mermaid
flowchart LR
  subgraph inputs["Inputs"]
    MDX["*.mdx articles"]
    KW["allowed_keywords"]
  end
  subgraph memory["In memory"]
    ART["List of article dicts"]
    IDX["tag to articles map"]
    PATHS["Tag page path list"]
  end
  subgraph outputs["Outputs"]
    TAGS["tags/<slug>.mdx"]
    IDXPG["<product>.mdx"]
  end
  MDX --> ART
  KW --> IDX
  ART --> IDX
  ART --> TAGS
  IDX --> TAGS
  IDX --> IDXPG
  ART --> IDXPG
  PATHS --> TAGS
```

`render_tag_pages` は、ソート済みのページ ID 文字列（例: `support/models/tags/security`）を返します。`update_docs_json` はこれをそのプロダクトの英語のナビゲーションタブにマージします。

<div id="components-and-files">
  ## コンポーネントとファイル
</div>

| Component | Path | Role |
|-----------|------|------|
| CLI とロジック | `generate_tags.py` | すべてのフェーズ、パース処理、slug ルール、プレビュー、JSON と MDX の書き換え |
| プロダクトとタグのレジストリ | `config.yaml` | プロダクトごとの `slug`、`display_name`、`allowed_keywords` |
| タグ一覧テンプレート | `templates/support_tag.mdx.j2` | タグページで記事ごとに 1 つの Card を表示 |
| プロダクトハブテンプレート | `templates/support_product_index.mdx.j2` | 注目セクションとカテゴリ別に閲覧するための Card |
| 依存関係 | `requirements.txt` | PyYAML、Jinja2 |
| 単体テスト | `tests/test_generate_tags.py` | モック化したファイルシステムと `docs.json` |
| インテグレーション テスト | `tests/test_golden_output.py` | 実際のリポジトリを一時コピーした環境でフルパイプラインを実行 |
| Pytest マーカー | `tests/conftest.py` | golden スイート用の `integration` マーカーを登録 |
| CI | `.github/workflows/knowledgebase-nav.yml` | トリガー、実行スクリプト、自動コミット |
| 作成者向けドキュメント | `README.md` | ライターと開発者向けのワークフロー |
| アーキテクチャ メモ | `Architecture.md` | 開発者向けの図とモジュール マップ |

<div id="functional-areas-inside-generate_tagspy">
  ## `generate_tags.py` 内の機能領域
</div>

以下では、関数をソース ファイル内での出現順にグループ化しています。各名は Python API でのものです。

<div id="configuration">
  ### 設定
</div>

* **`load_config`** は `config.yaml` を読み込み、各プロダクトで必須のキーが含まれているかを検証します。

<div id="article-structure-and-footers">
  ### 記事の構造とフッター
</div>

* **`parse_frontmatter`**、**`_extract_body`** は YAML front matter と本文を分割します。`_extract_body` は境界として `_BADGE_START` を使用し、末尾の `---` 行は体裁を整えるために取り除きます。
* **`_split_frontmatter_raw`** は、フッターを書き換えるために、生の MDX を front matter ブロックと残りの部分に分割します。
* **`_normalize_keywords`** は、front matter の `keywords` を文字列の list に正規化します (YAML の list。単一の文字列は警告付きで 1 つのタグとして扱われ、それ以外のタイプは警告を出したうえで空の list になります) 。
* **`_keywords_list_for_footer`** は、フッター生成用に正規化した `keywords` を返します (**`_normalize_keywords`** に委譲) 。
* **`_tab_badge_pattern`**、**`build_tab_badges_mdx`**、**`build_keyword_footer_mdx`**、**`_replace_tab_badges_in_body`** は、tab Badge のピンポイントな Sync を実装します。管理対象の Badges は `_BADGE_START` / `_BADGE_END` のマーカーコメントで囲まれます。マーカーがある場合は関数がそれを基準に一致させ、マーカー導入前の記事では regex にフォールバックします。新しいフッターには、空行、マーカー、Badges が追加されます。
* **`sync_support_article_footer`**、**`sync_all_support_article_footers`** は、tab Badges が `keywords` とずれて古くなっている場合に記事ファイルを書き込みます。

<div id="body-previews-card-snippets">
  ### 本文プレビュー (Card スニペット)
</div>

* **`plain_text`** は Markdown (水平線を含む) 、リンク、URL、HTML または MDX タグなどを取り除き、プレビューをプレーンテキストのまま保ちます (entity をデコードした後に U+00A0 をスペースに変換し、引用符は ASCII にマップし、許可リストでは識別子用に `_` と `=` を保持します) 。
* **`extract_body_preview`** は `plain_text` を適用し、`BODY_PREVIEW_MAX_LENGTH` で切り詰め、必要に応じて `BODY_PREVIEW_SUFFIX` を追加します。

<div id="slugs-and-crawling">
  ### スラッグとクロール
</div>

* **`tag_slug`** は Display キーワードをファイル名または URL セグメント (小文字・ハイフン区切り) に変換します。
* **`crawl_articles`** は `support/<slug>/articles/*.mdx` をたどって、記事の dict (`title`、`keywords`、`featured`、`body_preview`、`page_path`、`tag_links` など) を構築します。

<div id="tag-aggregation-and-featured-content">
  ### タグの集約と注目コンテンツ
</div>

- **`get_featured_articles`** は、プロダクトインデックス用の注目記事をフィルターして並べ替えます。
- **`build_tag_index`** は、記事をキーワードごとにグループ化し、各タグ内でタイトル順に並べ替え、`allowed_keywords` に含まれない未知のキーワードがあれば警告します。

<div id="rendering">
  ### レンダリング
</div>

* **`tojson_unicode`**、**`create_template_env`** は、MDX 用に Jinja2 を設定します (テンプレートでは、YAML フロントマターの値に `tojson_unicode` フィルターを使用します) 。
* **`render_tag_pages`** は `support/<product>/tags/<tag-slug>.mdx` に書き出します。
* **`cleanup_stale_tag_pages`** は、tags ディレクトリ内の、今回生成されなかった `.mdx` ファイルを削除し、ディレクトリと `docs.json` に古いエントリーが残らないようにします。
* **`render_product_index`** は `support/<product>.mdx` に書き出します。

<div id="site-wide-updates">
  ### サイト全体の更新
</div>

- **`update_docs_json`** は、`language` が `en` の `navigation.languages` 配下で、非表示の `Support: <display_name>` タブを更新または作成し、`pages` をプロダクトのインデックスとソート済みのタグパスに設定します。
- **`update_support_index`** は、ルートの `support.mdx` にあるプロダクトCard上の件数行を更新します。`{/* auto-generated counts */}` マーカーを優先し、移行時は `regex` にフォールバックします。
- **`update_support_featured`** は、ルートの `support.mdx` で `_FEATURED_START` / `_FEATURED_END` マーカー間の注目記事セクションを再生成します。

<div id="cli">
  ### CLI
</div>

* **`main`** は `--repo-root` と省略可能な `--config` を解析し、続いて **`run_pipeline`** を Call します。

<div id="constants">
  ## 定数
</div>

- **`BODY_PREVIEW_MAX_LENGTH`** と **`BODY_PREVIEW_SUFFIX`** は、Card プレビューの長さと省略記号を制御します。
- **`DOCS_JSON_NAV_LANGUAGE`** は `"en"` で、ナビゲーションの編集対象を英語のツリーのみに限定します。
- **`_BADGE_START`** / **`_BADGE_END`** は、各記事ページで管理対象のタブの Badge を囲む MDX コメントマーカーです。
- **`_FEATURED_START`** / **`_FEATURED_END`** は、ルートの `support.mdx` 内にある注目記事セクションを囲む MDX コメントマーカーです。

<div id="design-choices">
  ## 設計上の選択
</div>

- **モノリシックスクリプト**: 1 つのファイルにすべてのロジックを集約することで、ワークフローとコントリビューターが動作を確認・変更する場所を 1 か所にまとめています。
- **許可されたキーワード**: `config.yaml` には、プロダクトごとの有効なタグを列挙します。不明なタグでもページは生成されますが、警告が出力されるため、コンテンツが気付かないうちに失われることはありません。
- **Tab Badge の管理範囲**: `/support/<product>/tags/...` にリンクする `<Badge>` 要素のみが `keywords` から生成されます。これらはマーカーコメントで囲まれているため、移行後はジェネレーターで regex マッチングを行う必要がありません。本文と badge の間にある `---` 行は見た目のためだけのものです。`_extract_body` は `_BADGE_START` を境界として使用し、末尾の `---` はクリーンアップとしてのみ削除します。
- **古いタグのクリーンアップ**: どの記事キーワードにも対応しなくなったタグページは、生成後、`docs.json` を更新する前に削除されます。これにより、tags ディレクトリやナビゲーションに孤立したエントリが残りません。
- **マーカーベースの編集**: 自動生成されるすべてのセクション（記事タブの Badge、`support.mdx` の件数行、注目記事）では、MDX コメントマーカーを使用します。これにより、管理対象の領域がライターにも明確になり、壊れやすい regex アンカーに頼らずにジェネレーターが内容を正確に置き換えられます。各マーカーペアには、初回実行時にプレーンなコンテンツを囲む移行パスがあります。
- **Golden tests**: 生成されたタグページ、プロダクトインデックスページ、記事ファイル（フッターマーカーを含む）、`docs.json` 内のサポートタブ、およびルートの `support.mdx` をコミット済みのツリーと比較し、出力のずれが unified diff として見えるようにします。

<div id="related-reading">
  ## 関連資料
</div>

* 利用状況、ローカル venv の setup、トラブルシューティングについては [README.md](./README.md) を参照してください。
* Mintlify コンテンツを編集する際のドキュメントスタイルについては、リポジトリルートにある [AGENTS.md](../../AGENTS.md) を参照してください。
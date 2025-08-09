---
title: 文字列
menu:
  reference:
    identifier: ja-ref-query-panel-string
---

## チェーン可能な操作

<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### 戻り値
2つの値が等しくない場合に true となります。

<h3 id="string-add"><code>string-add</code></h3>

2つの [文字列](string.md) を連結します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 連結する最初の [文字列](string.md) |
| `rhs` | 連結する2つ目の [文字列](string.md) |

#### 戻り値
連結された [文字列](string.md)

<h3 id="string-equal"><code>string-equal</code></h3>

2つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### 戻り値
2つの値が等しい場合に true となります。

<h3 id="string-append"><code>string-append</code></h3>

[文字列](string.md) の末尾にサフィックス（接尾語）を追加します。

| 引数 |  |
| :--- | :--- |
| `str` | サフィックスを追加する [文字列](string.md) |
| `suffix` | 追加するサフィックス |

#### 戻り値
サフィックスが追加された [文字列](string.md)

<h3 id="string-contains"><code>string-contains</code></h3>

[文字列](string.md) に部分文字列が含まれているか確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 検査対象の [文字列](string.md) |
| `sub` | チェックする部分文字列 |

#### 戻り値
[文字列](string.md) に部分文字列が含まれていれば true

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[文字列](string.md) の末尾が特定のサフィックスで終了しているか確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 検査対象の [文字列](string.md) |
| `suffix` | チェックするサフィックス |

#### 戻り値
[文字列](string.md) がそのサフィックスで終わる場合に true

<h3 id="string-findAll"><code>string-findAll</code></h3>

[文字列](string.md) 内で部分文字列が現れるすべてのインデックスを取得します。

| 引数 |  |
| :--- | :--- |
| `str` | 検索対象の [文字列](string.md) |
| `sub` | 検索したい部分文字列 |

#### 戻り値
[文字列](string.md) 内の該当する部分文字列のインデックスの _リスト_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[文字列](string.md) が英数字のみで構成されているか判定します。

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の [文字列](string.md) |

#### 戻り値
[文字列](string.md) が英数字のみの場合に true

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[文字列](string.md) がアルファベットのみで構成されているか判定します。

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の [文字列](string.md) |

#### 戻り値
[文字列](string.md) がアルファベットのみの場合に true

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[文字列](string.md) が数字のみで構成されているか判定します。

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の [文字列](string.md) |

#### 戻り値
[文字列](string.md) が数字のみの場合に true

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白文字を取り除きます。

| 引数 |  |
| :--- | :--- |
| `str` | 前方の空白を除去する [文字列](string.md) |

#### 戻り値
空白が除かれた [文字列](string.md)

<h3 id="string-len"><code>string-len</code></h3>

[文字列](string.md) の長さを返します。

| 引数 |  |
| :--- | :--- |
| `str` | 長さを確認する [文字列](string.md) |

#### 戻り値
[文字列](string.md) の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[文字列](string.md) をすべて小文字に変換します。

| 引数 |  |
| :--- | :--- |
| `str` | 小文字に変換したい [文字列](string.md) |

#### 戻り値
小文字化された [文字列](string.md)

<h3 id="string-partition"><code>string-partition</code></h3>

[文字列](string.md) をセパレーターで区切り、_リスト_ 形式で返します。

| 引数 |  |
| :--- | :--- |
| `str` | 分割対象の [文字列](string.md) |
| `sep` | 分割に使うセパレーター |

#### 戻り値
_リスト_ 型：セパレーターの前の [文字列](string.md)、セパレーター、セパレーターの後の [文字列](string.md)

<h3 id="string-prepend"><code>string-prepend</code></h3>

[文字列](string.md) の先頭にプレフィックス（接頭語）を追加します。

| 引数 |  |
| :--- | :--- |
| `str` | プレフィックスを追加する [文字列](string.md) |
| `prefix` | 追加するプレフィックス |

#### 戻り値
プレフィックスが追加された [文字列](string.md)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白文字を取り除きます。

| 引数 |  |
| :--- | :--- |
| `str` | 後方の空白を除去する [文字列](string.md) |

#### 戻り値
空白が除かれた [文字列](string.md)

<h3 id="string-replace"><code>string-replace</code></h3>

[文字列](string.md) のすべての部分文字列を指定した新しい文字列に置き換えます。

| 引数 |  |
| :--- | :--- |
| `str` | 対象となる [文字列](string.md) |
| `sub` | 置き換える部分文字列 |
| `newSub` | 新たに置き換える文字列 |

#### 戻り値
置換後の [文字列](string.md)

<h3 id="string-slice"><code>string-slice</code></h3>

[文字列](string.md) を開始および終了インデックスに基づいて部分文字列として切り出します。

| 引数 |  |
| :--- | :--- |
| `str` | 分割する [文字列](string.md) |
| `begin` | 部分文字列の開始インデックス |
| `end` | 部分文字列の終了インデックス |

#### 戻り値
部分文字列

<h3 id="string-split"><code>string-split</code></h3>

[文字列](string.md) をセパレーターで区切って、_リスト_ の [文字列](string.md) に分割します。

| 引数 |  |
| :--- | :--- |
| `str` | 分割する [文字列](string.md) |
| `sep` | 区切りに使うセパレーター |

#### 戻り値
分割後の [文字列](string.md) の _リスト_

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[文字列](string.md) が特定のプレフィックスで始まるかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の [文字列](string.md) |
| `prefix` | チェックするプレフィックス |

#### 戻り値
[文字列](string.md) がそのプレフィックスで始まる場合に true

<h3 id="string-strip"><code>string-strip</code></h3>

[文字列](string.md) の両端から空白文字を削除します。

| 引数 |  |
| :--- | :--- |
| `str` | 空白を除去する [文字列](string.md) |

#### 戻り値
空白が除去された [文字列](string.md)

<h3 id="string-upper"><code>string-upper</code></h3>

[文字列](string.md) をすべて大文字に変換します。

| 引数 |  |
| :--- | :--- |
| `str` | 大文字に変換したい [文字列](string.md) |

#### 戻り値
大文字化された [文字列](string.md)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

2つの [文字列](string.md) 間のレーベンシュタイン距離（Levenshtein distance）を計算します。

| 引数 |  |
| :--- | :--- |
| `str1` | 最初の [文字列](string.md) |
| `str2` | 2つ目の [文字列](string.md) |

#### 戻り値
2つの [文字列](string.md) 間のレーベンシュタイン距離

## リスト操作

<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### 戻り値
2つの値が等しくない場合に true となります。

<h3 id="string-add"><code>string-add</code></h3>

2つの [文字列](string.md) を連結します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 連結する最初の [文字列](string.md) |
| `rhs` | 連結する2つ目の [文字列](string.md) |

#### 戻り値
連結された [文字列](string.md)

<h3 id="string-equal"><code>string-equal</code></h3>

2つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### 戻り値
2つの値が等しい場合に true となります。

<h3 id="string-append"><code>string-append</code></h3>

[文字列](string.md) の末尾にサフィックス（接尾語）を追加します。

| 引数 |  |
| :--- | :--- |
| `str` | サフィックスを追加する [文字列](string.md) |
| `suffix` | 追加するサフィックス |

#### 戻り値
サフィックスが追加された [文字列](string.md)

<h3 id="string-contains"><code>string-contains</code></h3>

[文字列](string.md) に部分文字列が含まれているか確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 検査対象の [文字列](string.md) |
| `sub` | チェックする部分文字列 |

#### 戻り値
[文字列](string.md) に部分文字列が含まれていれば true

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[文字列](string.md) の末尾が特定のサフィックスで終了しているか確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 検査対象の [文字列](string.md) |
| `suffix` | チェックするサフィックス |

#### 戻り値
[文字列](string.md) がそのサフィックスで終わる場合に true

<h3 id="string-findAll"><code>string-findAll</code></h3>

[文字列](string.md) 内で部分文字列が現れるすべてのインデックスを取得します。

| 引数 |  |
| :--- | :--- |
| `str` | 検索対象の [文字列](string.md) |
| `sub` | 検索したい部分文字列 |

#### 戻り値
[文字列](string.md) 内の該当する部分文字列のインデックスの _リスト_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[文字列](string.md) が英数字のみで構成されているか判定します。

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の [文字列](string.md) |

#### 戻り値
[文字列](string.md) が英数字のみの場合に true

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[文字列](string.md) がアルファベットのみで構成されているか判定します。

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の [文字列](string.md) |

#### 戻り値
[文字列](string.md) がアルファベットのみの場合に true

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[文字列](string.md) が数字のみで構成されているか判定します。

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の [文字列](string.md) |

#### 戻り値
[文字列](string.md) が数字のみの場合に true

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白文字を取り除きます。

| 引数 |  |
| :--- | :--- |
| `str` | 前方の空白を除去する [文字列](string.md) |

#### 戻り値
空白が除かれた [文字列](string.md)

<h3 id="string-len"><code>string-len</code></h3>

[文字列](string.md) の長さを返します。

| 引数 |  |
| :--- | :--- |
| `str` | 長さを確認する [文字列](string.md) |

#### 戻り値
[文字列](string.md) の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[文字列](string.md) をすべて小文字に変換します。

| 引数 |  |
| :--- | :--- |
| `str` | 小文字に変換したい [文字列](string.md) |

#### 戻り値
小文字化された [文字列](string.md)

<h3 id="string-partition"><code>string-partition</code></h3>

[文字列](string.md) をセパレーターで区切り、_リスト_ 形式で返します。

| 引数 |  |
| :--- | :--- |
| `str` | 分割対象の [文字列](string.md) |
| `sep` | 分割に使うセパレーター |

#### 戻り値
_リスト_ 型：セパレーターの前の [文字列](string.md)、セパレーター、セパレーターの後の [文字列](string.md)

<h3 id="string-prepend"><code>string-prepend</code></h3>

[文字列](string.md) の先頭にプレフィックス（接頭語）を追加します。

| 引数 |  |
| :--- | :--- |
| `str` | プレフィックスを追加する [文字列](string.md) |
| `prefix` | 追加するプレフィックス |

#### 戻り値
プレフィックスが追加された [文字列](string.md)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白文字を取り除きます。

| 引数 |  |
| :--- | :--- |
| `str` | 後方の空白を除去する [文字列](string.md) |

#### 戻り値
空白が除かれた [文字列](string.md)

<h3 id="string-replace"><code>string-replace</code></h3>

[文字列](string.md) のすべての部分文字列を指定した新しい文字列に置き換えます。

| 引数 |  |
| :--- | :--- |
| `str` | 対象となる [文字列](string.md) |
| `sub` | 置き換える部分文字列 |
| `newSub` | 新たに置き換える文字列 |

#### 戻り値
置換後の [文字列](string.md)

<h3 id="string-slice"><code>string-slice</code></h3>

[文字列](string.md) を開始および終了インデックスに基づいて部分文字列として切り出します。

| 引数 |  |
| :--- | :--- |
| `str` | 分割する [文字列](string.md) |
| `begin` | 部分文字列の開始インデックス |
| `end` | 部分文字列の終了インデックス |

#### 戻り値
部分文字列

<h3 id="string-split"><code>string-split</code></h3>

[文字列](string.md) をセパレーターで区切って、_リスト_ の [文字列](string.md) に分割します。

| 引数 |  |
| :--- | :--- |
| `str` | 分割する [文字列](string.md) |
| `sep` | 区切りに使うセパレーター |

#### 戻り値
分割後の [文字列](string.md) の _リスト_

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[文字列](string.md) が特定のプレフィックスで始まるかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の [文字列](string.md) |
| `prefix` | チェックするプレフィックス |

#### 戻り値
[文字列](string.md) がそのプレフィックスで始まる場合に true

<h3 id="string-strip"><code>string-strip</code></h3>

[文字列](string.md) の両端から空白文字を削除します。

| 引数 |  |
| :--- | :--- |
| `str` | 空白を除去する [文字列](string.md) |

#### 戻り値
空白が除去された [文字列](string.md)

<h3 id="string-upper"><code>string-upper</code></h3>

[文字列](string.md) をすべて大文字に変換します。

| 引数 |  |
| :--- | :--- |
| `str` | 大文字に変換したい [文字列](string.md) |

#### 戻り値
大文字化された [文字列](string.md)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

2つの [文字列](string.md) 間のレーベンシュタイン距離（Levenshtein distance）を計算します。

| 引数 |  |
| :--- | :--- |
| `str1` | 最初の [文字列](string.md) |
| `str2` | 2つ目の [文字列](string.md) |

#### 戻り値
2つの [文字列](string.md) 間のレーベンシュタイン距離
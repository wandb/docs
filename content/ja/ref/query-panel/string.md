---
title: 申し訳ありませんが、翻訳するドキュメントの内容が提供されていないようです。翻訳が必要なテキストを提供してください。あなたの指示に従って翻訳を行います。
menu:
  reference:
    identifier: ja-ref-query-panel-string
---

## Chainable Ops
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2つの値の不等を判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### Return Value
2つの値が等しくないかどうか。

<h3 id="string-add"><code>string-add</code></h3>

2つの[string](string.md)を連結します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[string](string.md) |
| `rhs` | 2つ目の[string](string.md) |

#### Return Value
連結された[string](string.md)

<h3 id="string-equal"><code>string-equal</code></h3>

2つの値の等価性を判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### Return Value
2つの値が等しいかどうか。

<h3 id="string-append"><code>string-append</code></h3>

接尾辞を[string](string.md)に追加します。

| 引数 |  |
| :--- | :--- |
| `str` | 追加する[string](string.md) |
| `suffix` | 追加する接尾辞 |

#### Return Value
接尾辞が追加された[string](string.md)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](string.md)が部分文字列を含んでいるかを確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](string.md) |
| `sub` | 確認する部分文字列 |

#### Return Value
[string](string.md)が部分文字列を含んでいるかどうか。

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](string.md)が接尾辞で終わるかを確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](string.md) |
| `suffix` | 確認する接尾辞 |

#### Return Value
[string](string.md)が接尾辞で終わるかどうか。

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](string.md)内の部分文字列のすべての出現を見つけます。

| 引数 |  |
| :--- | :--- |
| `str` | 部分文字列の出現を見つける[string](string.md) |
| `sub` | 見つける部分文字列 |

#### Return Value
[string](string.md)内の部分文字列のインデックスの_list_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](string.md)が英数字かどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](string.md) |

#### Return Value
[string](string.md)が英数字かどうか。

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[string](string.md)がアルファベット文字かどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](string.md) |

#### Return Value
[string](string.md)がアルファベット文字かどうか。

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[string](string.md)が数値かどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](string.md) |

#### Return Value
[string](string.md)が数値かどうか。

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白を削除します。

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[string](string.md)。 |

#### Return Value
空白が削除された[string](string.md)。

<h3 id="string-len"><code>string-len</code></h3>

[string](string.md)の長さを返します。

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](string.md) |

#### Return Value
[string](string.md)の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[string](string.md)を小文字に変換します。

| 引数 |  |
| :--- | :--- |
| `str` | 小文字に変換する[string](string.md) |

#### Return Value
小文字に変換された[string](string.md)

<h3 id="string-partition"><code>string-partition</code></h3>

[string](string.md)を_list_にパーティション分けします。

| 引数 |  |
| :--- | :--- |
| `str` | 分割する[string](string.md) |
| `sep` | 分割に使用するセパレータ |

#### Return Value
セパレータの前の[string](string.md)、セパレータ、セパレータの後の[string](string.md)を含む_list_の[string](string.md)

<h3 id="string-prepend"><code>string-prepend</code></h3>

接頭辞を[string](string.md)に追加します。

| 引数 |  |
| :--- | :--- |
| `str` | 追加する[string](string.md) |
| `prefix` | 追加する接頭辞 |

#### Return Value
接頭辞が追加された[string](string.md)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白を削除します。

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[string](string.md)。 |

#### Return Value
空白が削除された[string](string.md)。

<h3 id="string-replace"><code>string-replace</code></h3>

[string](string.md)内のすべての部分文字列を置換します。

| 引数 |  |
| :--- | :--- |
| `str` | 内容を置換する[string](string.md) |
| `sub` | 置換する部分文字列 |
| `newSub` | 古い部分文字列を置換する部分文字列 |

#### Return Value
置換された[string](string.md)

<h3 id="string-slice"><code>string-slice</code></h3>

開始インデックスと終了インデックスに基づいて[string](string.md)をスライスします。

| 引数 |  |
| :--- | :--- |
| `str` | スライスする[string](string.md) |
| `begin` | 部分文字列の開始インデックス |
| `end` | 部分文字列の終了インデックス |

#### Return Value
部分文字列

<h3 id="string-split"><code>string-split</code></h3>

[string](string.md)を_list_に分割します。

| 引数 |  |
| :--- | :--- |
| `str` | 分割する[string](string.md) |
| `sep` | 分割に使用するセパレータ |

#### Return Value
_list_の[string](string.md)

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](string.md)が接頭辞で始まるか確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](string.md) |
| `prefix` | 確認する接頭辞 |

#### Return Value
[string](string.md)が接頭辞で始まるかどうか。

<h3 id="string-strip"><code>string-strip</code></h3>

[string](string.md)の両端の空白を削除します。

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[string](string.md)。 |

#### Return Value
空白が削除された[string](string.md)。

<h3 id="string-upper"><code>string-upper</code></h3>

[string](string.md)を大文字に変換します。

| 引数 |  |
| :--- | :--- |
| `str` | 大文字に変換する[string](string.md) |

#### Return Value
大文字に変換された[string](string.md)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

2つの[string](string.md)間のレーベンシュタイン距離を計算します。

| 引数 |  |
| :--- | :--- |
| `str1` | 最初の[string](string.md)。 |
| `str2` | 2つ目の[string](string.md)。 |

#### Return Value
2つの[string](string.md)間のレーベンシュタイン距離


## List Ops
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2つの値の不等を判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### Return Value
2つの値が等しくないかどうか。

<h3 id="string-add"><code>string-add</code></h3>

2つの[string](string.md)を連結します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[string](string.md) |
| `rhs` | 2つ目の[string](string.md) |

#### Return Value
連結された[string](string.md)

<h3 id="string-equal"><code>string-equal</code></h3>

2つの値の等価性を判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### Return Value
2つの値が等しいかどうか。

<h3 id="string-append"><code>string-append</code></h3>

接尾辞を[string](string.md)に追加します。

| 引数 |  |
| :--- | :--- |
| `str` | 追加する[string](string.md) |
| `suffix` | 追加する接尾辞 |

#### Return Value
接尾辞が追加された[string](string.md)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](string.md)が部分文字列を含んでいるかを確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](string.md) |
| `sub` | 確認する部分文字列 |

#### Return Value
[string](string.md)が部分文字列を含んでいるかどうか。

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](string.md)が接尾辞で終わるかを確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](string.md) |
| `suffix` | 確認する接尾辞 |

#### Return Value
[string](string.md)が接尾辞で終わるかどうか。

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](string.md)内の部分文字列のすべての出現を見つけます。

| 引数 |  |
| :--- | :--- |
| `str` | 部分文字列の出現を見つける[string](string.md) |
| `sub` | 見つける部分文字列 |

#### Return Value
[string](string.md)内の部分文字列のインデックスの_list_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](string.md)が英数字かどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](string.md) |

#### Return Value
[string](string.md)が英数字かどうか。

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[string](string.md)がアルファベット文字かどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](string.md) |

#### Return Value
[string](string.md)がアルファベット文字かどうか。

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[string](string.md)が数値かどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](string.md) |

#### Return Value
[string](string.md)が数値かどうか。

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白を削除します。

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[string](string.md)。 |

#### Return Value
空白が削除された[string](string.md)。

<h3 id="string-len"><code>string-len</code></h3>

[string](string.md)の長さを返します。

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](string.md) |

#### Return Value
[string](string.md)の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[string](string.md)を小文字に変換します。

| 引数 |  |
| :--- | :--- |
| `str` | 小文字に変換する[string](string.md) |

#### Return Value
小文字に変換された[string](string.md)

<h3 id="string-partition"><code>string-partition</code></h3>

[string](string.md)を_list_にパーティション分けします。

| 引数 |  |
| :--- | :--- |
| `str` | 分割する[string](string.md) |
| `sep` | 分割に使用するセパレータ |

#### Return Value
セパレータの前の[string](string.md)、セパレータ、セパレータの後の[string](string.md)を含む_list_の[string](string.md)

<h3 id="string-prepend"><code>string-prepend</code></h3>

接頭辞を[string](string.md)に追加します。

| 引数 |  |
| :--- | :--- |
| `str` | 追加する[string](string.md) |
| `prefix` | 追加する接頭辞 |

#### Return Value
接頭辞が追加された[string](string.md)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白を削除します。

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[string](string.md)。 |

#### Return Value
空白が削除された[string](string.md)。

<h3 id="string-replace"><code>string-replace</code></h3>

[string](string.md)内のすべての部分文字列を置換します。

| 引数 |  |
| :--- | :--- |
| `str` | 内容を置換する[string](string.md) |
| `sub` | 置換する部分文字列 |
| `newSub` | 古い部分文字列を置換する部分文字列 |

#### Return Value
置換された[string](string.md)

<h3 id="string-slice"><code>string-slice</code></h3>

開始インデックスと終了インデックスに基づいて[string](string.md)をスライスします。

| 引数 |  |
| :--- | :--- |
| `str` | スライスする[string](string.md) |
| `begin` | 部分文字列の開始インデックス |
| `end` | 部分文字列の終了インデックス |

#### Return Value
部分文字列

<h3 id="string-split"><code>string-split</code></h3>

[string](string.md)を_list_に分割します。

| 引数 |  |
| :--- | :--- |
| `str` | 分割する[string](string.md) |
| `sep` | 分割に使用するセパレータ |

#### Return Value
_list_の[string](string.md)

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](string.md)が接頭辞で始まるか確認します。

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](string.md) |
| `prefix` | 確認する接頭辞 |

#### Return Value
[string](string.md)が接頭辞で始まるかどうか。

<h3 id="string-strip"><code>string-strip</code></h3>

[string](string.md)の両端の空白を削除します。

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[string](string.md)。 |

#### Return Value
空白が削除された[string](string.md)。

<h3 id="string-upper"><code>string-upper</code></h3>

[string](string.md)を大文字に変換します。

| 引数 |  |
| :--- | :--- |
| `str` | 大文字に変換する[string](string.md) |

#### Return Value
大文字に変換された[string](string.md)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

2つの[string](string.md)間のレーベンシュタイン距離を計算します。

| 引数 |  |
| :--- | :--- |
| `str1` | 最初の[string](string.md)。 |
| `str2` | 2つ目の[string](string.md)。 |

#### Return Value
2つの[string](string.md)間のレーベンシュタイン距離

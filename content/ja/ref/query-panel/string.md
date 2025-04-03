---
title: string
menu:
  reference:
    identifier: ja-ref-query-panel-string
---

## Chainable Ops
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2つの 値 が等しくないかどうかを判断します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する最初の 値 。 |
| `rhs` | 比較する2番目の 値 。 |

#### Return Value
2つの 値 が等しくないかどうか。

<h3 id="string-add"><code>string-add</code></h3>

2つの [strings](string.md) を連結します。

| Argument |  |
| :--- | :--- |
| `lhs` | 最初の [string](string.md) |
| `rhs` | 2番目の [string](string.md) |

#### Return Value
連結された [string](string.md)

<h3 id="string-equal"><code>string-equal</code></h3>

2つの 値 が等しいかどうかを判断します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する最初の 値 。 |
| `rhs` | 比較する2番目の 値 。 |

#### Return Value
2つの 値 が等しいかどうか。

<h3 id="string-append"><code>string-append</code></h3>

[string](string.md) にサフィックスを追加します。

| Argument |  |
| :--- | :--- |
| `str` | 追加先の [string](string.md) |
| `suffix` | 追加するサフィックス |

#### Return Value
サフィックスが追加された [string](string.md)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](string.md) が部分 文字列 を含むかどうかを確認します。

| Argument |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |
| `sub` | 確認する部分 文字列 |

#### Return Value
[string](string.md) が部分 文字列 を含むかどうか

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](string.md) がサフィックスで終わるかどうかを確認します。

| Argument |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |
| `suffix` | 確認するサフィックス |

#### Return Value
[string](string.md) がサフィックスで終わるかどうか

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](string.md) 内の部分 文字列 のすべての出現箇所を検索します。

| Argument |  |
| :--- | :--- |
| `str` | 部分 文字列 の出現箇所を検索する [string](string.md) |
| `sub` | 検索する部分 文字列 |

#### Return Value
[string](string.md) 内の部分 文字列 のインデックスの _list_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](string.md) が英数字かどうかを確認します。

| Argument |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |

#### Return Value
[string](string.md) が英数字かどうか

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[string](string.md) がアルファベットかどうかを確認します。

| Argument |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |

#### Return Value
[string](string.md) がアルファベットかどうか

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[string](string.md) が数値かどうかを確認します。

| Argument |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |

#### Return Value
[string](string.md) が数値かどうか

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白を削除します。

| Argument |  |
| :--- | :--- |
| `str` | 削除する [string](string.md)。 |

#### Return Value
削除された [string](string.md)。

<h3 id="string-len"><code>string-len</code></h3>

[string](string.md) の長さを返します。

| Argument |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |

#### Return Value
[string](string.md) の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[string](string.md) を小文字に変換します。

| Argument |  |
| :--- | :--- |
| `str` | 小文字に変換する [string](string.md) |

#### Return Value
小文字の [string](string.md)

<h3 id="string-partition"><code>string-partition</code></h3>

[string](string.md) を [strings](string.md) の _list_ に分割します。

| Argument |  |
| :--- | :--- |
| `str` | 分割する [string](string.md) |
| `sep` | 分割に使用する区切り 文字 |

#### Return Value
[strings](string.md) の _list_：区切り 文字 の前の [string](string.md)、区切り 文字 、区切り 文字 の後の [string](string.md)

<h3 id="string-prepend"><code>string-prepend</code></h3>

[string](string.md) の前にプレフィックスを追加します。

| Argument |  |
| :--- | :--- |
| `str` | 前に追加する [string](string.md) |
| `prefix` | 前に追加するプレフィックス |

#### Return Value
プレフィックスが前に追加された [string](string.md)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白を削除します。

| Argument |  |
| :--- | :--- |
| `str` | 削除する [string](string.md)。 |

#### Return Value
削除された [string](string.md)。

<h3 id="string-replace"><code>string-replace</code></h3>

[string](string.md) 内の部分 文字列 のすべての出現箇所を置き換えます。

| Argument |  |
| :--- | :--- |
| `str` | 内容を置き換える [string](string.md) |
| `sub` | 置き換える部分 文字列 |
| `newSub` | 古い部分 文字列 と置き換える部分 文字列 |

#### Return Value
置き換えられた [string](string.md)

<h3 id="string-slice"><code>string-slice</code></h3>

開始インデックスと終了インデックスに基づいて、[string](string.md) を部分 文字列 にスライスします。

| Argument |  |
| :--- | :--- |
| `str` | スライスする [string](string.md) |
| `begin` | 部分 文字列 の開始インデックス |
| `end` | 部分 文字列 の終了インデックス |

#### Return Value
部分 文字列

<h3 id="string-split"><code>string-split</code></h3>

[string](string.md) を [strings](string.md) の _list_ に分割します。

| Argument |  |
| :--- | :--- |
| `str` | 分割する [string](string.md) |
| `sep` | 分割に使用する区切り 文字 |

#### Return Value
[strings](string.md) の _list_

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](string.md) がプレフィックスで始まるかどうかを確認します。

| Argument |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |
| `prefix` | 確認するプレフィックス |

#### Return Value
[string](string.md) がプレフィックスで始まるかどうか

<h3 id="string-strip"><code>string-strip</code></h3>

[string](string.md) の両端から空白を削除します。

| Argument |  |
| :--- | :--- |
| `str` | 削除する [string](string.md)。 |

#### Return Value
削除された [string](string.md)。

<h3 id="string-upper"><code>string-upper</code></h3>

[string](string.md) を大文字に変換します。

| Argument |  |
| :--- | :--- |
| `str` | 大文字に変換する [string](string.md) |

#### Return Value
大文字の [string](string.md)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

2つの [strings](string.md) 間のレーベンシュタイン距離を計算します。

| Argument |  |
| :--- | :--- |
| `str1` | 最初の [string](string.md)。 |
| `str2` | 2番目の [string](string.md)。 |

#### Return Value
2つの [strings](string.md) 間のレーベンシュタイン距離。


## List Ops
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2つの 値 が等しくないかどうかを判断します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する最初の 値 。 |
| `rhs` | 比較する2番目の 値 。 |

#### Return Value
2つの 値 が等しくないかどうか。

<h3 id="string-add"><code>string-add</code></h3>

2つの [strings](string.md) を連結します。

| Argument |  |
| :--- | :--- |
| `lhs` | 最初の [string](string.md) |
| `rhs` | 2番目の [string](string.md) |

#### Return Value
連結された [string](string.md)

<h3 id="string-equal"><code>string-equal</code></h3>

2つの 値 が等しいかどうかを判断します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する最初の 値 。 |
| `rhs` | 比較する2番目の 値 。 |

#### Return Value
2つの 値 が等しいかどうか。

<h3 id="string-append"><code>string-append</code></h3>

[string](string.md) にサフィックスを追加します。

| Argument |  |
| :--- | :--- |
| `str` | 追加先の [string](string.md) |
| `suffix` | 追加するサフィックス |

#### Return Value
サフィックスが追加された [string](string.md)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](string.md) が部分 文字列 を含むかどうかを確認します。

| Argument |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |
| `sub` | 確認する部分 文字列 |

#### Return Value
[string](string.md) が部分 文字列 を含むかどうか

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](string.md) がサフィックスで終わるかどうかを確認します。

| Argument |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |
| `suffix` | 確認するサフィックス |

#### Return Value
[string](string.md) がサフィックスで終わるかどうか

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](string.md) 内の部分 文字列 のすべての出現箇所を検索します。

| Argument |  |
| :--- | :--- |
| `str` | 部分 文字列 の出現箇所を検索する [string](string.md) |
| `sub` | 検索する部分 文字列 |

#### Return Value
[string](string.md) 内の部分 文字列 のインデックスの _list_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](string.md) が英数字かどうかを確認します。

| Argument |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |

#### Return Value
[string](string.md) が英数字かどうか

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[string](string.md) がアルファベットかどうかを確認します。

| Argument |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |

#### Return Value
[string](string.md) がアルファベットかどうか

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[string](string.md) が数値かどうかを確認します。

| Argument |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |

#### Return Value
[string](string.md) が数値かどうか

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白を削除します。

| Argument |  |
| :--- | :--- |
| `str` | 削除する [string](string.md)。 |

#### Return Value
削除された [string](string.md)。

<h3 id="string-len"><code>string-len</code></h3>

[string](string.md) の長さを返します。

| Argument |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |

#### Return Value
[string](string.md) の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[string](string.md) を小文字に変換します。

| Argument |  |
| :--- | :--- |
| `str` | 小文字に変換する [string](string.md) |

#### Return Value
小文字の [string](string.md)

<h3 id="string-partition"><code>string-partition</code></h3>

[string](string.md) を [strings](string.md) の _list_ に分割します。

| Argument |  |
| :--- | :--- |
| `str` | 分割する [string](string.md) |
| `sep` | 分割に使用する区切り 文字 |

#### Return Value
[strings](string.md) の _list_：区切り 文字 の前の [string](string.md)、区切り 文字 、区切り 文字 の後の [string](string.md)

<h3 id="string-prepend"><code>string-prepend</code></h3>

[string](string.md) の前にプレフィックスを追加します。

| Argument |  |
| :--- | :--- |
| `str` | 前に追加する [string](string.md) |
| `prefix` | 前に追加するプレフィックス |

#### Return Value
プレフィックスが前に追加された [string](string.md)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白を削除します。

| Argument |  |
| :--- | :--- |
| `str` | 削除する [string](string.md)。 |

#### Return Value
削除された [string](string.md)。

<h3 id="string-replace"><code>string-replace</code></h3>

[string](string.md) 内の部分 文字列 のすべての出現箇所を置き換えます。

| Argument |  |
| :--- | :--- |
| `str` | 内容を置き換える [string](string.md) |
| `sub` | 置き換える部分 文字列 |
| `newSub` | 古い部分 文字列 と置き換える部分 文字列 |

#### Return Value
置き換えられた [string](string.md)

<h3 id="string-slice"><code>string-slice</code></h3>

開始インデックスと終了インデックスに基づいて、[string](string.md) を部分 文字列 にスライスします。

| Argument |  |
| :--- | :--- |
| `str` | スライスする [string](string.md) |
| `begin` | 部分 文字列 の開始インデックス |
| `end` | 部分 文字列 の終了インデックス |

#### Return Value
部分 文字列

<h3 id="string-split"><code>string-split</code></h3>

[string](string.md) を [strings](string.md) の _list_ に分割します。

| Argument |  |
| :--- | :--- |
| `str` | 分割する [string](string.md) |
| `sep` | 分割に使用する区切り 文字 |

#### Return Value
[strings](string.md) の _list_

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](string.md) がプレフィックスで始まるかどうかを確認します。

| Argument |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |
| `prefix` | 確認するプレフィックス |

#### Return Value
[string](string.md) がプレフィックスで始まるかどうか

<h3 id="string-strip"><code>string-strip</code></h3>

[string](string.md) の両端から空白を削除します。

| Argument |  |
| :--- | :--- |
| `str` | 削除する [string](string.md)。 |

#### Return Value
削除された [string](string.md)。

<h3 id="string-upper"><code>string-upper</code></h3>

[string](string.md) を大文字に変換します。

| Argument |  |
| :--- | :--- |
| `str` | 大文字に変換する [string](string.md) |

#### Return Value
大文字の [string](string.md)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

2つの [strings](string.md) 間のレーベンシュタイン距離を計算します。

| Argument |  |
| :--- | :--- |
| `str1` | 最初の [string](string.md)。 |
| `str2` | 2番目の [string](string.md)。 |

#### Return Value
2つの [strings](string.md) 間のレーベンシュタイン距離。

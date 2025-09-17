---
title: 文字列
menu:
  reference:
    identifier: ja-ref-query-panel-string
---

## 連結可能な演算
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2 つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する 1 つ目の値。 |
| `rhs` | 比較する 2 つ目の値。 |

#### 戻り値
2 つの値が等しくないかどうか。

<h3 id="string-add"><code>string-add</code></h3>

2 つの [文字列](string.md) を連結します

| 引数 |  |
| :--- | :--- |
| `lhs` | 1 つ目の [文字列](string.md) |
| `rhs` | 2 つ目の [文字列](string.md) |

#### 戻り値
連結された [文字列](string.md)

<h3 id="string-equal"><code>string-equal</code></h3>

2 つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する 1 つ目の値。 |
| `rhs` | 比較する 2 つ目の値。 |

#### 戻り値
2 つの値が等しいかどうか。

<h3 id="string-append"><code>string-append</code></h3>

[文字列](string.md) に接尾辞を追加します

| 引数 |  |
| :--- | :--- |
| `str` | 追加先の [文字列](string.md) |
| `suffix` | 追加する接尾辞 |

#### 戻り値
接尾辞が追加された [文字列](string.md)

<h3 id="string-contains"><code>string-contains</code></h3>

[文字列](string.md) が部分文字列を含むかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する [文字列](string.md) |
| `sub` | 確認する部分文字列 |

#### 戻り値
その [文字列](string.md) が部分文字列を含むかどうか

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[文字列](string.md) が指定の接尾辞で終わるかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する [文字列](string.md) |
| `suffix` | 確認する接尾辞 |

#### 戻り値
その [文字列](string.md) が接尾辞で終わるかどうか

<h3 id="string-findAll"><code>string-findAll</code></h3>

[文字列](string.md) 内で部分文字列が現れるすべての箇所を検索します

| 引数 |  |
| :--- | :--- |
| `str` | 部分文字列の出現箇所を探す対象の [文字列](string.md) |
| `sub` | 検索する部分文字列 |

#### 戻り値
その [文字列](string.md) 内における部分文字列のインデックスの _リスト_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[文字列](string.md) が英数字かどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する [文字列](string.md) |

#### 戻り値
その [文字列](string.md) が英数字かどうか

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[文字列](string.md) がアルファベットかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する [文字列](string.md) |

#### 戻り値
その [文字列](string.md) がアルファベットかどうか

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[文字列](string.md) が数字かどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する [文字列](string.md) |

#### 戻り値
その [文字列](string.md) が数字かどうか

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する対象の [文字列](string.md)。 |

#### 戻り値
空白が削除された [文字列](string.md)。

<h3 id="string-len"><code>string-len</code></h3>

[文字列](string.md) の長さを返します

| 引数 |  |
| :--- | :--- |
| `str` | 長さを確認する [文字列](string.md) |

#### 戻り値
その [文字列](string.md) の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[文字列](string.md) を小文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | 小文字に変換する [文字列](string.md) |

#### 戻り値
小文字に変換された [文字列](string.md)

<h3 id="string-partition"><code>string-partition</code></h3>

[文字列](string.md) を _リスト_ の [文字列](string.md) に分割します

| 引数 |  |
| :--- | :--- |
| `str` | 分割する [文字列](string.md) |
| `sep` | 分割に使う区切り文字 |

#### 戻り値
[文字列](string.md) の _リスト_ : 区切り文字より前の [文字列](string.md)、区切り文字自体、区切り文字より後ろの [文字列](string.md)

<h3 id="string-prepend"><code>string-prepend</code></h3>

[文字列](string.md) に接頭辞を追加します

| 引数 |  |
| :--- | :--- |
| `str` | 追加先の [文字列](string.md) |
| `prefix` | 追加する接頭辞 |

#### 戻り値
接頭辞が追加された [文字列](string.md)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する対象の [文字列](string.md)。 |

#### 戻り値
空白が削除された [文字列](string.md)。

<h3 id="string-replace"><code>string-replace</code></h3>

[文字列](string.md) 内にあるすべての部分文字列を置換します

| 引数 |  |
| :--- | :--- |
| `str` | 置換対象の [文字列](string.md) |
| `sub` | 置換する部分文字列 |
| `newSub` | 既存の部分文字列と置き換える新しい部分文字列 |

#### 戻り値
置換が適用された [文字列](string.md)

<h3 id="string-slice"><code>string-slice</code></h3>

開始・終了インデックスに基づいて [文字列](string.md) を部分文字列にスライスします

| 引数 |  |
| :--- | :--- |
| `str` | スライスする [文字列](string.md) |
| `begin` | 部分文字列の開始インデックス |
| `end` | 部分文字列の終了インデックス |

#### 戻り値
部分文字列

<h3 id="string-split"><code>string-split</code></h3>

[文字列](string.md) を [文字列](string.md) の _リスト_ に分割します

| 引数 |  |
| :--- | :--- |
| `str` | 分割する [文字列](string.md) |
| `sep` | 分割に使う区切り文字 |

#### 戻り値
[文字列](string.md) の _リスト_

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[文字列](string.md) が指定の接頭辞で始まるかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する [文字列](string.md) |
| `prefix` | 確認する接頭辞 |

#### 戻り値
その [文字列](string.md) が接頭辞で始まるかどうか

<h3 id="string-strip"><code>string-strip</code></h3>

[文字列](string.md) の両端の空白を削除します。

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する対象の [文字列](string.md)。 |

#### 戻り値
空白が削除された [文字列](string.md)。

<h3 id="string-upper"><code>string-upper</code></h3>

[文字列](string.md) を大文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | 大文字に変換する [文字列](string.md) |

#### 戻り値
大文字に変換された [文字列](string.md)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

2 つの [文字列](string.md) 間のレーベンシュタイン距離を計算します。

| 引数 |  |
| :--- | :--- |
| `str1` | 1 つ目の [文字列](string.md)。 |
| `str2` | 2 つ目の [文字列](string.md)。 |

#### 戻り値
2 つの [文字列](string.md) 間のレーベンシュタイン距離。


## リスト演算
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2 つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する 1 つ目の値。 |
| `rhs` | 比較する 2 つ目の値。 |

#### 戻り値
2 つの値が等しくないかどうか。

<h3 id="string-add"><code>string-add</code></h3>

2 つの [文字列](string.md) を連結します

| 引数 |  |
| :--- | :--- |
| `lhs` | 1 つ目の [文字列](string.md) |
| `rhs` | 2 つ目の [文字列](string.md) |

#### 戻り値
連結された [文字列](string.md)

<h3 id="string-equal"><code>string-equal</code></h3>

2 つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する 1 つ目の値。 |
| `rhs` | 比較する 2 つ目の値。 |

#### 戻り値
2 つの値が等しいかどうか。

<h3 id="string-append"><code>string-append</code></h3>

[文字列](string.md) に接尾辞を追加します

| 引数 |  |
| :--- | :--- |
| `str` | 追加先の [文字列](string.md) |
| `suffix` | 追加する接尾辞 |

#### 戻り値
接尾辞が追加された [文字列](string.md)

<h3 id="string-contains"><code>string-contains</code></h3>

[文字列](string.md) が部分文字列を含むかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する [文字列](string.md) |
| `sub` | 確認する部分文字列 |

#### 戻り値
その [文字列](string.md) が部分文字列を含むかどうか

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[文字列](string.md) が指定の接尾辞で終わるかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する [文字列](string.md) |
| `suffix` | 確認する接尾辞 |

#### 戻り値
その [文字列](string.md) が接尾辞で終わるかどうか

<h3 id="string-findAll"><code>string-findAll</code></h3>

[文字列](string.md) 内で部分文字列が現れるすべての箇所を検索します

| 引数 |  |
| :--- | :--- |
| `str` | 部分文字列の出現箇所を探す対象の [文字列](string.md) |
| `sub` | 検索する部分文字列 |

#### 戻り値
その [文字列](string.md) 内における部分文字列のインデックスの _リスト_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[文字列](string.md) が英数字かどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する [文字列](string.md) |

#### 戻り値
その [文字列](string.md) が英数字かどうか

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[文字列](string.md) がアルファベットかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する [文字列](string.md) |

#### 戻り値
その [文字列](string.md) がアルファベットかどうか

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[文字列](string.md) が数字かどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する [文字列](string.md) |

#### 戻り値
その [文字列](string.md) が数字かどうか

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する対象の [文字列](string.md)。 |

#### 戻り値
空白が削除された [文字列](string.md)。

<h3 id="string-len"><code>string-len</code></h3>

[文字列](string.md) の長さを返します

| 引数 |  |
| :--- | :--- |
| `str` | 長さを確認する [文字列](string.md) |

#### 戻り値
その [文字列](string.md) の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[文字列](string.md) を小文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | 小文字に変換する [文字列](string.md) |

#### 戻り値
小文字に変換された [文字列](string.md)

<h3 id="string-partition"><code>string-partition</code></h3>

[文字列](string.md) を _リスト_ の [文字列](string.md) に分割します

| 引数 |  |
| :--- | :--- |
| `str` | 分割する [文字列](string.md) |
| `sep` | 分割に使う区切り文字 |

#### 戻り値
[文字列](string.md) の _リスト_ : 区切り文字より前の [文字列](string.md)、区切り文字自体、区切り文字より後ろの [文字列](string.md)

<h3 id="string-prepend"><code>string-prepend</code></h3>

[文字列](string.md) に接頭辞を追加します

| 引数 |  |
| :--- | :--- |
| `str` | 追加先の [文字列](string.md) |
| `prefix` | 追加する接頭辞 |

#### 戻り値
接頭辞が追加された [文字列](string.md)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する対象の [文字列](string.md)。 |

#### 戻り値
空白が削除された [文字列](string.md)。

<h3 id="string-replace"><code>string-replace</code></h3>

[文字列](string.md) 内にあるすべての部分文字列を置換します

| 引数 |  |
| :--- | :--- |
| `str` | 置換対象の [文字列](string.md) |
| `sub` | 置換する部分文字列 |
| `newSub` | 既存の部分文字列と置き換える新しい部分文字列 |

#### 戻り値
置換が適用された [文字列](string.md)

<h3 id="string-slice"><code>string-slice</code></h3>

開始・終了インデックスに基づいて [文字列](string.md) を部分文字列にスライスします

| 引数 |  |
| :--- | :--- |
| `str` | スライスする [文字列](string.md) |
| `begin` | 部分文字列の開始インデックス |
| `end` | 部分文字列の終了インデックス |

#### 戻り値
部分文字列

<h3 id="string-split"><code>string-split</code></h3>

[文字列](string.md) を [文字列](string.md) の _リスト_ に分割します

| 引数 |  |
| :--- | :--- |
| `str` | 分割する [文字列](string.md) |
| `sep` | 分割に使う区切り文字 |

#### 戻り値
[文字列](string.md) の _リスト_

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[文字列](string.md) が指定の接頭辞で始まるかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する [文字列](string.md) |
| `prefix` | 確認する接頭辞 |

#### 戻り値
その [文字列](string.md) が接頭辞で始まるかどうか

<h3 id="string-strip"><code>string-strip</code></h3>

[文字列](string.md) の両端の空白を削除します。

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する対象の [文字列](string.md)。 |

#### 戻り値
空白が削除された [文字列](string.md)。

<h3 id="string-upper"><code>string-upper</code></h3>

[文字列](string.md) を大文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | 大文字に変換する [文字列](string.md) |

#### 戻り値
大文字に変換された [文字列](string.md)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

2 つの [文字列](string.md) 間のレーベンシュタイン距離を計算します。

| 引数 |  |
| :--- | :--- |
| `str1` | 1 つ目の [文字列](string.md)。 |
| `str2` | 2 つ目の [文字列](string.md)。 |

#### 戻り値
2 つの [文字列](string.md) 間のレーベンシュタイン距離。
---
title: 文字列
---

## チェイン可能な操作（Chainable Ops）
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値 |
| `rhs` | 比較する2番目の値 |

#### 戻り値
2つの値が等しくない場合は真を返します。

<h3 id="string-add"><code>string-add</code></h3>

2つの [string](string.md) を連結します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [string](string.md) |
| `rhs` | 2番目の [string](string.md) |

#### 戻り値
連結された [string](string.md)

<h3 id="string-equal"><code>string-equal</code></h3>

2つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値 |
| `rhs` | 比較する2番目の値 |

#### 戻り値
2つの値が等しい場合は真を返します。

<h3 id="string-append"><code>string-append</code></h3>

[suffix](string.md) を [string](string.md) の末尾に追加します

| 引数 |  |
| :--- | :--- |
| `str` | 追加先の [string](string.md) |
| `suffix` | 追加する接尾語 |

#### 戻り値
接尾語が追加された [string](string.md)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](string.md) に部分文字列が含まれているかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |
| `sub` | チェックする部分文字列 |

#### 戻り値
[string](string.md) に指定した部分文字列が含まれているかどうか

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](string.md) が特定の接尾語で終わるかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の [string](string.md) |
| `suffix` | チェックする接尾語 |

#### 戻り値
[string](string.md) がその接尾語で終わっているかどうか

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](string.md) 内で、指定した部分文字列のすべての出現箇所を探します

| 引数 |  |
| :--- | :--- |
| `str` | 部分文字列のすべての出現箇所を探す [string](string.md) |
| `sub` | 検索する部分文字列 |

#### 戻り値
部分文字列が含まれる [string](string.md) 内のインデックスの _リスト_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](string.md) が英数字のみで構成されているかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | チェックする [string](string.md) |

#### 戻り値
[string](string.md) が英数字のみなら真

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[string](string.md) がアルファベットのみで構成されているかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | チェックする [string](string.md) |

#### 戻り値
[string](string.md) がアルファベットのみなら真

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[string](string.md) が数値のみで構成されているかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | チェックする [string](string.md) |

#### 戻り値
[string](string.md) が数値のみな場合は真

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する [string](string.md) |

#### 戻り値
先頭の空白が削除された [string](string.md)

<h3 id="string-len"><code>string-len</code></h3>

[string](string.md) の長さを返します

| 引数 |  |
| :--- | :--- |
| `str` | 長さを調べる [string](string.md) |

#### 戻り値
[string](string.md) の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[string](string.md) を小文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | 小文字に変換する [string](string.md) |

#### 戻り値
小文字に変換された [string](string.md)

<h3 id="string-partition"><code>string-partition</code></h3>

[string](string.md) を _リスト_ の [string](string.md) にパーティション分割します

| 引数 |  |
| :--- | :--- |
| `str` | 分割する [string](string.md) |
| `sep` | 分割に使うセパレーター |

#### 戻り値
_リスト_ 型の [string](string.md)： セパレーターより前の [string](string.md)、セパレーター、セパレーターより後の [string](string.md)

<h3 id="string-prepend"><code>string-prepend</code></h3>

[string](string.md) の先頭に接頭語を追加します

| 引数 |  |
| :--- | :--- |
| `str` | 追加先の [string](string.md) |
| `prefix` | 追加する接頭語 |

#### 戻り値
接頭語が追加された [string](string.md)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する [string](string.md) |

#### 戻り値
末尾の空白が削除された [string](string.md)

<h3 id="string-replace"><code>string-replace</code></h3>

[string](string.md) 内で、すべての部分文字列の出現箇所を置換します

| 引数 |  |
| :--- | :--- |
| `str` | 置換する [string](string.md) |
| `sub` | 置換対象となる部分文字列 |
| `newSub` | 新しく置換する部分文字列 |

#### 戻り値
置換が反映された [string](string.md)

<h3 id="string-slice"><code>string-slice</code></h3>

開始・終了インデックスに基づき、[string](string.md) から部分文字列を取得します

| 引数 |  |
| :--- | :--- |
| `str` | スライス対象の [string](string.md) |
| `begin` | 部分文字列の開始インデックス |
| `end` | 部分文字列の終了インデックス |

#### 戻り値
抽出された部分文字列

<h3 id="string-split"><code>string-split</code></h3>

[string](string.md) をセパレーターで分割し、_リスト_ の [string](string.md) にします

| 引数 |  |
| :--- | :--- |
| `str` | 分割する [string](string.md) |
| `sep` | 分割に使うセパレーター |

#### 戻り値
分割された [string](string.md) の _リスト_

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](string.md) が特定の接頭語で始まるかどうか確認します

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の [string](string.md) |
| `prefix` | チェックする接頭語 |

#### 戻り値
[string](string.md) がその接頭語ではじまる場合は真

<h3 id="string-strip"><code>string-strip</code></h3>

[string](string.md) の両端の空白を削除します。

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する [string](string.md) |

#### 戻り値
両端の空白が削除された [string](string.md)

<h3 id="string-upper"><code>string-upper</code></h3>

[string](string.md) を大文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | 大文字に変換する [string](string.md) |

#### 戻り値
大文字に変換された [string](string.md)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

2つの [string](string.md) 間の Levenshtein 距離を計算します。

| 引数 |  |
| :--- | :--- |
| `str1` | 1つ目の [string](string.md) |
| `str2` | 2つ目の [string](string.md) |

#### 戻り値
2つの [string](string.md) 間の Levenshtein 距離


## リスト操作（List Ops）
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値 |
| `rhs` | 比較する2番目の値 |

#### 戻り値
2つの値が等しくない場合は真を返します。

<h3 id="string-add"><code>string-add</code></h3>

2つの [string](string.md) を連結します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [string](string.md) |
| `rhs` | 2番目の [string](string.md) |

#### 戻り値
連結された [string](string.md)

<h3 id="string-equal"><code>string-equal</code></h3>

2つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値 |
| `rhs` | 比較する2番目の値 |

#### 戻り値
2つの値が等しい場合は真を返します。

<h3 id="string-append"><code>string-append</code></h3>

[suffix](string.md) を [string](string.md) の末尾に追加します

| 引数 |  |
| :--- | :--- |
| `str` | 追加先の [string](string.md) |
| `suffix` | 追加する接尾語 |

#### 戻り値
接尾語が追加された [string](string.md)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](string.md) に部分文字列が含まれているかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する [string](string.md) |
| `sub` | チェックする部分文字列 |

#### 戻り値
[string](string.md) に指定した部分文字列が含まれているかどうか

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](string.md) が特定の接尾語で終わるかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の [string](string.md) |
| `suffix` | チェックする接尾語 |

#### 戻り値
[string](string.md) がその接尾語で終わっているかどうか

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](string.md) 内で、指定した部分文字列のすべての出現箇所を探します

| 引数 |  |
| :--- | :--- |
| `str` | 部分文字列のすべての出現箇所を探す [string](string.md) |
| `sub` | 検索する部分文字列 |

#### 戻り値
部分文字列が含まれる [string](string.md) 内のインデックスの _リスト_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](string.md) が英数字のみで構成されているかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | チェックする [string](string.md) |

#### 戻り値
[string](string.md) が英数字のみなら真

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[string](string.md) がアルファベットのみで構成されているかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | チェックする [string](string.md) |

#### 戻り値
[string](string.md) がアルファベットのみなら真

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[string](string.md) が数値のみで構成されているかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | チェックする [string](string.md) |

#### 戻り値
[string](string.md) が数値のみな場合は真

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する [string](string.md) |

#### 戻り値
先頭の空白が削除された [string](string.md)

<h3 id="string-len"><code>string-len</code></h3>

[string](string.md) の長さを返します

| 引数 |  |
| :--- | :--- |
| `str` | 長さを調べる [string](string.md) |

#### 戻り値
[string](string.md) の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[string](string.md) を小文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | 小文字に変換する [string](string.md) |

#### 戻り値
小文字に変換された [string](string.md)

<h3 id="string-partition"><code>string-partition</code></h3>

[string](string.md) を _リスト_ の [string](string.md) にパーティション分割します

| 引数 |  |
| :--- | :--- |
| `str` | 分割する [string](string.md) |
| `sep` | 分割に使うセパレーター |

#### 戻り値
_リスト_ 型の [string](string.md)： セパレーターより前の [string](string.md)、セパレーター、セパレーターより後の [string](string.md)

<h3 id="string-prepend"><code>string-prepend</code></h3>

[string](string.md) の先頭に接頭語を追加します

| 引数 |  |
| :--- | :--- |
| `str` | 追加先の [string](string.md) |
| `prefix` | 追加する接頭語 |

#### 戻り値
接頭語が追加された [string](string.md)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する [string](string.md) |

#### 戻り値
末尾の空白が削除された [string](string.md)

<h3 id="string-replace"><code>string-replace</code></h3>

[string](string.md) 内で、すべての部分文字列の出現箇所を置換します

| 引数 |  |
| :--- | :--- |
| `str` | 置換する [string](string.md) |
| `sub` | 置換対象となる部分文字列 |
| `newSub` | 新しく置換する部分文字列 |

#### 戻り値
置換が反映された [string](string.md)

<h3 id="string-slice"><code>string-slice</code></h3>

開始・終了インデックスに基づき、[string](string.md) から部分文字列を取得します

| 引数 |  |
| :--- | :--- |
| `str` | スライス対象の [string](string.md) |
| `begin` | 部分文字列の開始インデックス |
| `end` | 部分文字列の終了インデックス |

#### 戻り値
抽出された部分文字列

<h3 id="string-split"><code>string-split</code></h3>

[string](string.md) をセパレーターで分割し、_リスト_ の [string](string.md) にします

| 引数 |  |
| :--- | :--- |
| `str` | 分割する [string](string.md) |
| `sep` | 分割に使うセパレーター |

#### 戻り値
分割された [string](string.md) の _リスト_

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](string.md) が特定の接頭語で始まるかどうか確認します

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の [string](string.md) |
| `prefix` | チェックする接頭語 |

#### 戻り値
[string](string.md) がその接頭語ではじまる場合は真

<h3 id="string-strip"><code>string-strip</code></h3>

[string](string.md) の両端の空白を削除します。

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する [string](string.md) |

#### 戻り値
両端の空白が削除された [string](string.md)

<h3 id="string-upper"><code>string-upper</code></h3>

[string](string.md) を大文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | 大文字に変換する [string](string.md) |

#### 戻り値
大文字に変換された [string](string.md)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

2つの [string](string.md) 間の Levenshtein 距離を計算します。

| 引数 |  |
| :--- | :--- |
| `str1` | 1つ目の [string](string.md) |
| `str2` | 2つ目の [string](string.md) |

#### 戻り値
2つの [string](string.md) 間の Levenshtein 距離
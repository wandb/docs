# 文字列

## 連鎖操作
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

二つの値の不等を判断します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する二番目の値。 |

#### 返り値
二つの値が等しくないかどうか。

<h3 id="string-add"><code>string-add</code></h3>

二つの[文字列](https://docs.wandb.ai/ref/weave/string)を連結します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[文字列](https://docs.wandb.ai/ref/weave/string) |
| `rhs` | 二番目の[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 返り値
連結した[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-equal"><code>string-equal</code></h3>

二つの値の等価性を判断します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する二番目の値。 |

#### 返り値
二つの値が等しいかどうか。

<h3 id="string-append"><code>string-append</code></h3>
[string](https://docs.wandb.ai/ref/weave/string)に接尾辞を追加する

| 引数 |  |
| :--- | :--- |
| `str` | 接尾辞を追加する[string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 追加する接尾辞 |

#### 戻り値
接尾辞が追加された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が部分文字列を含むかどうかを確認する

| 引数 |  |
| :--- | :--- |
| `str` |  含むかどうかを確認する[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 確認する部分文字列 |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)が部分文字列を含むかどうか

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が接尾辞で終わるかどうかを確認する

| 引数 |  |
| :--- | :--- |
| `str` |  接尾辞で終わるかどうかを確認する[string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 確認する接尾辞 |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)が接尾辞で終わるかどうか

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)内の部分文字列のすべての出現箇所を見つける

| 引数 |  |
| :--- | :--- |
| `str` | 部分文字列の出現箇所を見つけるための[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 見つける部分文字列 |
#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)内の部分文字列のインデックスの_list_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)が英数字であるかどうかを確認する

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)が英数字であるかどうか

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)がアルファベット文字であるかどうかを確認する

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)がアルファベット文字であるかどうか

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)が数値であるかどうかを確認する

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)が数値であるかどうか

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白を削除する
| 引数 |  |
| :--- | :--- |
| `str` | ストリップする[文字列](https://docs.wandb.ai/ref/weave/string)。 |

#### 返り値
ストリップされた[文字列](https://docs.wandb.ai/ref/weave/string)。

<h3 id="string-len"><code>string-len</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)の長さを返します。

| 引数 |  |
| :--- | :--- |
| `str` | 長さを調べる[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 返り値
[文字列](https://docs.wandb.ai/ref/weave/string)の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)を小文字に変換します。

| 引数 |  |
| :--- | :--- |
| `str` | 小文字に変換する[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 返り値
小文字の[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-partition"><code>string-partition</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)を_リスト_に分割します。

| 引数 |  |
| :--- | :--- |
| `str` | 分割する[文字列](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 分割するセパレータ |

#### 返り値
[文字列](https://docs.wandb.ai/ref/weave/string)のリスト：セパレータの前の[文字列](https://docs.wandb.ai/ref/weave/string)、セパレータ、セパレータの後の[文字列](https://docs.wandb.ai/ref/weave/string)
<h3 id="string-prepend"><code>string-prepend</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) に接頭辞を付け足す

| 引数 |  |
| :--- | :--- |
| `str` | 接頭辞を付け足す [string](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | 付け足す接頭辞 |

#### 戻り値
接頭辞が付け足された [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白を削除する

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する [string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
空白が削除された [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-replace"><code>string-replace</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) 内の部分文字列のすべての出現を置換する

| 引数 |  |
| :--- | :--- |
| `str` | 内容を置換する [string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 置換する部分文字列 |
| `newSub` | 古い部分文字列と置換する新しい部分文字列 |

#### 戻り値
置換が行われた [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-slice"><code>string-slice</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) を、開始インデックスと終了インデックスに基づいて部分文字列にスライスする
| 引数 |  |
| :--- | :--- |
| `str` | スライスする[string](https://docs.wandb.ai/ref/weave/string) |
| `begin` | 部分文字列の開始インデックス |
| `end` | 部分文字列の終了インデックス |

#### 返り値
部分文字列

<h3 id="string-split"><code>string-split</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)を文字列のリストに分割します。

| 引数 |  |
| :--- | :--- |
| `str` | 分割する[string](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 分割するセパレータ |

#### 返り値
[string](https://docs.wandb.ai/ref/weave/string)のリスト

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)がプレフィクスで始まるかどうかをチェックします。

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[string](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | チェックするプレフィクス |

#### 返り値
[string](https://docs.wandb.ai/ref/weave/string)がプレフィクスで始まるかどうか

<h3 id="string-strip"><code>string-strip</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)の両端の空白を削除します。

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する[string](https://docs.wandb.ai/ref/weave/string) |
#### 戻り値
削除された[string](https://docs.wandb.ai/ref/weave/string)。

<h3 id="string-upper"><code>string-upper</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)を大文字に変換する。

| 引数 |  |
| :--- | :--- |
| `str` | 大文字に変換する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
大文字の[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

二つの[string](https://docs.wandb.ai/ref/weave/string)の間のレーベンシュタイン距離を計算する。

| 引数 |  |
| :--- | :--- |
| `str1` | 最初の[string](https://docs.wandb.ai/ref/weave/string)。 |
| `str2` | 二番目の[string](https://docs.wandb.ai/ref/weave/string)。 |

#### 戻り値
二つの[string](https://docs.wandb.ai/ref/weave/string)の間のレーベンシュタイン距離。

## リスト演算
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

二つの値の不等式を判断する。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する二番目の値。 |

#### 戻り値
二つの値が等しくないかどうか。
<h3 id="string-add"><code>string-add</code></h3>

二つの[文字列](https://docs.wandb.ai/ref/weave/string)を連結させる

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[文字列](https://docs.wandb.ai/ref/weave/string) |
| `rhs` | 二つ目の[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 返り値
連結された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-equal"><code>string-equal</code></h3>

二つの値の等価性を判断する。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値 |
| `rhs` | 比較する二つ目の値 |

#### 返り値
二つの値が等しいかどうか

<h3 id="string-append"><code>string-append</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)に接尾辞を追加する

| 引数 |  |
| :--- | :--- |
| `str` | 接尾辞を追加する[文字列](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 追加する接尾辞 |

#### 返り値
接尾辞が追加された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-contains"><code>string-contains</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)が部分文字列を含んでいるかどうか確認する

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | チェックする部分文字列 |

#### 返り値
[string](https://docs.wandb.ai/ref/weave/string)が部分文字列を含むかどうか

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が接尾辞で終わるかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | チェックする接尾辞 |

#### 返り値
[string](https://docs.wandb.ai/ref/weave/string)が接尾辞で終わるかどうか

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)内の部分文字列のすべての出現を見つけます

| 引数 |  |
| :--- | :--- |
| `str` | 部分文字列の出現を探す[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 見つける部分文字列 |

#### 返り値
[string](https://docs.wandb.ai/ref/weave/string)内の部分文字列のインデックスの_リスト_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が英数字かどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[string](https://docs.wandb.ai/ref/weave/string) |
#### 返り値
[文字列](https://docs.wandb.ai/ref/weave/string)が英数字であるかどうか

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)がアルファベットであるかどうかをチェックする

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 返り値
[文字列](https://docs.wandb.ai/ref/weave/string)がアルファベットであるかどうか

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)が数字であるかどうかをチェックする

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 返り値
[文字列](https://docs.wandb.ai/ref/weave/string)が数字であるかどうか

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白を削除する

| 引数 |  |
| :--- | :--- |
| `str` | 削除する [文字列](https://docs.wandb.ai/ref/weave/string) |

#### 返り値
削除された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-len"><code>string-len</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)の長さを返す
| 引数 |  |
| :--- | :--- |
| `str` | チェックする[string](https://docs.wandb.ai/ref/weave/string) |

#### 返り値
[string](https://docs.wandb.ai/ref/weave/string)の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)を小文字に変換する

| 引数 |  |
| :--- | :--- |
| `str` | 小文字に変換する[string](https://docs.wandb.ai/ref/weave/string) |

#### 返り値
小文字の[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-partition"><code>string-partition</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)を一覧表の[string](https://docs.wandb.ai/ref/weave/string)に分割する

| 引数 |  |
| :--- | :--- |
| `str` | 分割する[string](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 分割するための区切り文字 |

#### 返り値
[string](https://docs.wandb.ai/ref/weave/string)のリスト：区切り文字の前の[string](https://docs.wandb.ai/ref/weave/string)、区切り文字、区切り文字の後の[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-prepend"><code>string-prepend</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)に接頭辞を追加する

| 引数 |  |
| :--- | :--- |
| `str` | 接頭辞を追加する[string](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | 追加する接頭辞 |
#### 戻り値
接頭辞が付加された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白を削除

| 引数 |  |
| :--- | :--- |
| `str` | 空白を削除する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
空白が削除された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-replace"><code>string-replace</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)内の部分文字列のすべての出現を置き換える

| 引数 |  |
| :--- | :--- |
| `str` | 内容を置き換える[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 置き換える部分文字列 |
| `newSub` | 古い部分文字列と置き換える新しい部分文字列 |

#### 戻り値
置き換えが行われた[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-slice"><code>string-slice</code></h3>

始点と終点のインデックスに基づいて[string](https://docs.wandb.ai/ref/weave/string)を部分文字列にスライスする

| 引数 |  |
| :--- | :--- |
| `str` | スライスする[string](https://docs.wandb.ai/ref/weave/string) |
| `begin` | 部分文字列の開始インデックス |
| `end` | 部分文字列の終了インデックス |

#### 戻り値
部分文字列

<h3 id="string-split"><code>string-split</code></h3>
[string](https://docs.wandb.ai/ref/weave/string)を_list_の[string](https://docs.wandb.ai/ref/weave/string)に分割します

| 引数 |  |
| :--- | :--- |
| `str` | 分割する[string](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 分割するセパレーター |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)の_list_

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)がプレフィックスで始まるかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[string](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | チェックするプレフィックス |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)がプレフィックスで始まるかどうか

<h3 id="string-strip"><code>string-strip</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)の両端から空白を取り除きます

| 引数 |  |
| :--- | :--- |
| `str` | 空白を取り除く[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
空白を取り除いた[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-upper"><code>string-upper</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)を大文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | 大文字に変換する[string](https://docs.wandb.ai/ref/weave/string) |
#### 返り値
大文字にされた[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

二つの[string](https://docs.wandb.ai/ref/weave/string)間のレーベンシュタイン距離を計算します。

| 引数 |  |
| :--- | :--- |
| `str1` | 最初の[string](https://docs.wandb.ai/ref/weave/string)。 |
| `str2` | 二つ目の[string](https://docs.wandb.ai/ref/weave/string)。 |

#### 返り値
二つの[string](https://docs.wandb.ai/ref/weave/string)間のレーベンシュタイン距離。
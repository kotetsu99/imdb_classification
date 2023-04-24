import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys, os

# imdbデータセット読み込みの設定

# 出現数上位ワード数の設定(単語インデックスは出現数ランキング番号に対応)
vocab_size = 20000
# 各レビューで学習の対象とする最初のワード数の上限
maxlen = 200

# レビュー文最初の単語のインデックス番号
start_char = 1
# レビュー文の上位ワード数(vocab_size)に入っていない単語の
# インデックス番号はoov_charで置換
oov_char = 2
# 単語インデックスは以下の番号以降が振られる
# ここではstart_char,oov_charを除く3以降が割り振り
index_from = 3


# main関数
def main():

    # 学習済モデル読み込み。引数に学習済みモデルがなければ強制終了
    if not len(sys.argv)==2:
        print('使用法: python 02-imdb_test.py 学習済ファイル名')
        sys.exit()
    savefile = sys.argv[1]
    model = keras.models.load_model(savefile)

    # トークンデータ生成
    x_train, inverted_word_index = generate_token()

    # レビュー評価プログラムスタート(Ctrl + C が押下されるまで繰り返し)
    while True:

        try:
            # レビュー番号を入力
            input_key = input("\nレビュー番号を入力してください(1～25000の番号):")
            # レビュー番号を変数に退避
            d_index = int(input_key) - 1
            if not (0 <= d_index < 25000):
                raise ValueError

            # レビュー文の単語をインデックスから引いて組み立て。単語間はスペースでつなげる
            decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[d_index])
            # レビュー番号に該当するレビュー文を表示する
            print(decoded_sequence)

            # 上記のレビューについて、レビュースコア(最高100、最低0)の計算を行う、
            x_test = keras.preprocessing.sequence.pad_sequences([x_train[d_index]], maxlen=maxlen)
            res = model(x_test)
            rev_score = res.numpy()[0][0] * 100

            # レビュースコア50以上であれば高評価、50未満であれば低評価とする、
            if rev_score >= 50 :
                print("スコア="+ str(rev_score) + ":高評価をつけたレビューです。")
            else:
                print("スコア="+ str(rev_score) + ":低評価をつけたレビューです。")

        except ValueError:
            print("1～25,000の整数を入力してください")
            continue

        # 強制終了コマンドが入った場合終了
        except KeyboardInterrupt:
            print("\nCtrl + C により強制終了")
            break


# imdbレビューのトークンデータ生成
def generate_token():
    # imdbレビューから学習用データを取得.
    (x_train, _), _ = keras.datasets.imdb.load_data(
        start_char=start_char, oov_char=oov_char, index_from=index_from
    ,num_words=vocab_size)
    # 単語とそれに対応するインデックスを取得
    word_index = keras.datasets.imdb.get_word_index()
    # 単語-インデックスをインデックス-単語にしてマッピング。新しくディクショナリを作る
    # `index_from` を引数にしてインデックスを `x_train` のものに合わせる
    inverted_word_index = dict(
        (i + index_from, word) for (word, i) in word_index.items()
    )
    # 開始インデックス:開始文字 (1:[START])
    inverted_word_index[start_char] = "[START]"
    # 除外文字インデックス:除外文字 (2:[OOV])を追加
    inverted_word_index[oov_char] = "[OOV]"

    return x_train, inverted_word_index


# main関数実行
if __name__ == '__main__':
    main() 

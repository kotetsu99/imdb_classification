import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys, os

# 出現上位単語数の設定
vocab_size = 20000
# 各レビューで学習の対象とする最初の単語数の上限
maxlen = 200

# 各単語のEmbedding(埋め込み)ベクトル次元数
embed_dim = 32
# Multi-Head Attentionのヘッド数
num_heads = 2
# Feed Forward Networkのニューロン数
ff_dim = 32


# main関数 
def main():

    # モデル名取得。引数にモデル名がなければ、強制終了。
    if not len(sys.argv)==2:
        print('使用法: python 01-imdb_train.py モデルファイル名')
        sys.exit()
    savefile = sys.argv[1]

    # 学習済モデルがあれば、読み込み
    if os.path.exists(savefile):
        print('モデル再学習')
        model = keras.models.load_model(savefile)
    # 学習済モデルがなければ、新規作成
    else:
        print('モデル新規作成')
        # モデル作成
        model = transformer_model_maker()
        # モデルの学習設定（コンパイル）
        model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    # imdbレビュー学習用および検証用データをダウンロード
    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
    print(len(x_train), "学習用imdbレビュー件数")
    print(len(x_val), "検証用imdbレビュー件数")

    # 学習用および検証用データを整形。各レビューの文字数をmaxlenに合わせ。短いレビューは不足分を0で埋める
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

    # モデル学習 
    history = model.fit(
        x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val)
    )

    # モデル保存
    model.save(savefile)


# Transformerモデル組み立て
def transformer_model_maker():

    # 入力層の定義
    inputs = layers.Input(shape=(maxlen,))
    # Embedding層の初期化
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    # Attention層の初期化
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

    # 入力層、Embedding層、Attenstion層を順に組み立て
    x = embedding_layer(inputs)
    x = transformer_block(x)

    # プーリング層（データ圧縮）、ドロップアウト層、全結合層を追加
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    # 出力層。シグモイド関数により0～1の値を出力し、二値分類を行う
    outputs = layers.Dense(1, activation="sigmoid")(x)

    # 入力層と出力層を引数に指定してモデル作成
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model    


# Embedding層定義
class TokenAndPositionEmbedding(layers.Layer):

    # 初期化処理
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        # 単語(単語数=vocab_size)を埋め込みベクトル化(次元数=embed_dim)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        # 単語の位置情報を埋め込みベクトル化(次元数=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    # 呼び出し時の処理
    def call(self, x):
        # レビュー文の最大単語数
        maxlen = tf.shape(x)[-1]
        # 位置情報を[0,1,2...maxlen-1]までの整数列で表現
        positions = tf.range(start=0, limit=maxlen, delta=1)
        # 位置情報を埋め込みベクトル化
        positions = self.pos_emb(positions)
        # 単語を埋め込みベクトル化
        x = self.token_emb(x)
        # 単語と位置情報の埋め込みベクトルの和を取ることで、単語に位置情報を持たせる
        return x + positions


# Transformerブロック(Attention層、FFN層)を定義
class TransformerBlock(layers.Layer):

    # 初期化
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        # MultiHead Attention層を定義
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # FFN層を定義
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        # 正規化設定
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        # ドロップアウト層を定義
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    # 呼び出し時の処理
    def call(self, inputs, training):
        # Attention層の出力
        attn_output = self.att(inputs, inputs)
        # ドロップアウト層を通過
        attn_output = self.dropout1(attn_output, training=training)
        # Attention出力と入力を加算したものを正規化
        out1 = self.layernorm1(inputs + attn_output)
        # FFN層を通過
        ffn_output = self.ffn(out1)
        # ドロップアウト層を通過
        ffn_output = self.dropout2(ffn_output, training=training)
        # Attention層の出力、FFN層の出力を加算しさらに正規化 
        return self.layernorm2(out1 + ffn_output)


# main関数実行
if __name__ == '__main__':
    main()

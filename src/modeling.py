# =============================================================
# ライブラリ設定
# =============================================================

"""
畳み込みニューラルネットワークを作成
 - Sequential     ：モデルクラス
 - Conv2D         ：畳み込み層
 - MaxPooling2D   ：プーリング層
 - Activation     ：活性化関数
 - Dropout        ：ドロップアウト
 - Flatten        ：データを一次元に変換する処理
 - Dense          ：全結合層を定義
"""

import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
from config import *


# =============================================================
# モデル学習＆評価用関数
# =============================================================

class modeling():
    
    def __init__(self, **kwargs):
        self.X_train    = kwargs["X_train"]
        self.X_test     = kwargs["X_test"]
        self.y_train    = kwargs["y_train"]
        self.y_test     = kwargs["y_test"]
        self.model_path = model_path

    # モデル構築（学習・推論）
    def build_model(self, mode:str):

        # ======================================================
        # ニューラルネットワークの定義
        # ======================================================
        
        # モデルインスタンス
        model = Sequential()
        
        # 1層目 (畳み込み）
        model.add(Conv2D(32,(3,3),padding="same", input_shape=self.X_train.shape[1:]))
        model.add(Activation('relu'))
        
        # 2層目（Max Pooling)
        model.add(Conv2D(32,(3,3)))
        model.add(Activation('relu'))
        
        # 3層目 (Max Pooling)
        model.add(MaxPooling2D(pool_size=(2,2)))                     
        model.add(Dropout(0.3))                     
        
        # 4層目 (畳み込み)
        model.add(Conv2D(64,(3,3),padding="same"))                   
        model.add(Activation('relu'))
        
        # 5層目 (畳み込み)
        model.add(Conv2D(64,(3,3))) 
        model.add(Activation('relu'))
      
        # 6層目 (Max Pooling)
        model.add(MaxPooling2D(pool_size=(2,2)))
        # データを1次元化
        model.add(Flatten())
        
        # 7層目 (全結合層)
        model.add(Dense(512))                                       
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        
        # 出力層(softmaxで0〜1の確率を返す)
        model.add(Dense(3)) 
        model.add(Activation('softmax'))

        # ======================================================
        # 学習方法の定義
        # ======================================================
        
        # 最適化手法
        opt = RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07)

        # 損失関数
        model.compile(loss="categorical_crossentropy",  # 損失関数
                      optimizer=opt,                    # 最適化アルゴリズム
                      metrics=["accuracy"],             # 最適化指標
                    )
        
        # ======================================================
        # 処理プロセスの定義
        # ======================================================

        match mode:
            # 学習
            case "train":
                model.fit(self.X_train,  # 学習データ
                        self.y_train,    # 学習データ（クラスラベル）
                        batch_size=32,   # バッチサイズ
                        epochs=100,      # エポック数
                        )

                # 学習済モデル保存
                model.save(self.model_path)

            # 検証
            case "eval":
                # 学習済みモデル呼出
                model  = keras.models.load_model(self.model_path) 

            # 推論
            case "prediction":
                # 学習済みモデル呼出
                model = keras.models.load_model(self.model_path)

        # モデル出力
        return model
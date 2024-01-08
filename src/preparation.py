from PIL import Image
import os
import glob
import numpy as np
from sklearn import model_selection
from keras.src.utils.np_utils import to_categorical
from config import *

# 中間データ作成関数
def data_prep():

    # =======================================================================
    # データ準備
    # =======================================================================

    # 画像
    X = [] #学習
    Y = [] #ラベル

    for index, classlabel in enumerate(classes):

        # 画像データ(.jpg)全取得
        files = glob.glob(f'{input_dir}/{classlabel}/*.{extension}')
        
        # 写真を順番に取得
        for i, file in enumerate(files):
            
            # 画像を1つ読込
            image = Image.open(file)
            
            # 画像をRGB変換
            image = image.convert("RGB")
            
            # サイズを揃える
            image = image.resize((image_size, image_size))
            
            # 画像を数字の配列に変換
            data  = np.asarray(image)
            
            # Xに配列、Yにインデックスを追加
            X.append(data)
            Y.append(index)
            
    # X,YがリストなのでTensorflowが扱いやすいようnumpyの配列に変換
    X = np.array(X)
    Y = np.array(Y)

    # X,Yを学習用と評価用のデータに分類
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X, Y, test_size=0.2)

    # =======================================================================
    # データの正規化
    # =======================================================================

    # 特徴量：0~255の整数を0~1区間として正規化
    X_train = X_train.astype("float") / X_train.max()
    X_test  = X_test.astype("float") /  X_train.max()

    # クラスラベル：正解は1、他は0になるようワンホット表現に変換
    y_train = to_categorical(y_train, num_classes)
    y_test  = to_categorical(y_test, num_classes)

    print("=======X_train=========")
    print(f"shape: {X_train.shape}")
    print(f"data : {X_train[0]}")
    print("=======y_train=========")
    print(f"shape: {y_train.shape}")
    print(f"data : {y_train[0]}")

    # =======================================================================
    # 中間データ保存
    # =======================================================================

    # npz型式でデータを保存
    xy = (X_train,X_test,y_train,y_test)
    np.savez(intermediate_data,*xy)

    # 出力
    print("data prep: successfully create a intermediate data for modeling")

    return [X_train, X_test, y_train,y_test]


# 実行関数
def exe_prep_data(create_data=True):
    match create_data:
        # 中間データがない場合
        case True:
            print("creating a intermediate data for modeling.")
            loaded_data = data_prep()
            return {"X_train": loaded_data[0],
                    "X_test":  loaded_data[1],
                    "y_train": loaded_data[2], 
                    "y_test":  loaded_data[3],
                  }

        # 中間データ作成済みの場合
        case False:
            print("skip creating a intermediate data.")
            # npzファイルをロード
            loaded_data = np.load(intermediate_data)
            # npzファイルの各種データを定義付
            return {"X_train": loaded_data["arr_0"],
                    "X_test":  loaded_data["arr_1"],
                    "y_train": loaded_data["arr_2"], 
                    "y_test":  loaded_data["arr_3"],
                  }
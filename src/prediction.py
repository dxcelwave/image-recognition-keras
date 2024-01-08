from PIL import Image
from config import *
import numpy as np

def pred(predict_path, pred_model):

    # =======================================================
    # 推論用のデータ準備
    # =======================================================
    image = Image.open(predict_path)              # 画像読込
    image = image.convert("RGB")                  # RGB変換
    image = image.resize((image_size,image_size)) # リサイズ
    data  = np.asarray(image)                     # 数値の配列変換
    X     = []
    X.append(data)
    X     = np.array(X)

    # =======================================================
    # 推論
    # =======================================================

    # Xを与えて予測値を取得
    # データを1つしかしれてないので0番目の配列を最後に指定
    result = pred_model.predict([X], verbose=0)[0]
    print("===================")
    print(f"class: {classes}")
    print(f"pred: {result}")

    # 推定値 argmax()を指定しresultの配列にある推定値が一番高いインデックス抽出
    predicted = result.argmax()
    
    # 精度を[%]表記に変換
    percentage = int(result[predicted] *100)

    # 結果出力
    print(f"result: {classes[predicted]} ({percentage}[%])")
    print("===================")

    return {"result":classes[predicted], "percentage": percentage}
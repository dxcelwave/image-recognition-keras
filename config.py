import glob 
      
# =================================================
# モデル学習に必要な設定値を確認
# =================================================

# 学習・検証データの格納ディレクトリ
input_dir = "data/image"

# 画像データの拡張子
extension = "jpg"

# 画像サイズ(Default: 縦横 50px)
image_size = 50

# モデル学習用中間出力ファイル（npz）
intermediate_data = "data/" + "intermediate_data.npz"

# 学習済みモデルを保存するパス
model_path = "data/" + "model_cnn_h5"

# =================================================
# モデル学習に必要な設定値を確認（対応不要）
# =================================================

# 分類クラス
classes_dir = glob.glob(f'{input_dir}/*')
classes     = [label.replace(input_dir,"").replace("/","") for label in classes_dir]
print(f"class info: {classes}")

# 分類クラス数
num_classes = len(classes)
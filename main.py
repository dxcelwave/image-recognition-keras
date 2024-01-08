from src.preparation import exe_prep_data
from src.modeling    import modeling
from src.evaluation  import eval
from src.prediction  import pred
from config import *
import argparse

# ==================================================
# MAIN実行関数
# ==================================================
    
def main():
    # 引数のパーサー作成
    parser = argparse.ArgumentParser(description='modeling script with argparse')

    # 引数追加
    parser.add_argument('--mode',     type=str, help='decide to which flow to be executed.', required=True)
    parser.add_argument('--filepath', type=str, help='set the image file path to predict with learned model.')

    # 引数解析
    args = parser.parse_args()

    # 実行関数を定義
    match args.mode:
        
        # モデル学習用中間データを作成
        case "data_prep":
            load_data(create_data=True)
        
        # モデル学習
        case "train":
            create_model()

        # 検証
        case "eval":
            eval_model()
        
        # 推論
        case "prediction":
            predict_model(args.filepath)
        
        case _:
            print("could not execute any functions.")


# ==================================================
# 各種関数
# ==================================================

# ①データ作成
def load_data(create_data=True):
    return exe_prep_data(create_data)

# ②モデル学習    
def create_model():
    data = load_data(create_data=True)                           # データ読込
    learned_model = modeling(**data).build_model(mode="train")   # モデル学習

# ③モデル評価 
def eval_model():
    data          = load_data(create_data=False)                 # データ読込
    learned_model = modeling(**data).build_model(mode="eval")    # 学習済みモデル読込
    exe_eval      = eval(learned_model, **data)                  # 検証

# ④モデル推論
def predict_model(predict_path):
    data       = load_data(create_data=False)                    # データ読込
    pred_model = modeling(**data).build_model(mode="prediction") # 学習済みモデル読込
    prediction = pred(predict_path, pred_model)                  # 推論


if __name__ == '__main__':
    main()
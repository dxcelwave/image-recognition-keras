from PIL import Image
from config import *

def eval(learned_model, **kwargs):

    # 検証データ読込
    X_test             = kwargs["X_test"]
    y_test             = kwargs["y_test"]

    # 学習済みモデル読込
    model = learned_model

    # Kerasのevalueateメソッドで検証
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("================================")
    print(f"Test Loss: {round(scores[0],2)}")
    print(f"test Accuracy: {round(scores[1],3)*100} [%]")
    print("================================")

    return {"Test Loss": scores[0], "test Accuracy": scores[1]}
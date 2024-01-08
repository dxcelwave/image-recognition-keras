## Docker Build
docker build -f Dockerfile -t image-classify .

## コンテナ起動
docker run -it --rm --name image-classify -p 5005:5000 --mount type=bind,src="$(pwd)",dst=/work image-classify bash

## モデル学習用中間データ作成
python main.py --mode data_prep

## モデル学習
python main.py --mode train

## 検証
python main.py --mode eval

## 推論
python main.py --mode prediction --filepath data/predict/grape.jpg
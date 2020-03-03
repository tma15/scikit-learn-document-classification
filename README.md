# パーセプトロンに基づく文書分類器
## データ
```
wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
tar zxvf ldcc-20140209.tar.gz
```

`text` というディレクトリが解凍されます。

## 文書分類器の構築
```
python preprocess.py --dir text
python train.py
```

カレントディレクトリに `data` ディレクトリが作成され、以下のデータが配置されます。
```
data
├── label_encoder.pickle
├── model.pickle
├── test.pickle
├── train.pickle
└── vectorizer.pickle
```

## 学習済み文書分類器を用いた予測
```
python predict.py
```

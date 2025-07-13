# 🚣‍♂️ ローイングフォーム分析アプリ

## 📋 アプリの概要
MediaPipeを使用したリアルタイムローイングフォーム分析アプリです。動画から腰の角度を計算し、ドライブ/リカバリー比率などのメトリクスを分析します。

## 🎯 主要機能
- **リアルタイム角度分析**: 腰の角度のリアルタイム計算
- **骨格検出**: MediaPipeによる姿勢検出とオーバーレイ表示
- **角度グラフ**: インタラクティブな角度変化グラフ
- **フレーム制御**: 手動・自動フレーム進み
- **統計情報**: 最大・最小・平均角度の表示
- **フィードバック**: 理想値との比較分析

## 🚀 デプロイ方法

### Streamlit Cloud
1. GitHubにコードをプッシュ
2. [Streamlit Cloud](https://share.streamlit.io/)にアクセス
3. GitHubリポジトリを選択
4. メインファイル: `streamlit_app_full.py`を指定
5. デプロイ完了！

### ローカル実行
```bash
pip install -r requirements.txt
streamlit run streamlit_app_full.py
```

## 📁 ファイル構成
```
Rowingapp/
├── streamlit_app_full.py    # メインアプリ
├── requirements.txt         # 依存関係
├── model_data.json         # 理想値データ
├── .streamlit/config.toml  # Streamlit設定
└── README.md               # このファイル
```

## 🎬 使用方法
1. 動画ファイルをアップロード
2. 表示モードを選択（分析結果オーバーレイ/両方表示）
3. フレーム制御で動画を操作
4. 角度グラフと統計情報を確認
5. フィードバックを参考に改善

## 🔧 技術スタック
- **Streamlit**: Webアプリフレームワーク
- **MediaPipe**: 姿勢検出
- **OpenCV**: 動画処理
- **Plotly**: インタラクティブグラフ
- **NumPy**: 数値計算 
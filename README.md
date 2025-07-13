# ローイングフォーム分析アプリ

## 📋 概要
MediaPipeを使用したローイングフォームのリアルタイム分析アプリです。腰の角度、ドライブ/リカバリー比率、姿勢検出などの機能を提供します。

## 🚀 デプロイ方法

### Streamlit Cloudでの公開
1. GitHubにリポジトリをプッシュ
2. [Streamlit Cloud](https://share.streamlit.io/)にアクセス
3. GitHubアカウントでログイン
4. 「New app」をクリック
5. リポジトリを選択
6. メインファイルパス: `streamlit_app_full.py`
7. 依存関係ファイル: `requirements.txt`
8. 「Deploy!」をクリック

## 📦 依存関係
- streamlit>=1.28.0
- opencv-python-headless>=4.8.0
- mediapipe>=0.10.0
- numpy>=1.24.0
- scipy>=1.11.0
- plotly>=5.17.0
- pillow>=10.0.0

## 🎯 機能
- リアルタイム姿勢検出
- 腰の角度測定
- ドライブ/リカバリー比率分析
- インタラクティブグラフ表示
- 動画再生制御
- 理想値との比較フィードバック 
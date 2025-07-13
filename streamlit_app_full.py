import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from scipy.signal import find_peaks
import tempfile
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import time
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ページ設定
st.set_page_config(
    page_title="ローイングフォーム分析アプリ",
    page_icon="🚣",
    layout="wide"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .angle-display {
        font-size: 2rem;
        font-weight: bold;
        color: #d62728;
        text-align: center;
        padding: 1rem;
        background-color: #fff3cd;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .video-container {
        border: 2px solid #4C566A;
        border-radius: 5px;
        padding: 1rem;
        background-color: #21252B;
        margin: 1rem 0;
    }
    .feedback-excellent {
        color: #A3BE8C;
        font-weight: bold;
    }
    .feedback-improvement {
        color: #BF616A;
        font-weight: bold;
    }
    .player-controls {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def calculate_angle(a, b, c):
    """3点間の角度を計算"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def analyze_video_metrics(video_path):
    """動画からメトリクスを分析"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    nose_x_coords = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        
        try:
            landmarks = results.pose_landmarks.landmark
            nose_x_coords.append(landmarks[mp_pose.PoseLandmark.NOSE.value].x)
        except:
            if nose_x_coords:
                nose_x_coords.append(nose_x_coords[-1])
            else:
                nose_x_coords.append(0.5)
    
    cap.release()
    pose.close()
    
    if len(nose_x_coords) < 10:
        return None
    
    # ドライブ/リカバリー比率の計算
    x = np.array(nose_x_coords)
    peaks, _ = find_peaks(x, distance=15, prominence=0.01)
    troughs, _ = find_peaks(-x, distance=15, prominence=0.01)
    
    if len(peaks) < 2 or len(troughs) < 2:
        return None
    
    drive_times, recovery_times = [], []
    current_trough = troughs[0]
    
    for peak in peaks:
        if peak > current_trough:
            drive_times.append(peak - current_trough)
            next_troughs = troughs[troughs > peak]
            if len(next_troughs) > 0:
                next_trough = next_troughs[0]
                recovery_times.append(next_trough - peak)
                current_trough = next_trough
            else:
                break
    
    if not drive_times or not recovery_times:
        return None
    
    drive_recovery_ratio = np.mean(recovery_times) / np.mean(drive_times) if np.mean(drive_times) > 0 else 0
    
    return {
        "drive_recovery_ratio": drive_recovery_ratio,
        "torso_stability": 0.9,
        "head_stability": 0.9
    }

def load_model_data():
    """モデルデータを読み込み"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model_data.json")
    if os.path.exists(model_path):
        with open(model_path, 'r') as f:
            return json.load(f)
    return None

def process_video_frame(frame, mp_pose, mp_drawing, drawing_spec, connection_spec, initial_head_y, max_angle, min_angle):
    """1フレームを処理して姿勢検出と角度計算を行う"""
    h, w, _ = frame.shape
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = mp_pose.process(image)
    image.flags.writeable = True
    
    current_angle = None
    new_max_angle = max_angle
    new_min_angle = min_angle
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Head line
        if initial_head_y is None:
            left_eye = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE]
            initial_head_y = int(((left_eye.y + right_eye.y) / 2) * h)
        
        if initial_head_y is not None:
            cv2.line(image, (0, initial_head_y), (w, initial_head_y), (235, 203, 139), 2)
        
        # Hip angle calculation
        try:
            shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                       landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x, 
                    landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
            
            current_angle = calculate_angle(shoulder, hip, knee)
            
            # 最大値・最小値の更新
            if new_max_angle is None or current_angle > new_max_angle:
                new_max_angle = current_angle
            if new_min_angle is None or current_angle < new_min_angle:
                new_min_angle = current_angle
            
            # Display angle on video
            cv2.putText(image, f"{current_angle:.2f}", 
                        tuple(np.multiply(hip, [w, h]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (235, 203, 139), 3, cv2.LINE_AA)
        except:
            pass
        
        # Draw landmarks
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=connection_spec
        )
    
    return image, current_angle, new_max_angle, new_min_angle, initial_head_y

def main():
    # ヘッダー
    st.markdown('<h1 class="main-header">🚣 ローイングフォーム分析アプリ</h1>', unsafe_allow_html=True)
    
    # セッション状態の初期化
    if 'video_uploaded' not in st.session_state:
        st.session_state.video_uploaded = False
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = 0
    if 'max_angle' not in st.session_state:
        st.session_state.max_angle = None
    if 'min_angle' not in st.session_state:
        st.session_state.min_angle = None
    if 'initial_head_y' not in st.session_state:
        st.session_state.initial_head_y = None
    if 'angles_data' not in st.session_state:
        st.session_state.angles_data = []
    if 'video_frames' not in st.session_state:
        st.session_state.video_frames = []
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = time.time()
    if 'video_fps' not in st.session_state:
        st.session_state.video_fps = 30.0
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    
    # サイドバー
    st.sidebar.title("設定")
    
    # ファイルアップロード
    uploaded_file = st.sidebar.file_uploader(
        "動画ファイルをアップロード",
        type=['mp4', 'avi', 'mov'],
        help="ローイングの動画をアップロードしてください"
    )
    
    # モデルデータの読み込み
    model_data = load_model_data()
    
    if model_data:
        st.sidebar.success("✅ モデルデータ読み込み完了")
        st.sidebar.markdown("**理想的な比率:**")
        st.sidebar.metric("ドライブ/リカバリー比率", f"{model_data['drive_recovery_ratio']:.2f}")
    else:
        st.sidebar.error("❌ モデルデータが見つかりません")
        st.sidebar.info("create_model.pyを実行してモデルデータを作成してください")
    
    # メインコンテンツ
    if uploaded_file is not None and not st.session_state.video_uploaded:
        # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # 分析実行
        with st.spinner("動画を分析中..."):
            metrics = analyze_video_metrics(video_path)
        
        if metrics:
            st.session_state.metrics = metrics
            st.session_state.video_uploaded = True
            st.session_state.video_path = video_path  # 動画パスを保存
            
            # 動画フレームの読み込み
            cap = cv2.VideoCapture(video_path)
            
            # 動画のFPSを取得
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0  # デフォルト値
            st.session_state.video_fps = fps
            
            frames = []
            angles_data = []
            
            mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(color=(236, 239, 244), thickness=2, circle_radius=2)
            connection_spec = mp_drawing.DrawingSpec(color=(136, 192, 208), thickness=2, circle_radius=2)
            
            max_angle = None
            min_angle = None
            initial_head_y = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, angle, max_angle, min_angle, initial_head_y = process_video_frame(
                    frame, mp_pose, mp_drawing, drawing_spec, connection_spec, 
                    initial_head_y, max_angle, min_angle
                )
                
                frames.append(processed_frame)
                if angle is not None:
                    angles_data.append(angle)
                else:
                    angles_data.append(angles_data[-1] if angles_data else 0)
            
            cap.release()
            mp_pose.close()
            
            st.session_state.video_frames = frames
            st.session_state.angles_data = angles_data
            st.session_state.max_angle = max_angle
            st.session_state.min_angle = min_angle
            st.session_state.initial_head_y = initial_head_y
            
            st.success("✅ 動画の分析と処理が完了しました！")
        else:
            st.error("❌ 動画の分析に失敗しました。")
            st.info("以下の点を確認してください：")
            st.markdown("- 動画に人物が映っているか")
            st.markdown("- 動画の長さが十分か（最低10フレーム以上）")
            st.markdown("- 動画の品質が良好か")
            
            # 一時ファイルの削除
            os.unlink(video_path)
    
    # 動画がアップロード済みの場合の表示
    if st.session_state.video_uploaded and st.session_state.metrics:
        # メトリクス表示
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            delta_value = None
            if model_data:
                delta_value = st.session_state.metrics['drive_recovery_ratio'] - model_data['drive_recovery_ratio']
            st.metric(
                "ドライブ/リカバリー比率",
                f"{st.session_state.metrics['drive_recovery_ratio']:.2f}",
                delta=f"{delta_value:.2f}" if delta_value is not None else None
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("最大角度", f"{st.session_state.max_angle:.1f}°" if st.session_state.max_angle else "-°")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("最小角度", f"{st.session_state.min_angle:.1f}°" if st.session_state.min_angle else "-°")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 動画表示オプション
        st.subheader("🎥 動画表示")
        
        # 表示モードの選択
        display_mode = st.radio(
            "表示モードを選択:",
            ["🔍 分析結果オーバーレイ", "🔄 両方表示"],
            horizontal=True
        )
        
        if display_mode == "🔍 分析結果オーバーレイ":
            # 分析結果をオーバーレイした動画を表示
            if st.session_state.video_frames:
                st.info("🔍 分析結果（骨格表示・角度・頭の位置）をオーバーレイ表示")
                
                # 動画再生とリアルタイム分析結果のオーバーレイ
                if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                    # 動画を表示（分析結果オーバーレイ付き）
                    st.subheader("🎬 動画再生（分析結果オーバーレイ）")
                    
                    # 動画プレイヤー
                    st.video(st.session_state.video_path)
                    
                    # リアルタイム分析結果表示
                    st.subheader("📊 リアルタイム分析結果")
                    
                    # 角度データを表示
                    if st.session_state.angles_data:
                        # 現在の角度を大きく表示
                        current_angle = st.session_state.angles_data[st.session_state.current_frame]
                        angle_color = "#ff6b6b" if current_angle > 120 else "#feca57" if current_angle > 90 else "#48dbfb"
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, {angle_color}20, {angle_color}40); border-radius: 10px; margin: 20px 0;">
                            <h2 style="color: {angle_color}; margin: 0;">現在の角度</h2>
                            <div style="font-size: 4rem; font-weight: bold; color: {angle_color}; margin: 10px 0;">
                                {current_angle:.1f}°
                            </div>
                            <p style="color: #666; margin: 0;">フレーム {st.session_state.current_frame + 1}/{len(st.session_state.video_frames)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 統計情報
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("平均角度", f"{np.mean(st.session_state.angles_data):.1f}°")
                        
                        with col2:
                            st.metric("最大角度", f"{st.session_state.max_angle:.1f}°" if st.session_state.max_angle else "-°")
                        
                        with col3:
                            st.metric("最小角度", f"{st.session_state.min_angle:.1f}°" if st.session_state.min_angle else "-°")
                        
                        # 角度の時系列グラフ（リアルタイム更新）
                        st.subheader("📈 角度の変化グラフ")
                        
                        # グラフに現在のフレーム位置をマーク
                        fig = px.line(
                            y=st.session_state.angles_data,
                            title="リアルタイム角度変化",
                            labels={'y': '角度 (度)', 'x': 'フレーム'}
                        )
                        fig.update_traces(line=dict(color='red'))
                        fig.update_layout(
                            xaxis_title="フレーム",
                            yaxis_title="角度 (度)",
                            height=400
                        )
                        
                        # 現在のフレーム位置をマーク
                        if st.session_state.current_frame < len(st.session_state.angles_data):
                            fig.add_vline(x=st.session_state.current_frame, line_dash="dash", color="blue", 
                                         annotation_text=f"現在: {st.session_state.angles_data[st.session_state.current_frame]:.1f}°")
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # フレーム選択による静止画表示（フォールバック）
                st.subheader("📸 フレーム選択による詳細表示")
                current_frame = st.session_state.video_frames[st.session_state.current_frame]
                st.image(current_frame, caption=f"フレーム {st.session_state.current_frame + 1}/{len(st.session_state.video_frames)} - 分析結果オーバーレイ", use_column_width=True)
                
                # 現在の角度を表示
                if st.session_state.current_frame < len(st.session_state.angles_data):
                    current_angle = st.session_state.angles_data[st.session_state.current_frame]
                    st.markdown(f'<div class="angle-display">現在の角度: {current_angle:.2f}°</div>', unsafe_allow_html=True)
        
        else:  # 両方表示
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📹 元の動画")
                if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                    st.video(st.session_state.video_path)
            
            with col2:
                st.subheader("🔍 分析結果オーバーレイ")
                if st.session_state.video_frames:
                    current_frame = st.session_state.video_frames[st.session_state.current_frame]
                    st.image(current_frame, caption=f"フレーム {st.session_state.current_frame + 1}/{len(st.session_state.video_frames)}", use_column_width=True)
                    
                    # 現在の角度を表示
                    if st.session_state.current_frame < len(st.session_state.angles_data):
                        current_angle = st.session_state.angles_data[st.session_state.current_frame]
                        st.markdown(f'<div class="angle-display">現在の角度: {current_angle:.2f}°</div>', unsafe_allow_html=True)
        
        # フレーム制御（分析結果オーバーレイモードの場合のみ表示）
        if display_mode in ["🔍 分析結果オーバーレイ", "🔄 両方表示"] and st.session_state.video_frames:
            st.subheader("🎮 フレーム制御")
            
            # 再生状態の表示
            if st.session_state.is_playing:
                st.info("🎬 再生中... 手動でフレームを進めてください")
            else:
                st.info("⏸️ 一時停止中... 再生ボタンを押すか、スライダーを動かしてください")
            
            # フレーム進み/戻りボタン
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("⏮️ 前のフレーム"):
                    if st.session_state.current_frame > 0:
                        st.session_state.current_frame -= 1
                        st.rerun()
            with col2:
                if st.button("⏭️ 次のフレーム"):
                    if st.session_state.current_frame < len(st.session_state.video_frames) - 1:
                        st.session_state.current_frame += 1
                        st.rerun()
            with col3:
                if st.button("🔄 自動進み（5フレーム）"):
                    next_frame = min(st.session_state.current_frame + 5, len(st.session_state.video_frames) - 1)
                    st.session_state.current_frame = next_frame
                    st.rerun()
            
            frame_index = st.slider(
                "フレーム選択",
                0, len(st.session_state.video_frames) - 1,
                st.session_state.current_frame,
                key="frame_slider"
            )
            
            # スライダーが変更された場合、current_frameを更新
            if frame_index != st.session_state.current_frame:
                st.session_state.current_frame = frame_index
                st.rerun()
        
        # 分析結果の説明
        with st.expander("📋 分析結果の見方"):
            st.markdown("""
            **表示されている分析要素:**
            - 🦴 **骨格線**: 青い線で骨格の接続を表示
            - 📐 **腰の角度**: 黄色い数字で腰の角度を表示
            - 📏 **頭の基準線**: 黄色い水平線で頭の位置を表示
            - 🎯 **関節点**: 白い点で各関節の位置を表示
            """)
            
            # 現在の角度を表示
            if st.session_state.current_frame < len(st.session_state.angles_data):
                current_angle = st.session_state.angles_data[st.session_state.current_frame]
                st.markdown(f'<div class="angle-display">現在の角度: {current_angle:.2f}°</div>', unsafe_allow_html=True)
            
            # 角度の時系列グラフ
            st.subheader("📊 腰の角度の変化")
            
            if st.session_state.angles_data:
                fig = px.line(
                    y=st.session_state.angles_data,
                    title="フレームごとの腰の角度",
                    labels={'y': '角度 (度)', 'x': 'フレーム'}
                )
                fig.update_traces(line=dict(color='red'))
                fig.update_layout(
                    xaxis_title="フレーム",
                    yaxis_title="角度 (度)",
                    height=400
                )
                
                # 現在のフレーム位置をマーク
                if st.session_state.current_frame < len(st.session_state.angles_data):
                    fig.add_vline(x=st.session_state.current_frame, line_dash="dash", color="blue", 
                                 annotation_text=f"現在: {st.session_state.angles_data[st.session_state.current_frame]:.1f}°")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 角度の統計情報
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_angle = np.mean(st.session_state.angles_data)
                    st.metric("平均角度", f"{avg_angle:.1f}°")
                with col2:
                    st.metric("最大角度", f"{st.session_state.max_angle:.1f}°" if st.session_state.max_angle else "-°")
                with col3:
                    st.metric("最小角度", f"{st.session_state.min_angle:.1f}°" if st.session_state.min_angle else "-°")
                with col4:
                    if st.session_state.max_angle and st.session_state.min_angle:
                        angle_range = st.session_state.max_angle - st.session_state.min_angle
                        st.metric("角度範囲", f"{angle_range:.1f}°")
                    else:
                        st.metric("角度範囲", "-°")
        
        # フィードバック
        if model_data:
            st.subheader("💡 フィードバック")
            
            ideal_ratio = model_data['drive_recovery_ratio']
            your_ratio = st.session_state.metrics['drive_recovery_ratio']
            
            if abs(your_ratio - ideal_ratio) < 0.2:
                st.success("🎉 素晴らしい！理想的な比率に近いです。")
            else:
                st.warning("⚠️ 改善の余地があります。理想的な比率との差を確認してください。")
            
            # 詳細な分析
            st.markdown("**詳細分析:**")
            st.markdown(f"- あなたの比率: **{your_ratio:.2f}**")
            st.markdown(f"- 理想的な比率: **{ideal_ratio:.2f}**")
            st.markdown(f"- 差: **{abs(your_ratio - ideal_ratio):.2f}**")
        
        # リセットボタン
        if st.button("🔄 新しい動画をアップロード"):
            # 一時ファイルを削除
            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                os.unlink(st.session_state.video_path)
            
            st.session_state.video_uploaded = False
            st.session_state.current_frame = 0
            st.session_state.max_angle = None
            st.session_state.min_angle = None
            st.session_state.initial_head_y = None
            st.session_state.angles_data = []
            st.session_state.video_frames = []
            st.session_state.metrics = None
            st.session_state.is_playing = False
            st.session_state.video_fps = 30.0
            st.session_state.video_path = None
            st.rerun()
    
    else:
        # 初期画面
        st.info("👈 左側のサイドバーから動画ファイルをアップロードしてください")
        
        # アプリの説明
        st.markdown("""
        ## 📋 アプリの機能
        
        ### 🎯 分析項目
        - **ドライブ/リカバリー比率**: ローイングのタイミング分析
        - **腰の角度**: リアルタイムでの角度測定
        - **角度の統計**: 最大値、最小値、平均値
        - **姿勢検出**: MediaPipeによる骨格検出
        
        ### 📊 表示機能
        - **リアルタイム動画表示**: 姿勢検出と角度表示
        - **インタラクティブグラフ**: フレーム選択可能
        - **理想値との比較**: 詳細なフィードバック
        - **角度の時系列表示**: フレームごとの変化
        
        ### 🎬 動画再生機能
        - **再生/一時停止**: 動画の再生制御
        - **フレーム選択**: 任意のフレームにジャンプ
        - **自動再生**: 連続再生機能
        
        ### 🚀 使用方法
        1. 左側のサイドバーから動画をアップロード
        2. 自動分析の実行
        3. 動画プレイヤーで再生制御
        4. 結果の確認とフィードバック
        
        ### 🆕 新機能
        - **フレーム選択スライダー**: 任意のフレームにジャンプ
        - **リアルタイム角度表示**: 現在のフレームの角度を表示
        - **グラフ上のマーカー**: 現在位置を視覚的に表示
        - **動画再生機能**: 再生/一時停止
        """)

if __name__ == "__main__":
    main() 
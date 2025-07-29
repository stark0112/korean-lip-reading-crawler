import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class LipDetector:
    def __init__(self, debug_mode=False):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.debug_mode = debug_mode
        
    def extract_lip_landmarks(self, frame):
        """프레임에서 입술 랜드마크 추출"""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        
        # 입술 랜드마크 인덱스 (MediaPipe Face Mesh)
        lip_indices = [
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415
        ]
        
        lip_points = []
        for idx in lip_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            lip_points.append([x, y])
        
        # 단순한 검증: 최소 10개 포인트만 있으면 OK
        if len(lip_points) < 10:
            return None
            
        return lip_points

    def check_lip_in_video(self, video_path):
        """비디오에서 모든 프레임에 입술이 있는지 확인"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🔍 입술 검출 분석 중... (총 {total_frames}프레임, 매 프레임 확인)")
        
        lip_detected_count = 0
        total_checked = 0
        
        # 디버그용 이미지 저장
        debug_dir = Path("test_results")
        debug_dir.mkdir(exist_ok=True)
        debug_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 매 프레임 확인
            lip_points = self.extract_lip_landmarks(frame)
            
            # 디버그 모드에서 검출 결과 시각화 (처음 5개만)
            if self.debug_mode and debug_count < 5:
                debug_frame = frame.copy()
                if lip_points:
                    # 입술 랜드마크 그리기
                    for point in lip_points:
                        cv2.circle(debug_frame, (point[0], point[1]), 3, (0, 255, 0), -1)
                    cv2.putText(debug_frame, f"LIP DETECTED: {len(lip_points)} points", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(debug_frame, "NO LIP DETECTED", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                video_name = Path(video_path).stem
                debug_path = debug_dir / f"cleanup_lip_detection_{video_name}_{debug_count}.jpg"
                cv2.imwrite(str(debug_path), debug_frame)
                print(f"  📸 디버그 이미지 저장: {debug_path}")
                debug_count += 1
            
            if lip_points and len(lip_points) >= 10:
                lip_detected_count += 1
            total_checked += 1
            
            # 진행률 표시 (10프레임마다)
            if total_checked % 10 == 0:
                current_ratio = lip_detected_count / total_checked if total_checked > 0 else 0
                print(f"  📊 진행률: {total_checked}/{total_frames}프레임 | 검출률: {current_ratio*100:.1f}%")
            
            # 조기 종료: 한 프레임이라도 입술이 없으면
            if lip_detected_count < total_checked:
                print(f"  ⚠️ 입술 없는 프레임 발견으로 조기 종료: {total_checked}프레임에서 {lip_detected_count}개만 검출")
                break
        
        cap.release()
        
        # 95% 이상 검출되어야 함
        lip_ratio = lip_detected_count / total_checked if total_checked > 0 else 0
        print(f"  📊 최종 검출률: {lip_ratio*100:.1f}% ({lip_detected_count}/{total_checked})")
        return lip_ratio >= 0.95  # 95% 이상 검출

def load_download_history():
    """다운로드 히스토리 로드"""
    history_file = Path("download_history.json")
    if history_file.exists():
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def update_download_history(video_id, action, reason):
    """다운로드 히스토리 업데이트"""
    history = load_download_history()
    
    # 기존 항목 찾기
    for item in history:
        if item.get('video_id') == video_id:
            item['last_checked'] = datetime.now().isoformat()
            item['status'] = action
            item['reason'] = reason
            break
    
    # 히스토리 저장
    with open("download_history.json", 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def clean_invalid_videos():
    """입술이 안나오는 영상들 정리"""
    lip_videos_dir = Path("data/lip_videos/videos")  # videos 서브디렉토리 추가
    if not lip_videos_dir.exists():
        print("❌ data/lip_videos/videos 디렉토리가 없습니다.")
        return
    
    # 비디오 파일들 찾기
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(lip_videos_dir.glob(f"*{ext}"))
    
    print(f"🔍 총 {len(video_files)}개의 비디오 파일을 검사합니다...")
    
    detector = LipDetector(debug_mode=True)
    deleted_count = 0
    valid_count = 0
    
    for video_file in video_files:
        print(f"\n📹 검사 중: {video_file.name}")
        
        # 입술 검출 확인
        has_lips = detector.check_lip_in_video(video_file)
        
        if has_lips:
            print(f"✅ 유효한 영상: {video_file.name}")
            valid_count += 1
        else:
            print(f"❌ 삭제 대상: {video_file.name} (입술 검출 실패)")
            
            # 파일 삭제
            try:
                video_file.unlink()
                print(f"🗑️ 삭제됨: {video_file.name}")
                deleted_count += 1
                
                # 히스토리 업데이트 (파일명에서 video_id 추출 시도)
                video_name = video_file.stem
                update_download_history(video_name, "deleted", "lip_detection_failed")
                
            except Exception as e:
                print(f"⚠️ 삭제 실패: {e}")
    
    print(f"\n📊 정리 완료:")
    print(f"  - 유효한 영상: {valid_count}개")
    print(f"  - 삭제된 영상: {deleted_count}개")
    print(f"  - 총 검사: {len(video_files)}개")

if __name__ == "__main__":
    print("🧹 입술 검출 실패 영상 정리 시작...")
    clean_invalid_videos()
    print("✅ 정리 완료!") 
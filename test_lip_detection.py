import cv2
import numpy as np
import mediapipe as mp
import os
from pathlib import Path

class LipDetectionTester:
    """입술 검출 테스트 클래스"""
    
    def __init__(self):
        # MediaPipe Face Mesh 초기화 (더 관대한 설정)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,  # 단일 이미지용
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,  # 관대한 설정
            min_tracking_confidence=0.3
        )
        
        # 입술 랜드마크 인덱스
        self.outer_lip_indices = [
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
        ]
        
        self.inner_lip_indices = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318
        ]
        
        self.all_lip_indices = self.outer_lip_indices + self.inner_lip_indices
    
    def test_single_frame(self, frame):
        """단일 프레임에서 입술 검출 테스트"""
        try:
            # RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe로 얼굴 랜드마크 검출
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                print("❌ 얼굴을 찾을 수 없습니다!")
                return None, None, False
            
            print("✅ 얼굴 검출 성공!")
            
            # 첫 번째 얼굴 선택
            face_landmarks = results.multi_face_landmarks[0]
            
            # 입술 랜드마크 좌표 추출
            lip_points = []
            for idx in self.all_lip_indices:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                lip_points.append([x, y])
            
            lip_points = np.array(lip_points)
            
            # 입술 바운딩 박스 계산
            x_min, y_min = lip_points.min(axis=0)
            x_max, y_max = lip_points.max(axis=0)
            
            # 여백 추가
            margin = 10  # 작은 영역
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(frame.shape[1], x_max + margin)
            y_max = min(frame.shape[0], y_max + margin)
            
            # 정사각형 ROI 생성
            width = x_max - x_min
            height = y_max - y_min
            size = max(width, height)
            
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            half_size = size // 2
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(frame.shape[1], center_x + half_size)
            y2 = min(frame.shape[0], center_y + half_size)
            
            # ROI 추출
            lip_roi = frame[y1:y2, x1:x2]
            
            # 112x112로 리사이즈
            lip_region = cv2.resize(lip_roi, (112, 112))
            
            print(f"✅ 입술 검출 성공! 크기: {lip_region.shape}")
            return lip_region, lip_points, True
            
        except Exception as e:
            print(f"❌ 입술 추출 실패: {e}")
            return None, None, False
    
    def test_video_frame(self, video_path):
        """비디오의 첫 번째 프레임으로 테스트"""
        print(f"🎬 비디오 테스트: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ 비디오를 열 수 없습니다: {video_path}")
            return
        
        # 첫 번째 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다")
            cap.release()
            return
        
        print(f"📐 프레임 크기: {frame.shape}")
        cap.release()
        
        # 입술 검출 테스트
        lip_region, lip_points, success = self.test_single_frame(frame)
        
        if success:
            # 결과 저장
            os.makedirs("test_results", exist_ok=True)
            
            # 원본 프레임에 입술 영역 표시
            frame_with_roi = frame.copy()
            if lip_points is not None:
                # 입술 랜드마크 그리기
                for point in lip_points:
                    cv2.circle(frame_with_roi, tuple(point), 2, (0, 255, 0), -1)
                
                # 바운딩 박스 그리기
                x_min, y_min = lip_points.min(axis=0)
                x_max, y_max = lip_points.max(axis=0)
                margin = 10  # 더 넓은 영역
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(frame.shape[1], x_max + margin)
                y_max = min(frame.shape[0], y_max + margin)
                
                cv2.rectangle(frame_with_roi, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            
            # 결과 이미지 저장
            cv2.imwrite("test_results/original_frame.jpg", frame)
            cv2.imwrite("test_results/frame_with_roi.jpg", frame_with_roi)
            cv2.imwrite("test_results/lip_region.jpg", lip_region)
            
            print("✅ 테스트 결과가 'test_results' 폴더에 저장되었습니다!")
            print("   - original_frame.jpg: 원본 프레임")
            print("   - frame_with_roi.jpg: 입술 영역이 표시된 프레임")
            print("   - lip_region.jpg: 추출된 입술 영역")
        else:
            print("❌ 입술 검출에 실패했습니다")

def main():
    """메인 테스트 함수"""
    print("🔍 입술 검출 테스트 시작")
    print("=" * 50)
    
    # 테스터 생성
    tester = LipDetectionTester()
    
    # 비디오 파일 경로
    video_path = Path("data/unlabeled_videos/1.mp4")
    
    if not video_path.exists():
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
        return
    
    # 테스트 실행
    tester.test_video_frame(video_path)

if __name__ == "__main__":
    main() 
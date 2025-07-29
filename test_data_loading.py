# 데이터 로딩 테스트
import json
from pathlib import Path

def test_data_loading():
    """데이터 로딩 테스트"""
    print("🔍 데이터 로딩 테스트 시작!")
    
    # 경로 설정
    labeled_data_dir = Path("009.립리딩(입모양) 음성인식 데이터/01.데이터/1.Training/자소_라벨링데이터")
    source_data_dir = Path("009.립리딩(입모양) 음성인식 데이터/01.데이터/1.Training/원천데이터")
    
    tl48_label_dir = labeled_data_dir / "TL48" / "소음환경1" / "C(일반인)" / "M(남성)" / "M(남성)_24"
    ts48_source_dir = source_data_dir / "TS48" / "소음환경1" / "C(일반인)" / "M(남성)" / "M(남성)_24"
    
    print(f"📁 라벨 디렉토리: {tl48_label_dir}")
    print(f"📁 원천 디렉토리: {ts48_source_dir}")
    
    # 디렉토리 존재 확인
    print(f"✅ 라벨 디렉토리 존재: {tl48_label_dir.exists()}")
    print(f"✅ 원천 디렉토리 존재: {ts48_source_dir.exists()}")
    
    # JSON 파일 찾기
    json_files = list(tl48_label_dir.glob("*.json"))
    print(f"📊 JSON 파일 수: {len(json_files)}")
    
    if json_files:
        json_path = json_files[0]
        print(f"📄 첫 번째 JSON: {json_path.name}")
        
        # JSON 파일 읽기
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"📊 JSON 구조: {type(data)}")
            if isinstance(data, list):
                print(f"📊 리스트 길이: {len(data)}")
                if len(data) > 0:
                    print(f"📊 첫 번째 항목 키: {list(data[0].keys())}")
                    print(f"📊 첫 번째 항목: {data[0]}")
            else:
                print(f"📊 딕셔너리 키: {list(data.keys())}")
        
        except Exception as e:
            print(f"⚠️ JSON 읽기 실패: {e}")
    
    # 비디오 파일 찾기
    if json_files:
        json_path = json_files[0]
        video_filename = json_path.stem + ".mp4"
        video_path = ts48_source_dir / video_filename
        
        print(f"🎥 비디오 파일: {video_path}")
        print(f"✅ 비디오 파일 존재: {video_path.exists()}")
        
        if video_path.exists():
            print(f"📊 비디오 파일 크기: {video_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    test_data_loading() 
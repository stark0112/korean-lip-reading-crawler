# 자소 라벨링 데이터 테스트
import json
from pathlib import Path

def test_jaso_data():
    """자소 라벨링 데이터 구조 확인"""
    print("🔍 자소 라벨링 데이터 구조 확인!")
    
    # 경로 설정
    jaso_data_dir = Path("009.립리딩(입모양) 음성인식 데이터/01.데이터/1.Training/자소_라벨링데이터")
    jaso_file = jaso_data_dir / "TL48" / "소음환경1" / "C(일반인)" / "M(남성)" / "M(남성)_24" / "lip_J_1_M_05_C442_A_001_jaso.json"
    
    print(f"📄 파일: {jaso_file}")
    print(f"✅ 파일 존재: {jaso_file.exists()}")
    
    if jaso_file.exists():
        try:
            with open(jaso_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"📊 데이터 타입: {type(data)}")
            
            if isinstance(data, list):
                print(f"📊 리스트 길이: {len(data)}")
                if len(data) > 0:
                    print(f"📊 첫 번째 항목 키: {list(data[0].keys())}")
                    print(f"📊 첫 번째 항목: {data[0]}")
            else:
                print(f"📊 딕셔너리 키: {list(data.keys())}")
                if 'Sentence_info' in data:
                    print(f"📊 Sentence_info 타입: {type(data['Sentence_info'])}")
                    if isinstance(data['Sentence_info'], list) and len(data['Sentence_info']) > 0:
                        print(f"📊 첫 번째 문장: {data['Sentence_info'][0]}")
        
        except Exception as e:
            print(f"⚠️ 파일 읽기 실패: {e}")

if __name__ == "__main__":
    test_jaso_data() 
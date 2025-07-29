# 🎯 Korean Lip Reading AI Project

한국어 입술 읽기 AI 모델을 위한 YouTube 비디오 크롤링 및 데이터 수집 도구입니다.

## 📋 프로젝트 개요

이 프로젝트는 한국어 입술 읽기 AI 모델을 훈련하기 위한 고품질 데이터를 수집하는 도구입니다.

### 🎯 주요 기능
- **YouTube 비디오 크롤링**: 입술이 포함된 한국어 비디오 자동 수집
- **CC-BY 라이선스 필터링**: 저작권 문제 없는 영상만 수집
- **입술 검출**: MediaPipe를 사용한 정확한 입술 영역 검출
- **분리 다운로드**: 비디오와 오디오를 별도로 저장
- **자동 재시작**: 중간에 멈춰도 자동으로 재시작

## 🚀 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 기본 실행
```bash
# 단일 검색어로 크롤링
python lip_video_crawler_simple.py "입술"

# 대용량 수집 (모든 검색어)
python lip_video_crawler_simple.py --mass-collection

# 분리 다운로드 (비디오/오디오)
python lip_video_crawler_simple.py --mass-collection --separate-audio

# CC-BY 전용 검색
python lip_video_crawler_simple.py --mass-collection --separate-audio --cc-only-search
```

### 3. 자동 재시작 모드
```bash
# 중간에 멈춰도 자동 재시작
python auto_restart_crawler.py --mass-collection --separate-audio --cc-only-search
```

## 📁 프로젝트 구조

```
lips_command/
├── lip_video_crawler_simple.py    # 메인 크롤러
├── auto_restart_crawler.py        # 자동 재시작 래퍼
├── clean_invalid_files.py         # 무효한 파일 정리
├── train.py                       # AI 모델 훈련 코드
├── test_*.py                      # 테스트 스크립트들
├── data/
│   └── lip_videos/
│       ├── videos/                # 다운로드된 비디오
│       └── audio/                 # 다운로드된 오디오
├── models/                        # 훈련된 모델들
└── requirements.txt               # 의존성 목록
```

## ⚙️ 설정 옵션

### 크롤러 옵션
- `--mass-collection`: 대용량 수집 모드
- `--separate-audio`: 비디오/오디오 분리 다운로드
- `--cc-only-search`: CC-BY 전용 검색
- `--debug`: 디버그 모드
- `--max-videos-per-query`: 쿼리당 최대 비디오 수

### 자동 재시작 옵션
- `--max-restarts`: 최대 재시작 횟수 (기본: 10)
- `--restart-delay`: 재시작 대기 시간 (기본: 30초)

## 🔧 주요 기능

### 1. 입술 검출
- **MediaPipe Face Mesh** 사용
- **95% 이상 프레임**에서 입술 검출 시 영상 포함
- **실시간 미리보기** 검출로 효율성 향상

### 2. 라이선스 필터링
- **CC-BY 라이선스** 영상만 수집
- **키워드 기반** 라이선스 확인
- **YouTube API** 활용한 정확한 필터링

### 3. 품질 관리
- **1080p 이상** 화질만 수집
- **입술 검출 실패** 영상 자동 삭제
- **중복 다운로드** 방지

## 📊 수집된 데이터

### 검색어 목록
- 뉴스 관련: "뉴스 발음", "뉴스 앵커", "아나운서 발음"
- 기본 입술: "입술", "입모양", "입술 움직임"
- 한국어 발음: "한국어 발음", "발음 교정"
- 상황별: "강의 발음", "인터뷰 발음"

### 데이터 품질
- **화질**: 1080p 이상
- **라이선스**: CC-BY만
- **입술 검출**: 95% 이상 프레임
- **언어**: 한국어 중심

## 🤝 협업 가이드

### 1. 이슈 보고
- [Issues](https://github.com/your-username/your-repo/issues) 탭에서 버그나 기능 요청
- 명확한 제목과 상세한 설명 포함

### 2. 풀 리퀘스트
- 새로운 기능이나 버그 수정 시
- 코드 리뷰 후 병합

### 3. 브랜치 전략
- `main`: 안정적인 메인 브랜치
- `develop`: 개발 브랜치
- `feature/기능명`: 새로운 기능 개발

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👥 기여자

- [Your Name](https://github.com/your-username)

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 [Issues](https://github.com/your-username/your-repo/issues)를 통해 연락해주세요. 
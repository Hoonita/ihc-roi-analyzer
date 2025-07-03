# IHC ROI Analyzer

**IHC ROI Analyzer**는 면역조직화학(IHC) 이미지에서 관심영역(ROI)을 직접 선택하고 분석할 수 있는 GUI 기반 데스크탑 애플리케이션입니다. 사용자는 이미지에서 원하는 영역을 마우스로 선택하고, DAB/H&E 염색의 색상 분리, 핵 검출, 염색 강도 정량화, H-score 계산 등을 자동으로 수행할 수 있습니다.

## 주요 기능

- 마우스를 이용한 ROI 생성, 이동, 삭제, 복제
- 다양한 염색 유형(DAB, H&E, Sirius Red 등) 지원
- Color Deconvolution을 통한 염색 성분 분리
- Hematoxylin 기반 핵 검출 및 개수 계산
- DAB 영역의 positive area 백분율, 평균 강도, H-score 계산
- 분석 결과를 CSV 및 텍스트 파일로 저장
- ROI 템플릿 저장 및 불러오기 지원
- 확대/축소, 격자 표시, 분석 오버레이 표시 기능

## 설치 방법

1. Python 3.8 이상이 설치되어 있어야 합니다.
2. 다음 명령어로 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
python ROI_IHC.py
```

## 파일 구성

```
ihc-roi-analyzer/
├── ROI_IHC.py           # 메인 애플리케이션 소스 코드
├── requirements.txt     # 필수 패키지 목록
└── README.md            # 설명 파일 (본 문서)
```

## 분석 결과 저장 기능

- **Export Analysis Results**: 각 ROI에 대한 분석 결과를 CSV 파일로 저장
- **Save Analysis Report**: 전체 분석 요약 및 ROI별 상세 결과를 텍스트 파일로 저장
- **Export ROI Images**: 각 ROI를 개별 이미지로 잘라서 저장

## 단축키 안내

- `F5`: 전체 ROI에 대한 IHC 분석 실행
- `Delete` 또는 `Backspace`: 선택된 ROI 삭제
- `Ctrl + Z / Ctrl + Y`: Undo / Redo
- `Ctrl + D`: ROI 복제

## 참고 사항

- 프로그램은 `tkinter` 기반 GUI로 작동하므로 터미널이 아닌 데스크탑 환경에서 실행해야 합니다.
- 분석 대상은 DAB 또는 H&E 염색된 조직 이미지이며, JPG/PNG/TIF 등의 형식을 지원합니다.

## 문의 및 기여

이 프로젝트는 실험실 또는 연구 환경에서 IHC 정량분석을 지원하기 위해 제작되었습니다. 개선 아이디어, 버그 리포트, 기능 요청은 GitHub 이슈 또는 Pull Request를 통해 자유롭게 참여해 주세요.

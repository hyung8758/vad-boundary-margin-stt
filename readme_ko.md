# vad-boundary-margin-stt

KCC 2026 논문 **Analysis of the Effects of VAD Boundary Margin Adjustment on STT Performance** 에서 사용한 실험 코드를 정리한 저장소입니다.

저자: 양형원, 김태호, 최윤수, 손수한

## 개요
이 저장소에는 KCC 2026 논문에서 사용한 VAD 기반 STT 경계 마진 실험 코드가 포함되어 있습니다.

논문에서는 음성 구간의 앞쪽/뒤쪽 경계 마진을 조절했을 때 STT 성능이 어떻게 달라지는지 분석하고, 이 결과를 실제 VAD 기반 디코딩에 적용할 수 있는지 확인합니다.

코드는 논문에서 수행한 두 단계의 실험으로 구성되어 있습니다.

- **Step 1: Controlled Boundary Margin Analysis**
  - MFA TextGrid의 강제 정렬 경계를 사용합니다.
  - 각 발화의 앞쪽/뒤쪽 경계 마진을 바꿔 가며 실험합니다.
  - CER과 경계 오류에 민감한 지표를 계산합니다.
  - 논문 Figure 1, Figure 2에 사용한 표와 그림을 생성합니다.

- **Step 2: Practical VAD-Based Decoding**
  - Silero VAD로 음성 구간을 검출한 뒤 STT를 수행합니다.
  - 원본 발화 baseline, 기본 VAD, FA-informed VAD 조건을 비교합니다.
  - Step 1에서 얻은 마진 설정이 실제 VAD 기반 디코딩에도 도움이 되는지 평가합니다.

## 논문 핵심 결과
논문의 주요 결과는 다음과 같습니다.

1. 발화 앞부분이 잘릴 때가 뒷부분이 잘릴 때보다 STT 성능에 더 큰 영향을 줍니다.
2. 음성 경계를 너무 딱 맞게 자르는 것보다 약간의 여유를 두는 편이 안정적입니다.
3. 적절한 양의 마진을 추가하면 과도한 경계 절단으로 인한 성능 저하를 줄일 수 있습니다.
4. Step 1에서 찾은 마진 설정을 실제 VAD 디코딩에 적용했을 때, 기본 VAD 설정보다 CER가 개선되었습니다.

실제 VAD 기반 실험에서 얻은 상대 CER 개선율은 다음과 같습니다.
- **clean: 18.1%**
- **other: 14.9%**

## 저장소 구조

```text
conf/       설정 파일
scripts/    데이터 준비, 디코딩, 결과 요약용 실행 스크립트
src/        핵심 소스 코드
artifacts/  중간 생성 파일 경로 (공개 저장소에서는 비워 둠)
exp/        실험 결과 경로 (공개 저장소에서는 비워 둠)
logs/       실행 로그 경로 (공개 저장소에서는 비워 둠)
data/       데이터셋 루트 경로 (공개 저장소에서는 비워 둠)
```

## 설치

```bash
pip install -r requirements.txt
```

이 저장소는 다음을 전제로 합니다.
- LibriSpeech 오디오는 `data/LibriSpeech` 아래에 준비합니다.
- MFA / forced-alignment TextGrid 파일은 `data/LibriSpeech/textgrid` 아래에 준비합니다.
- MFA는 별도의 conda 환경에서 설치해 사용합니다.

## 데이터 준비
논문 실험에서는 LibriSpeech 평가 split만 사용했습니다.

먼저 LibriSpeech 평가 split을 다운로드합니다.

```bash
bash scripts/data/download_librispeech.sh
```

이 스크립트는 OpenSLR SLR12에서 아래 split을 다운로드하고 압축을 해제합니다.
- `dev-clean`
- `dev-other`
- `test-clean`
- `test-other`

강제 정렬은 MFA 전용 conda 환경에서 진행하는 것을 권장합니다.

```bash
conda create -n mfa-aligner -c conda-forge montreal-forced-aligner pyyaml
conda activate mfa-aligner
```

MFA에서 사용할 pretrained dictionary와 acoustic model을 다운로드합니다.

```bash
mfa model download dictionary english_us_mfa
mfa model download acoustic english_mfa
```

이후 LibriSpeech를 MFA 입력 형식으로 준비하고 TextGrid 파일을 생성합니다.

```bash
python scripts/data/run_mfa_librispeech.py --jobs 8
```

기본값으로 `conf/base.yaml`에 적힌 네 split(`dev-clean`, `dev-other`, `test-clean`, `test-other`)을 모두 처리합니다.
스크립트 내부에서 MFA 임시 작업 파일을 정리하는 옵션을 기본으로 사용하므로 별도 옵션을 지정할 필요는 없습니다.
생성된 TextGrid 파일은 `data/LibriSpeech/textgrid/<split>` 아래에 저장됩니다.
MFA가 끝난 뒤에는 다시 본 실험용 Python 환경으로 돌아와 STT pipeline을 실행하면 됩니다.

데이터 준비가 끝나면 아래와 같은 구조가 됩니다.

```text
data/
  LibriSpeech/
    dev-clean/
    dev-other/
    test-clean/
    test-other/
    textgrid/
      dev-clean/
      dev-other/
      test-clean/
      test-other/
```

논문에서 사용한 STT 모델은 `wav2vec2-base-960h`입니다.

## 설정
기본 설정은 `conf/base.yaml`에 정의되어 있습니다.

중요한 항목은 다음과 같습니다.
- `datasets.librispeech.data_path`
- `datasets.librispeech.textgrid_path`
- `oracle.leading_ms`
- `oracle.trailing_ms`
- `silero_vad`
- `vad_policy`

Step 2에서 사용한 FA-informed VAD 마진 설정은 다음과 같습니다.
- clean: leading 400 ms / trailing 400 ms
- other: leading 500 ms / trailing 500 ms

## 논문 실험 실행

### Step 1: Controlled Boundary Margin Analysis
이 단계는 다음을 수행합니다.
1. manifest 생성
2. oracle margin variant 생성
3. baseline 발화 디코딩
4. oracle variant 디코딩
5. 발화 단위/요약 CSV 생성
6. 논문용 표와 그림 생성

실행:

```bash
bash scripts/run_experiment_pipeline.sh step1 librispeech
```

### Step 2: Practical VAD-Based Decoding
이 단계는 다음을 수행합니다.
1. Step 1에서 생성한 manifest 재사용
2. Step 1의 baseline decode 결과 재사용
3. `vad_default` 디코딩
4. `vad_fa_informed` 디코딩
5. 발화 단위/요약 CSV 생성
6. 논문용 표 생성

실행:

```bash
bash scripts/run_experiment_pipeline.sh step2 librispeech
```

### 전체 논문 실험 실행

```bash
bash scripts/run_experiment_pipeline.sh all_steps librispeech
```

## 출력 결과
생성된 결과는 `exp/step1`과 `exp/step2` 아래에 저장됩니다.

### Step 1 출력
- 디코딩 결과: `exp/step1/decodes/`
- 평가 결과: `exp/step1/eval/`
- 분석 표/그림: `exp/step1/analysis/`

자주 사용하는 Step 1 산출물:
- `exp/step1/analysis/tables/leading_trailing_cer.csv`
- `exp/step1/analysis/tables/first_and_last_word_errors.csv`
- `exp/step1/analysis/figures/leading_trailing_cer.pdf`
- `exp/step1/analysis/figures/first_and_last_word_errors.pdf`

### Step 2 출력
- 디코딩 결과: `exp/step2/decodes/`
- 평가 결과: `exp/step2/eval/`
- 분석 표: `exp/step2/analysis/`

자주 사용하는 Step 2 산출물:
- `exp/step2/analysis/tables/cer_boundary_changed_table.csv`

## 인용
이 저장소를 사용하신다면 아래 논문을 인용해 주세요.

```bibtex
@inproceedings{yang2026kcc,
  title={Analysis of the Effects of VAD Boundary Margin Adjustment on STT Performance},
  author={Yang, Hyungwon and Kim, Taeho and Choi, Younsou and Son, Suhan},
  booktitle={Proceedings of KCC},
  year={2026}
}
```

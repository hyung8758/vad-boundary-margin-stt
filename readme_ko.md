# vad-boundary-margin-stt

KCC 2026 논문 **Analysis of the Effects of VAD Boundary Margin Adjustment on STT Performance** 에 사용한 코드 공개 저장소입니다.

저자: 양형원, 김태호, 최윤수, 손수한

## 개요
이 저장소는 KCC 2026 논문에서 사용한 VAD 기반 STT 경계 마진 실험 코드를 제공합니다.

논문은 leading / trailing boundary margin이 STT 성능에 어떤 영향을 주는지 분석하고, 그 결과를 실제 VAD 기반 디코딩 개선에 적용할 수 있는지 평가합니다.

코드는 논문에서 사용한 두 단계의 실험을 지원합니다.

- **Step 1: Controlled Boundary Margin Analysis**
  - MFA TextGrid 강제정렬 경계를 사용
  - utterance 주변 leading / trailing margin sweep 수행
  - CER 및 boundary-sensitive error metric 평가
  - 논문에 사용된 표와 그림 생성

- **Step 2: Practical VAD-Based Decoding**
  - Silero VAD를 사용한 실제 segmentation 실험
  - original baseline, default VAD, FA-informed VAD 조건 비교
  - Step 1에서 얻은 margin 설정이 실제 디코딩 개선에 도움이 되는지 평가

## 논문 핵심 결과
논문에서는 다음 결과를 보고합니다.

1. leading cut은 trailing cut보다 더 해롭습니다.
2. exact boundary trimming은 안정적인 STT 디코딩에 불리합니다.
3. 적절한 positive margin은 aggressive trimming으로 인한 성능 저하를 줄여줍니다.
4. Step 1에서 얻은 margin 설정을 Step 2에 적용하면 default VAD 대비 CER가 개선됩니다.

실제 VAD 기반 실험에서 보고된 상대 CER 개선은 다음과 같습니다.
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
- LibriSpeech 오디오가 이미 준비되어 있음
- MFA / forced-alignment TextGrid 파일이 이미 준비되어 있음
- MFA 자체는 이 저장소 안에서 설치하거나 실행하지 않음

## 데이터 준비
논문 실험은 LibriSpeech 평가 split만 사용합니다.

예상 데이터 구조는 아래와 같습니다.

```text
data/
  LibriSpeech/
    dev-clean/
    dev-other/
    test-clean/
    test-other/
    textgrid/
```

논문에서 사용한 STT 백엔드는 `wav2vec2-base-960h`입니다.

## 설정
기본 설정은 `conf/base.yaml`에 정의되어 있습니다.

중요한 항목은 다음과 같습니다.
- `datasets.librispeech.data_path`
- `datasets.librispeech.textgrid_path`
- `oracle.leading_ms`
- `oracle.trailing_ms`
- `silero_vad`
- `vad_policy`

논문 Step 2에서 사용한 FA-informed VAD 정책은 다음과 같습니다.
- clean: leading 400 ms / trailing 400 ms
- other: leading 500 ms / trailing 500 ms

## 논문 실험 실행

### Step 1: Controlled Boundary Margin Analysis
이 단계는 다음을 수행합니다.
1. manifest 생성
2. oracle margin variant 생성
3. baseline utterance 디코딩
4. oracle variant 디코딩
5. per-utterance / summary CSV 생성
6. 분석용 표와 그림 생성

실행:

```bash
bash scripts/run_experiment_pipeline.sh step1 librispeech
```

### Step 2: Practical VAD-Based Decoding
이 단계는 다음을 수행합니다.
1. Step 1에서 생성한 manifest 재사용
2. Step 1 baseline decode 재사용
3. `vad_default` 디코딩
4. `vad_fa_informed` 디코딩
5. per-utterance / summary CSV 생성
6. 분석용 표와 그림 생성

실행:

```bash
bash scripts/run_experiment_pipeline.sh step2 librispeech
```

### 전체 논문 실험 실행

```bash
bash scripts/run_experiment_pipeline.sh all_steps librispeech
```

## 출력 결과
생성 결과는 `exp/step1`과 `exp/step2` 아래에 저장됩니다.

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
이 저장소를 사용한다면 아래 논문을 인용해 주세요.

```bibtex
@inproceedings{yang2026kcc,
  title={Analysis of the Effects of VAD Boundary Margin Adjustment on STT Performance},
  author={Yang, Hyungwon and Kim, Taeho and Choi, Younsou and Son, Suhan},
  booktitle={Proceedings of KCC},
  year={2026}
}
```

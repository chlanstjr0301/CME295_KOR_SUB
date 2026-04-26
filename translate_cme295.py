import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
ENG_DIR = ROOT / "eng"
KOR_DIR = ROOT / "kor"
BILINGUAL_DIR = ROOT / "kor + eng"
CACHE_DIR = ROOT / ".translation_cache"

MODEL = "gpt-4.1"
BATCH_SIZE = 20
CONTEXT_SIZE = 6


@dataclass
class SRTBlock:
    index: str
    timing: str
    lines: List[str]


SRT_BLOCK_RE = re.compile(
    r"(?P<index>\d+)\s*\n"
    r"(?P<timing>\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3})\s*\n"
    r"(?P<text>.*?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)


SYSTEM_PROMPT = """
너는 Stanford CME295 Transformers & LLMs 강의 자막을 한국어로 번역하는 전문 번역가다.

목표:
- 자동자막을 단순 직역하지 말고, 강의 맥락에 맞게 자연스럽고 정확한 한국어 강의 자막으로 번역한다.
- SRT 번호와 타임코드는 절대 바꾸지 않는다.
- 각 자막 블록은 원래 의미를 보존하되, 한국어로 읽었을 때 강의 자막처럼 자연스러워야 한다.
- technical term은 무리하게 번역하지 않는다.
- 사용자가 이전에 만족한 스타일은 다음과 같다:
  "좋습니다.", "여러분 안녕하세요. CME 295에 오신 것을 환영합니다.",
  "이 강의는 트랜스포머와 대규모 언어모델을 다룹니다."
  이런 식의 자연스러운 강의체를 유지한다.

용어 번역 원칙:
- Transformer → 트랜스포머
- Attention → 어텐션
- Self-attention → 셀프 어텐션
- Multi-head attention → 멀티헤드 어텐션
- Embedding → 임베딩
- Positional encoding → 위치 인코딩
- Token → 토큰
- Tokenization → 토큰화
- Vocabulary → 어휘집
- Pretraining → 사전학습
- Fine-tuning → 파인튜닝
- Inference → 추론
- Training → 학습
- Model → 모델
- Layer → 레이어
- Hidden state → 은닉 상태
- Query/Key/Value → 쿼리/키/값 또는 Q/K/V
- Language model → 언어모델
- Large language model / LLM → 대규모 언어모델 / LLM
- Neural network → 신경망
- Machine learning → 머신러닝
- Natural language processing / NLP → 자연어처리 / NLP
- Loss → 손실
- Gradient descent → 경사하강법
- Backpropagation → 역전파
- Parameter → 파라미터
- Weight → 가중치
- Bias → 편향
- Logits → 로짓
- Probability distribution → 확률분포
- Softmax → 소프트맥스
- Encoder → 인코더
- Decoder → 디코더

스타일:
- 강의체로 자연스럽게 번역한다.
- 과도한 존댓말이나 설명 추가는 피한다.
- "we're going to"는 문맥상 "이제 ~를 살펴보겠습니다", "~를 다룰 것입니다"로 옮긴다.
- "basically", "kind of", "sort of", "you know" 같은 filler는 필요하면 생략한다.
- 자동자막 오류로 문장이 끊겨 있으면 주변 문맥을 보고 의미를 복원한다.
- 단, 원문에 없는 개념을 새로 만들지 않는다.
- 수식, 코드, 고유명사, 날짜, 숫자는 보존한다.
- Stanford, CME 295, ICME, MIT, Uber, Google, Netflix 등 고유명사는 원문 유지한다.

출력 규칙:
- 반드시 JSON만 출력한다.
- 입력된 target_blocks 각각에 대해 하나의 번역 결과를 반환한다.
- 각 결과는 {"block_id": "...", "ko_lines": [...]} 형식이다.
- block_id는 입력의 block_id와 정확히 같아야 한다.
- ko_lines는 해당 SRT 블록 안에 들어갈 한국어 자막 줄 목록이다.
- 가능하면 1~2줄로 자연스럽게 나눈다.
"""


TRANSLATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "block_id": {"type": "string"},
                    "ko_lines": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["block_id", "ko_lines"],
            },
        }
    },
    "required": ["translations"],
}


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_srt(path: Path) -> List[SRTBlock]:
    raw = normalize_newlines(path.read_text(encoding="utf-8-sig"))
    blocks: List[SRTBlock] = []

    for m in SRT_BLOCK_RE.finditer(raw):
        index = m.group("index").strip()
        timing = m.group("timing").strip()
        text = m.group("text").strip("\n")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        blocks.append(SRTBlock(index=index, timing=timing, lines=lines))

    if not blocks:
        raise ValueError(f"SRT 파싱 실패: {path}")

    return blocks


def safe_filename_from_input(path: Path) -> str:
    name = path.name

    name = re.sub(r"^\[English \(United States\)\]\s*", "", name)
    name = name.replace("Stanford CME295 Transformers & LLMs  Autumn 2025  ", "")
    name = name.replace("Stanford CME295 Transformers & LLMs Autumn 2025 ", "")
    name = name.replace(".srt", "")

    name = re.sub(r"\s+", "_", name.strip())
    name = name.replace("&", "and")
    name = re.sub(r"[^A-Za-z0-9가-힣_.\-]+", "_", name)

    return name


def write_korean_srt(
    blocks: List[SRTBlock],
    translations: Dict[str, List[str]],
    out_path: Path,
) -> None:
    chunks = []

    for b in blocks:
        ko_lines = translations.get(b.index)
        if not ko_lines:
            ko_lines = ["[번역 누락] " + " ".join(b.lines)]

        chunks.append(
            f"{b.index}\n"
            f"{b.timing}\n"
            f"{chr(10).join(ko_lines)}\n"
        )

    out_path.write_text("\n".join(chunks), encoding="utf-8")


def write_bilingual_srt(
    blocks: List[SRTBlock],
    translations: Dict[str, List[str]],
    out_path: Path,
) -> None:
    chunks = []

    for b in blocks:
        ko_lines = translations.get(b.index)
        if not ko_lines:
            ko_lines = ["[번역 누락] " + " ".join(b.lines)]

        both_lines = []
        both_lines.extend(ko_lines)
        both_lines.extend(b.lines)

        chunks.append(
            f"{b.index}\n"
            f"{b.timing}\n"
            f"{chr(10).join(both_lines)}\n"
        )

    out_path.write_text("\n".join(chunks), encoding="utf-8")


def load_cache(cache_path: Path) -> Dict[str, List[str]]:
    if not cache_path.exists():
        return {}

    cache: Dict[str, List[str]] = {}

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            block_id = str(obj["block_id"])
            ko_lines = obj["ko_lines"]
            if isinstance(ko_lines, list):
                cache[block_id] = [str(x).strip() for x in ko_lines if str(x).strip()]

    return cache


def append_cache(cache_path: Path, items: List[Dict[str, Any]]) -> None:
    with cache_path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def make_batches(blocks: List[SRTBlock], batch_size: int) -> List[List[SRTBlock]]:
    return [blocks[i:i + batch_size] for i in range(0, len(blocks), batch_size)]


def build_payload(
    batch: List[SRTBlock],
    prev_context: List[SRTBlock],
    next_context: List[SRTBlock],
) -> str:
    payload = {
        "task": "Translate target_blocks into detailed Korean lecture subtitles.",
        "context_before": [
            {
                "block_id": b.index,
                "time": b.timing,
                "english_lines": b.lines,
            }
            for b in prev_context
        ],
        "target_blocks": [
            {
                "block_id": b.index,
                "time": b.timing,
                "english_lines": b.lines,
            }
            for b in batch
        ],
        "context_after": [
            {
                "block_id": b.index,
                "time": b.timing,
                "english_lines": b.lines,
            }
            for b in next_context
        ],
        "important": [
            "Only translate target_blocks.",
            "Use context_before and context_after only to resolve meaning.",
            "Return all target block_ids exactly once.",
            "Do not translate context blocks unless they are also target blocks.",
        ],
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_openai(
    client: OpenAI,
    batch: List[SRTBlock],
    prev_context: List[SRTBlock],
    next_context: List[SRTBlock],
    model: str,
    max_retries: int = 5,
) -> List[Dict[str, Any]]:
    expected_ids = {b.index for b in batch}
    payload = build_payload(batch, prev_context, next_context)

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "srt_translation_batch",
                        "schema": TRANSLATION_SCHEMA,
                        "strict": True,
                    }
                },
            )

            data = json.loads(response.output_text)
            items = data["translations"]

            cleaned: List[Dict[str, Any]] = []
            seen = set()

            for item in items:
                block_id = str(item["block_id"])
                if block_id not in expected_ids:
                    continue
                if block_id in seen:
                    continue

                ko_lines = item["ko_lines"]
                ko_lines = [str(x).strip() for x in ko_lines if str(x).strip()]

                if not ko_lines:
                    original = next(b for b in batch if b.index == block_id)
                    ko_lines = ["[번역 실패] " + " ".join(original.lines)]

                cleaned.append({"block_id": block_id, "ko_lines": ko_lines})
                seen.add(block_id)

            missing = expected_ids - seen
            if missing:
                raise ValueError(f"누락된 block_id: {sorted(missing)}")

            return cleaned

        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"\n[재시도 {attempt + 1}/{max_retries}] {e}")
            time.sleep(wait)

    raise RuntimeError("API 호출 반복 실패")

def call_openai_with_split(
    client: OpenAI,
    batch: List[SRTBlock],
    prev_context: List[SRTBlock],
    next_context: List[SRTBlock],
    model: str,
) -> List[Dict[str, Any]]:
    try:
        return call_openai(
            client=client,
            batch=batch,
            prev_context=prev_context,
            next_context=next_context,
            model=model,
        )
    except RuntimeError:
        if len(batch) == 1:
            b = batch[0]
            print(f"[강제 fallback] block_id={b.index}")
            return [{
                "block_id": b.index,
                "ko_lines": ["[수동 확인 필요] " + " ".join(b.lines)]
            }]

        mid = len(batch) // 2
        left = batch[:mid]
        right = batch[mid:]

        print(f"[배치 분할] {batch[0].index}-{batch[-1].index} "
              f"→ {left[0].index}-{left[-1].index}, {right[0].index}-{right[-1].index}")

        left_result = call_openai_with_split(
            client=client,
            batch=left,
            prev_context=prev_context,
            next_context=right[:CONTEXT_SIZE],
            model=model,
        )

        right_result = call_openai_with_split(
            client=client,
            batch=right,
            prev_context=left[-CONTEXT_SIZE:],
            next_context=next_context,
            model=model,
        )

        return left_result + right_result


def translate_one_file(path: Path, client: OpenAI) -> None:
    print(f"\n=== 번역 시작: {path.name} ===")

    blocks = read_srt(path)
    base = safe_filename_from_input(path)

    ko_out = KOR_DIR / f"{base}.ko_detailed.srt"
    bilingual_out = BILINGUAL_DIR / f"{base}.ko_en_detailed.srt"
    cache_path = CACHE_DIR / f"{base}.cache.jsonl"

    translations = load_cache(cache_path)
    position = {b.index: i for i, b in enumerate(blocks)}
    batches = make_batches(blocks, BATCH_SIZE)

    for batch in tqdm(batches, desc=base):
        batch_ids = [b.index for b in batch]

        if all(block_id in translations for block_id in batch_ids):
            continue

        first_pos = position[batch[0].index]
        last_pos = position[batch[-1].index]

        prev_context = blocks[max(0, first_pos - CONTEXT_SIZE):first_pos]
        next_context = blocks[last_pos + 1:last_pos + 1 + CONTEXT_SIZE]

        result_items = call_openai_with_split(
            client=client,
            batch=batch,
            prev_context=prev_context,
            next_context=next_context,
            model=MODEL,
        )

        append_cache(cache_path, result_items)

        for item in result_items:
            translations[item["block_id"]] = item["ko_lines"]

        write_korean_srt(blocks, translations, ko_out)
        write_bilingual_srt(blocks, translations, bilingual_out)

    missing = [b.index for b in blocks if b.index not in translations]

    if missing:
        print(f"[경고] 번역 누락 {len(missing)}개: {missing[:20]}")
    else:
        print(f"[완료] {ko_out}")
        print(f"[완료] {bilingual_out}")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않음")

    KOR_DIR.mkdir(exist_ok=True)
    BILINGUAL_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    srt_files = sorted(ENG_DIR.glob("*.srt"))

    if not srt_files:
        raise RuntimeError(f"eng 폴더에 .srt 파일이 없음: {ENG_DIR}")

    client = OpenAI()

    print(f"입력 폴더: {ENG_DIR}")
    print(f"한국어 출력 폴더: {KOR_DIR}")
    print(f"병기 출력 폴더: {BILINGUAL_DIR}")
    print(f"모델: {MODEL}")
    print(f"배치 크기: {BATCH_SIZE}")
    print(f"문맥 크기: {CONTEXT_SIZE}")
    print(f"SRT 파일 수: {len(srt_files)}")

    for path in srt_files:
        translate_one_file(path, client)


if __name__ == "__main__":
    main()
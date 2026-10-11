import hashlib
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from app.agent.runtime import SynthesisReplay, _collect_synthesis, _semantic_assessment_prompt
from app.agent.contracts import Evidence
from app.agent.verification import SemanticAssessment, SynthesisRequest, SynthesisReview
from app.core.settings_store import ChatModelConfig


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def real_reviews():
    case_path = ROOT / ".tmp_probe/agent-eval40-20260930/repair-document-coverage-10002/Q31.json"
    if not case_path.exists():
        pytest.skip("需要本地 Q31 真实评测记录")
    case = json.loads(case_path.read_text(encoding="utf-8"))
    paths = list((ROOT / ".tmp_probe/agent-eval40-20260930/instance-10002/workspaces").glob(
        f"*/artifacts/synthesis/{case['run_id']}/review-*.json"
    ))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    responses = [record["raw"] for record in records if record.get("raw")]
    assert len(responses) >= 3
    return responses


@pytest.fixture
def journal_path():
    path = ROOT / ".tmp_probe/synthesis-replay-tests" / f"{uuid4().hex}.json"
    yield path
    path.unlink(missing_ok=True)


def test_completed_real_responses_survive_restart(real_reviews, journal_path):
    replay = SynthesisReplay(journal_path)
    responses = real_reviews[:3]
    keys = [hashlib.sha256(response.encode()).hexdigest() for response in responses]
    for key, response in zip(keys, responses, strict=True):
        assert replay.take(key) is None
        replay.append(key, response)
    resumed = SynthesisReplay(journal_path)
    assert [resumed.take(key) for key in keys] == responses
    assert resumed.take("unfinished-semantic-request") is None


def test_request_change_invalidates_following_responses(real_reviews, journal_path):
    replay = SynthesisReplay(journal_path)
    for index, response in enumerate(real_reviews[:3]):
        replay.append(str(index), response)
    resumed = SynthesisReplay(journal_path)
    assert resumed.take("0") == real_reviews[0]
    assert resumed.take("changed-requirements") is None
    assert len(SynthesisReplay(journal_path).records) == 1
    resumed.append("changed-requirements", real_reviews[1])
    assert resumed.take("2") is None


def test_real_response_survives_process_restart(real_reviews, journal_path):
    SynthesisReplay(journal_path).append("completed", real_reviews[0])
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; from pathlib import Path; from app.agent.runtime import SynthesisReplay; "
         "sys.stdout.buffer.write(SynthesisReplay(Path(sys.argv[1])).take('completed').encode('utf-8'))",
         str(journal_path)],
        cwd=ROOT / "backend", capture_output=True, check=True,
    )
    assert result.stdout.decode("utf-8") == real_reviews[0]


def test_success_or_validation_failure_removes_saved_progress(real_reviews, journal_path):
    replay = SynthesisReplay(journal_path)
    replay.append("accepted-provider-response", real_reviews[0])
    replay.clear()
    assert not journal_path.exists()
    assert SynthesisReplay(journal_path).records == []


def test_real_semantic_response_reused_by_model_entry(journal_path):
    case_path = ROOT / ".tmp_probe/agent-eval40-20260930/repair-document-coverage-10002/Q31.json"
    if not case_path.exists():
        pytest.skip("需要本地 Q31 真实评测记录")
    case = json.loads(case_path.read_text(encoding="utf-8"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in (
        ROOT / ".tmp_probe/agent-eval40-20260930/instance-10002/workspaces"
    ).glob(f"*/artifacts/synthesis/{case['run_id']}/review-*.json")]
    record = next(item for item in records if item.get("stage") == "semantic" and item.get("review"))
    evidence = [Evidence.model_validate(item) for item in record["evidence"]]
    review = SynthesisReview.model_validate(record["review"])
    request = SynthesisRequest(question=record["question"], requirements=record["requirements"], evidence=evidence)
    prompt = _semantic_assessment_prompt(request, review, evidence, [])
    config = ChatModelConfig()
    duty = "独立验收答案与来源"
    packet = [config.model_dump(mode="json"), prompt, "off", duty, "语义验收", []]
    key = hashlib.sha256(json.dumps(packet, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    SynthesisReplay(journal_path).append(key, record["raw"])
    replay = SynthesisReplay(journal_path)
    response = _collect_synthesis(config, prompt, "off", duty, "语义验收", replay=replay)
    assert SemanticAssessment.model_validate_json(response) == SemanticAssessment.model_validate_json(record["raw"])
    assert replay.position == 1
    assert not replay.model_failed

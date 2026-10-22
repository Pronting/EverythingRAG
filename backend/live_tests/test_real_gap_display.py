import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest

from app.agent.contracts import RunStatus
from app.agent.research import MaterialGap
from app.agent.runtime import synthesis_model_for_chat
from app.agent.verification import CandidateAnswer, SynthesisRequest, publish_answer

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '.tmp_probe' / 'agent-eval40-20260930' / 'results-10001'


@pytest.mark.parametrize('case_id', ['Q10', 'Q11'])
@pytest.mark.skipif(os.environ.get('EVERYTHING_RAG_LIVE_EVAL') != '1', reason='需要真实模型调用')
async def test_real_gap_answer_is_concise_and_published_unchanged(case_id):
    record = json.loads((BASE / f'{case_id}.json').read_text(encoding='utf-8'))
    completed = next(event['payload'] for event in record['events'] if event['type'] == 'run.completed')
    gaps = [MaterialGap(missing=missing) for missing in completed['gaps']]
    assert gaps
    model = synthesis_model_for_chat(
        reasoning_level='off', limits=[gap.missing for gap in gaps],
    )
    request = SynthesisRequest(question='我文档中没有这些内容吗？')
    content = await asyncio.to_thread(model, request)
    assert 0 < len(content) <= 120
    assert any(word in content for word in ['本轮', '这次', '目前', '暂', '无法确认'])
    assert all(word not in content for word in ['资料不足说明', '对结论的影响', '缺少的依据：'])
    candidate = CandidateAnswer(
        question=request.question, content=content, run_id=record['run_id'], gaps=gaps,
        answer_basis='insufficient',
    )
    artifact = publish_answer(candidate, requirements={})
    assert artifact.status is RunStatus.COMPLETED_WITH_GAPS
    assert artifact.content == content
    assert artifact.content_hash == hashlib.sha256(content.encode('utf-8')).hexdigest()
    output = ROOT / '.tmp_probe' / 'concise-answers'
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{case_id}.json').write_text(artifact.model_dump_json(), encoding='utf-8')
    print(f'{case_id}: {content}')

import json
import io
from hashlib import sha256
from unittest.mock import patch
from PIL import Image
import pytest

from provenance import inspect_provenance, check_reference_hash, interpret_c2pa_json


def image_bytes():
    out = io.BytesIO()
    Image.new('RGB', (48, 48), '#abc').save(out, format='JPEG')
    return out.getvalue()


def test_hash_matches_known_reference():
    payload = image_bytes()
    result = inspect_provenance(payload, sha256(payload).hexdigest(), tool=None)
    assert result.sha256 == sha256(payload).hexdigest()
    assert result.reference_status == 'Referansla aynı dosya baytları.'


def test_hash_different_from_reference():
    payload = image_bytes()
    assert 'AYNI DEĞİL' in check_reference_hash(sha256(payload).hexdigest(), '0' * 64)


def test_invalid_hash_refuses_comparison():
    with pytest.raises(ValueError):
        check_reference_hash('0' * 64, 'foo')


def test_unavailable_cli_never_fakes_verification():
    with patch('provenance.shutil.which', return_value=None):
        result = inspect_provenance(image_bytes())
    assert result.c2pa_status == 'Araç kurulu değil'
    assert result.c2pa_report is None


def test_no_manifest_does_not_mean_fake():
    status, state, notes = interpret_c2pa_json({'active_manifest': None, 'manifests': {}})
    assert 'bulunamadı' in status
    assert 'sahte' in ' '.join(notes)


def test_invalid_signature_is_not_trusted():
    status, _, _ = interpret_c2pa_json({'active_manifest':'urn:test', 'validation_state':'Trusted',
                              'validation_results': {'activeManifest': {'failure':[{'code':'assertion.hashedURI.mismatch'}]}}})
    assert status == 'C2PA doğrulama sorunu'


def test_valid_not_trusted():
    status, _, _ = interpret_c2pa_json({'active_manifest':'urn:test', 'validation_state':'Valid'})
    assert 'imzacı güveni ayrı' in status


def test_cli_json_integration():
    payload = image_bytes()
    report = {'active_manifest':'urn:test','validation_state':'Trusted',
              'manifests': {'urn:test': {'claim_generator':'Camera A', 'assertions':[{
                   'label':'c2pa.actions.v2', 'data':{'actions':[{'action':'c2pa.created'}]}}]}}}
    class FakeResult:
        returncode = 0
        stdout = json.dumps(report)
        stderr = ''
    with patch('provenance.subprocess.run', return_value=FakeResult()) as call:
        output = inspect_provenance(payload, tool='c2patool')
    assert output.c2pa_status.startswith('C2PA doğrulandı')
    assert output.recorded_actions == ['c2pa.created']
    assert output.claim_generator == 'Camera A'
    args = call.call_args.args[0]
    assert args[0] == 'c2patool'
    assert args[1].endswith('.jpg')

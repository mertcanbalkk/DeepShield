"""Cryptographic/provenance inspection. No hashes alone prove a photo is authentic.

Uses official c2patool when installed; DOES NOT infer authenticity from missing
credentials and DOES NOT claim signature-trust when the CLI did not report it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import re
import shutil
import subprocess
import tempfile

from PIL import Image

MAX_FILE_SIZE = 35 * 1024 * 1024
FORMAT_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tif"}


@dataclass
class ProvenanceReport:
    sha256: str
    reference_status: str
    c2pa_status: str
    validation_state: str | None
    active_manifest: str | None
    claim_generator: str | None
    recorded_actions: list[str]
    notices: list[str]
    c2pa_report: dict | None

    def to_dict(self):
        return asdict(self)


def check_reference_hash(actual_hash: str, expected_hash: str | None) -> str:
    if not expected_hash or not expected_hash.strip():
        return "Referans hash sağlanmadı (dosyanın geçmiş bütünlüğü ölçülemez)."
    expected = expected_hash.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Referans SHA-256 64 karakterlik hex değeri olmalı.")
    return "Referansla aynı dosya baytları." if actual_hash.lower() == expected else "Referansla AYNI DEĞİL (dosya baytları farklı)."


def _manifest_actions(report: dict) -> tuple[str | None, list[str]]:
    key = report.get("active_manifest")
    manifests = report.get("manifests")
    manifest = manifests.get(key, {}) if isinstance(manifests, dict) and key else {}
    if not isinstance(manifest, dict):
        return None, []
    generator = manifest.get("claim_generator")
    if not generator and isinstance(manifest.get("claim_generator_info"), list):
        generator = ", ".join(str(x.get("name", "")) for x in manifest["claim_generator_info"] if isinstance(x, dict))
    actions = []
    for assertion in manifest.get("assertions", []):
        if not isinstance(assertion, dict):
            continue
        if "c2pa.actions" not in str(assertion.get("label", "")):
            continue
        data = assertion.get("data") or {}
        if isinstance(data, dict):
            for entry in data.get("actions", []):
                if isinstance(entry, dict):
                    actions.append(str(entry.get("action", "?")))
    return str(generator)[:160] if generator else None, actions[:30]


def interpret_c2pa_json(raw: dict) -> tuple[str, str | None, list[str]]:
    """Conservative result: Valid != trusted. Missing != invalid. Never trust silently."""
    if not isinstance(raw, dict):
        return "Okunamadı", None, ["C2PA çıktısı JSON nesnesi değil."]
    active = raw.get("active_manifest")
    if not active:
        return "C2PA kaydı bulunamadı", None, ["C2PA yokluğu görselin sahte olduğunu göstermez."]
    validation_state = raw.get("validation_state")
    failures = []
    status = raw.get("validation_status")
    if isinstance(status, list):
        failures.extend(str(x.get("code", x)) for x in status)
    results = raw.get("validation_results")
    if isinstance(results, dict):
        for source in (results.get("activeManifest"), results.get("active_manifest")):
            if isinstance(source, dict):
                for entry in source.get("failure", []):
                    failures.append(str(entry.get("code", entry)))
        if isinstance(results.get("failure"), list):
            failures += [str(x.get("code", x)) for x in results["failure"]]
    notices = []
    if failures or str(validation_state).lower() == "invalid":
        notices.append("Manifest doğrulamasında sorun bildirildi: " + ", ".join(failures[:5]))
        return "C2PA doğrulama sorunu", str(validation_state) if validation_state is not None else None, notices
    if str(validation_state).lower() == "trusted":
        return "C2PA doğrulandı ve güvenilir imzacı bildirildi", str(validation_state), ["Kayıtlı kaynağın doğrulanması, fotoğraftaki her iddiayı kanıtlamaz."]
    if str(validation_state).lower() == "valid":
        return "C2PA yapısal doğrulama başarılı (imzacı güveni ayrı)", str(validation_state), ["İmzacının güven listesinde olduğuna dair doğrulama yok."]
    return "C2PA manifesti var — güven durumu doğrulanamadı", str(validation_state) if validation_state is not None else None, ["Manifest varlığı tek başına güvenilir imza kanıtı değildir."]


def inspect_provenance(data: bytes, expected_sha256: str | None = None, tool: str | None = None,
                       timeout_seconds: int = 18) -> ProvenanceReport:
    if not data or len(data) > MAX_FILE_SIZE:
        raise ValueError("Görsel boş veya dosya 35 MB sınırını aşıyor.")
    digest = sha256(data).hexdigest()
    baseline = check_reference_hash(digest, expected_sha256)
    notices = ["SHA-256 dosyanın mevcut baytlarının parmak izidir; geçmişte müdahale olup olmadığını tek başına açıklamaz."]
    cmd = tool or shutil.which("c2patool")
    if not cmd:
        notices.append("Gerçek C2PA kontrolü için resmî c2patool kurulmalı ve PATH üzerinde bulunmalı.")
        return ProvenanceReport(digest, baseline, "Araç kurulu değil", None, None, None, [], notices, None)
    try:
        with Image.open(BytesIO(data)) as img:
            ext = FORMAT_EXTENSIONS.get((img.format or "").upper())
        if ext is None:
            return ProvenanceReport(digest, baseline, "Biçim desteklenmiyor", None, None, None, [], notices, None)
        with tempfile.TemporaryDirectory(prefix="deepshield-c2pa-") as tmp:
            original_file = Path(tmp) / ("original" + ext)
            original_file.write_bytes(data)
            result = subprocess.run([cmd, str(original_file)], capture_output=True, text=True,
                                    timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        notices.append("C2PA incelemesi süre aşımı nedeniyle tamamlanamadı.")
        return ProvenanceReport(digest, baseline, "Zaman aşımı", None, None, None, [], notices, None)
    except (OSError, ValueError) as e:
        notices.append("C2PA çalıştırılamadı: " + str(e)[:240])
        return ProvenanceReport(digest, baseline, "Teknik hata", None, None, None, [], notices, None)

    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError:
        notices.append(f"c2patool JSON üretmedi (çıkış kodu: {result.returncode}). " + result.stderr[-200:])
        return ProvenanceReport(digest, baseline, "Çıktı okunamadı", None, None, None, [], notices, None)
    c2pa_status, state, new_notices = interpret_c2pa_json(decoded)
    notices.extend(new_notices)
    if result.returncode != 0:
        notices.append("c2patool sıfır dışı çıkış kodu bildirdi; sonuç dikkatle yorumlanmalı.")
        if c2pa_status.startswith("C2PA doğrulandı"):
            c2pa_status = "C2PA araç uyarısı — güven durumu belirsiz"
    active = decoded.get("active_manifest")
    generator, actions = _manifest_actions(decoded)
    return ProvenanceReport(digest, baseline, c2pa_status, state,
                            str(active) if active else None, generator, actions, notices, decoded)

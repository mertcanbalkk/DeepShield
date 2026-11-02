"""Quick public-release checks; does not upload files or access network."""
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
required = ('app.py','model_adapters.py','upload_safety.py','requirements.txt',
            'requirements-ai.txt','requirements-dev.txt','.gitignore',
            '.streamlit/config.toml','README.md','SECURITY.md','PRIVACY.md',
            'THIRD_PARTY_NOTICES.md','.github/workflows/ci.yml')
missing = [name for name in required if not (root/name).is_file()]
local_artifacts = [str(p.relative_to(root)) for folder in ('outputs','datasets','models','.venv')
                   for p in (root/folder).rglob('*') if p.is_file()] if any((root/f).exists() for f in ('outputs','datasets','models','.venv')) else []
if missing:
    print('Eksik yayın dosyaları:', ', '.join(missing))
if local_artifacts:
    print('UYARI: Kaynak ZIP içinde özel veri ya da model dosyaları var:', local_artifacts[:10])
if missing or local_artifacts:
    sys.exit(1)
print('GitHub release temel dosyaları OK. Model doğruluğu ve dış lisanslar ayrıca doğrulanmalıdır.')

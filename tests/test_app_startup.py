"""Panel startup smoke test without installing Streamlit or pulling models."""
from __future__ import annotations
import runpy
import sys
import types
from pathlib import Path


def test_panel_can_start_to_upload_screen(monkeypatch):
    class StopHere(Exception):
        pass

    class MockStreamlit(types.ModuleType):
        sidebar = None

        def __init__(self):
            super().__init__('streamlit')
            self.sidebar = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def cache_resource(self, **_):
            return lambda func: func

        def cache_data(self, **_):
            return lambda func: func

        def radio(self, _label, values, **_):
            return values[0]

        def selectbox(self, _label, values, **_):
            return values[0]

        def checkbox(self, _label, **_):
            return False

        def text_input(self, _label, *, value='', **_):
            return value

        def file_uploader(self, *_args, **_kwargs):
            return None

        def expander(self, *_args, **_kwargs):
            return self

        def stop(self):
            raise StopHere()

        def __getattr__(self, name):
            return lambda *_args, **_kwargs: None

    monkeypatch.setitem(sys.modules, 'streamlit', MockStreamlit())
    project = Path(__file__).resolve().parents[1]
    if str(project) not in sys.path:
        monkeypatch.syspath_prepend(str(project))
    try:
        runpy.run_path(str(project/'app.py'), run_name='__main__')
    except StopHere:
        pass
    else:
        assert False, 'upload screen should wait for an uploaded image'

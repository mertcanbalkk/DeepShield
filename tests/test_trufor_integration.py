import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from unittest.mock import patch
from io import BytesIO
import numpy as np
from PIL import Image
import model_adapters


def test_trufor_cli_receives_directory_and_parses_npz(tmp_path):
    repo=tmp_path/'repo'/'TruFor_train_test'
    repo.mkdir(parents=True)
    (repo/'test.py').write_text('# dummy')
    weights=tmp_path/'weights.pth.tar'
    weights.write_bytes(b'not real')
    image=np.zeros((128,192,3),dtype=np.uint8)
    with BytesIO() as b:
        Image.fromarray(image).save(b,format='JPEG')
        data=b.getvalue()

    def fake_run(command, **kwargs):
        out=Path(command[command.index('-out')+1])
        input_path=Path(command[command.index('-in')+1])
        assert out.parent.is_dir() and out.suffix == '.npz'
        assert out.name == 'result.npz'
        assert input_path.suffix=='.jpg' and input_path.read_bytes()==data
        np.savez_compressed(out,
                            map=np.ones((128,192),np.float32)*.72,
                            conf=np.ones((128,192),np.float32)*.85,
                            score=np.asarray(.65))
        from subprocess import CompletedProcess
        return CompletedProcess(command,returncode=0,stdout='ok',stderr='')

    with patch.object(model_adapters.subprocess,'run',side_effect=fake_run):
        result=model_adapters.run_trufor(image,str(repo.parent),str(weights),
                                         original_bytes=data,original_format='JPEG')
    assert result.mask_scores.shape==(128,192)
    assert abs(result.image_score-.65)<1e-6
    assert abs(result.mask_scores.mean()-.72)<1e-6


def test_trufor_cli_failures_are_visible(tmp_path):
    repo=tmp_path/'repo'/'TruFor_train_test'
    repo.mkdir(parents=True)
    (repo/'test.py').write_text('# dummy')
    weights=tmp_path/'weights.pth.tar'
    weights.write_bytes(b'not real')
    from subprocess import CompletedProcess
    with patch.object(model_adapters.subprocess,'run',return_value=CompletedProcess([],returncode=1,stdout='',stderr='no weights')):
        try:
            model_adapters.run_trufor(np.zeros((10,10,3),dtype=np.uint8),str(repo.parent),str(weights))
            assert False,'must raise'
        except RuntimeError as e:
            assert 'no weights' in str(e)


def test_trufor_windows_absolute_path_safe_output(tmp_path):
    """The official CLI should receive an explicit .npz path, not a directory.

    This avoids its Windows C:\\suspect.png.npz permission failure when passing
    absolute -in paths and asking test.py to derive its output sub-path.
    """
    repo = tmp_path/'TruFor'/'TruFor_train_test'
    repo.mkdir(parents=True)
    (repo/'test.py').write_text('# dummy', encoding='utf-8')
    weights = tmp_path/'trufor.pth.tar'
    weights.write_bytes(b'test')
    from subprocess import CompletedProcess

    def fake_run(command, **kwargs):
        outfile = Path(command[command.index('-out')+1])
        assert outfile.name == 'result.npz'
        assert outfile.parent.is_dir()
        assert str(outfile).endswith('result.npz')
        np.savez(outfile, map=np.ones((32,32),dtype=np.float32)*.3)
        return CompletedProcess(command, 0, 'epoch 81', '')

    with patch.object(model_adapters.subprocess, 'run', side_effect=fake_run):
        result = model_adapters.run_trufor(np.zeros((32,32,3),dtype=np.uint8),
                                          str(repo.parent), str(weights))
    assert result.mask_scores.shape == (32,32)
    assert result.image_score is None


def test_trufor_zero_exit_with_hidden_traceback_is_failure(tmp_path):
    repo = tmp_path/'TruFor'/'TruFor_train_test'
    repo.mkdir(parents=True)
    (repo/'test.py').write_text('# dummy', encoding='utf-8')
    weights = tmp_path/'trufor.pth.tar'
    weights.write_bytes(b'test')
    from subprocess import CompletedProcess
    with patch.object(model_adapters.subprocess,'run', return_value=CompletedProcess([],0,'Epoch: 81',
                                                                                     'PermissionError: C:\\suspect.png.npz')):
        try:
            model_adapters.run_trufor(np.zeros((32,32,3),dtype=np.uint8),str(repo.parent),str(weights))
        except RuntimeError as err:
            assert 'PermissionError' in str(err)
            assert 'GERÇEK HATA' in str(err)
        else:
            assert False, 'No output file must be treated as failure'


def test_trufor_reencoding_when_original_pixels_differ(tmp_path):
    repo = tmp_path/'TruFor'/'TruFor_train_test'
    repo.mkdir(parents=True)
    (repo/'test.py').write_text('# dummy', encoding='utf-8')
    weights = tmp_path/'trufor.pth.tar'
    weights.write_bytes(b'test')
    original = Image.new('RGB',(50,50),'blue')
    with BytesIO() as buff:
        original.save(buff,format='JPEG')
        content = buff.getvalue()
    new_pixels = np.zeros((25,25,3),np.uint8)
    from subprocess import CompletedProcess

    def fake_run(command,**kwargs):
        outfile=Path(command[command.index('-out')+1])
        image_file=Path(command[command.index('-in')+1])
        assert image_file.suffix == '.png'
        with Image.open(image_file) as source:
            assert source.size == (25,25) and source.format == 'PNG'
        np.savez(outfile,map=np.zeros((25,25),np.float32))
        return CompletedProcess(command,0,'ok','')

    with patch.object(model_adapters.subprocess,'run',side_effect=fake_run):
        model_adapters.run_trufor(new_pixels,str(repo.parent),str(weights),
                                  original_bytes=content,original_format='JPEG')


def test_trufor_pickle_legacy_opt_in_only(tmp_path):
    repo=tmp_path/'TruFor'/'TruFor_train_test'
    repo.mkdir(parents=True)
    (repo/'test.py').write_text('# dummy',encoding='utf-8')
    weights=tmp_path/'trufor.pth.tar'
    weights.write_bytes(b'test')
    from subprocess import CompletedProcess
    seen=[]

    def fake_run(command,**kwargs):
        env=kwargs['env']
        seen.append(env.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'))
        outfile=Path(command[command.index('-out')+1])
        np.savez(outfile,map=np.zeros((16,16),np.float32))
        return CompletedProcess(command,0,'','')

    with patch.object(model_adapters.subprocess,'run',side_effect=fake_run):
        with patch.dict(model_adapters.os.environ,{'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':''}):
            model_adapters.run_trufor(np.zeros((16,16,3),np.uint8), str(repo.parent),str(weights))
            model_adapters.run_trufor(np.zeros((16,16,3),np.uint8), str(repo.parent),str(weights),
                                      trust_official_checkpoint=True)
    assert seen==['', '1']


def test_real_subprocess_with_stub_model_produces_output(tmp_path):
    """Exercise real subprocess/temp file plumbing; *not* real TruFor inference."""
    repo=tmp_path/'TruFor'/'TruFor_train_test'
    repo.mkdir(parents=True)
    (repo/'test.py').write_text('''import sys
from pathlib import Path
import numpy as np
out = Path(sys.argv[sys.argv.index('-out')+1])
input_file = Path(sys.argv[sys.argv.index('-in')+1])
assert input_file.exists()
assert out.suffix == '.npz' and out.parent.is_dir()
np.savez(out, map=np.full((24,24),.9,np.float32), conf=np.full((24,24),.8,np.float32), score=np.asarray(.7))
''',encoding='utf-8')
    weights=tmp_path/'weights.pth.tar'
    weights.write_bytes(b'fake only')
    result=model_adapters.run_trufor(np.zeros((24,24,3),np.uint8),str(repo.parent),str(weights))
    assert result.mask_scores.shape==(24,24)
    assert result.image_score == .7

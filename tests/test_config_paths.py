import json
from pathlib import Path
from acsd_surv.config import load_config


def test_paths_resolve_from_config_directory(tmp_path,monkeypatch):
    cfg={'data_root':'../images','split_root':'splits','cache_dir':'cache','output_root':'output',
         'preprocess':{'reference_csv':'references.csv'},
         'model':{'geometry':{'isotropic_cache_dir':'eigen'}}}
    folder=tmp_path/'settings';folder.mkdir();p=folder/'config.json';p.write_text(json.dumps(cfg))
    monkeypatch.chdir(tmp_path)
    loaded=load_config(p)
    assert loaded['data_root']==str(tmp_path/'images')
    assert loaded['preprocess']['reference_csv']==str(folder/'references.csv')
    assert loaded['model']['geometry']['isotropic_cache_dir']==str(folder/'eigen')


def test_cli_accepts_config_relative_to_calling_directory(tmp_path):
    import subprocess,sys
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/NPC.json').read_text())
    cfg['data_provenance'].update(time_origin_verified=True,time_origin_evidence='Generated test definition')
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    result=subprocess.run([sys.executable,str(root/'scripts/train.py'),'--cohort','NPC','--fold','1',
                           '--arm','H','--config','config.json','--output','output','--validate-config'],
                          cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

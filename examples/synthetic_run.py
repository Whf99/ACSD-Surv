"""Run clinical-plus-geometry survival training on generated observations."""
import argparse
import csv
import json
import sys
from pathlib import Path
import nibabel as nib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose an empty output directory')
    data=output/'data';data.mkdir(parents=True,exist_ok=True)
    splits=data/'splits/fold_1';splits.mkdir(parents=True)
    references=[]
    for split_index,split in enumerate(('train','val','test')):
        rows=[]
        for i in range(12):
            identifier=f'synthetic_{split_index}_{i}'
            image=np.random.default_rng(12*split_index+i).random((12,12,12)).astype(np.float32)
            grid=np.indices(image.shape)
            mask=(((grid-5.5)**2).sum(0)<(2+i%3*.2)**2).astype(np.uint8)
            for array,suffix in ((image,'image'),(mask,'mask')):
                nib.save(nib.Nifti1Image(array,np.eye(4)),str(data/f'{identifier}_{suffix}.nii.gz'))
            rows.append(dict(patient_id=identifier,image_path=f'{identifier}_image.nii.gz',
                             label_path=f'{identifier}_mask.nii.gz',stage=1+i%4,time=i+1,recur_label=int(i%3!=1)))
            references.append(dict(patient_id=identifier,origin_x=0,origin_y=0,origin_z=0,length_mm=100))
        with (splits/f'{split}.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    reference=data/'references.csv'
    with reference.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=references[0]);writer.writeheader();writer.writerows(references)
    cfg=json.loads((ROOT/'configs/NPC.json').read_text())
    cfg.update(data_root=str(data),split_root=str(data/'splits'),cache_dir=str(output/'cache'),
               expected_cohort_size=36,evaluation_grid=[2,3,4])
    cfg['data_provenance']={
        'time_origin_verified':True,'time_origin_evidence':'Generated durations',
        'anatomical_reference_verified':True,'anatomical_reference_evidence':'Generated coordinates'}
    cfg['preprocess'].update(reference_csv=str(reference),shape=[8,8,8],crop_mm=[12,12,12],mesh_max_vertices=64)
    cfg['training'].update(epochs=2,patience=2,batch_size=4,replay_block_size=3,geometry_rho=0)
    config=output/'config.json';config.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    from acsd_surv import training
    training.CONFIG_FILE=config;training.HOME_STUDY=output/'results'
    training.worker('NPC',1,'G')
    print('Generated-data results:',output/'results/NPC/fold_1/G')


if __name__=='__main__':main()

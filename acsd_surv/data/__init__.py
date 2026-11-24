"""Cohort-specific input adapters."""
from importlib import import_module
from pathlib import Path
from torch.utils.data import DataLoader

def adapter(cohort):
    if cohort not in ('NPC','HECKTOR','Foshan'):raise ValueError('Unknown cohort')
    return import_module('acsd_surv.data.'+cohort.lower())

def make_loader(rows,config,cohort,train=False):
    module=adapter(cohort)
    dataset=module.NPCDataset(rows,Path(config['data_root']),Path(config['cache_dir']),
                              config['preprocess'],augment=False)
    return DataLoader(dataset,batch_size=config['training']['batch_size'],shuffle=train,
                      num_workers=config['training'].get('num_workers',0),collate_fn=module.collate_batch)

"""Train a survival model on a patient-level fold."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from acsd_surv.config import load_config,validate_config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',required=True,choices=['NPC','HECKTOR','Foshan'])
    p.add_argument('--fold',type=int,required=True,choices=range(1,6))
    p.add_argument('--arm',default='D',choices=['H','A','G','D','E_S','E_eta'])
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validate-config',action='store_true')
    args=p.parse_args()
    config=(args.config or ROOT/'configs'/f'{args.cohort}.json').resolve()
    validate_config(load_config(config),args.arm)
    if args.validate_config:
        print('Configuration is valid');return
    from acsd_surv import training
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    training.CONFIG_FILE=config;training.HOME_STUDY=output
    training.worker(args.cohort,args.fold,args.arm)

if __name__=='__main__':main()

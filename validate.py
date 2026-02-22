from dataset import BER, FER, bin_to_sign, EbN0_to_std, ECC_Dataset
from initialization import initialize, MissingStateException
from models.ECCM import ECCM, ECCM_only_mamba

from torch.utils.data import DataLoader
import torch

import numpy as np

from argparse import ArgumentParser
from typing import List
from tqdm import tqdm
import logging
import random
import json
import os

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def test(model, device, test_loader_list: List[DataLoader], EbNo_range_test):
    model.eval()
    results = {}
    total_ber = 0
    code_length = 1
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_loss = test_ber = cum_count = test_fer = 0.
            with tqdm(total=len(test_loader.dataset), unit='codewords',unit_scale=True, position=0, leave=True, desc=f"Testing {EbNo_range_test[ii]}") as pbar:
                for m, x, z, y, magnitude, syndrome in test_loader:
                    code_length = x.shape[1]
                    z_pred = model(magnitude.to(device), syndrome.to(device))
                    x_pred = model.get_codeword(z_pred, y.to(device))

                    test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                    test_fer += FER(x_pred, x.to(device)) * x.shape[0]
                    cum_count += x.shape[0]
                    pbar.update(x.shape[0])
                    if test_fer >= 100:
                        break
            test_ber /= cum_count
            ln_ber = -np.log(test_ber)
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, BER={test_ber:.2e} -ln(BER)={ln_ber:.2e} TotalBitsCount={cum_count * code_length}')
            results[f"BER_{EbNo_range_test[ii]}"] = test_ber
            total_ber += test_ber/len(test_loader_list)
    results['test_ber'] = total_ber
    return results

def _test(config, model, EbNo_range_test):

    code = config.code
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    test_dataloader_list = [DataLoader(ECC_Dataset(code, [std_test[ii]], len=int(config.test_batch_size)*1000000, zero_cw=False),
                                        batch_size=config.test_batch_size, shuffle=False, num_workers=1) for ii in range(len(std_test))]
    return test(model, 'cuda', test_dataloader_list, EbNo_range_test)

TEST_BATCH_SIZE = 512
def load_path(path, best=False, best_ber=None):
    config, model, *rest = initialize(path, ECCM_only_mamba, experiment=True, summary=False, best=best, best_ber=best_ber)
    config.test_batch_size = TEST_BATCH_SIZE
    return config, model

def find_experiments(test_result_dir):
    experiments = set()
    for path, dirs, files in os.walk(os.path.join(test_result_dir)):
        if 'config.json' not in files:
            continue
        experiments.add(os.path.join(path))
    return experiments

def validate(path):
    EbNo_range_test = range(-10,10)
    experiments = set()
    for experiment in find_experiments(path):
        experiments.add(experiment)

        options = [
            {'best': True, 'best_ber': f'BER_3'}
            
          
        ]

    for experiment in sorted(experiments):
        results = {}
        for kwargs in options:
            key_kwargs = ','.join(str({k:v}) for k,v in kwargs.items())
            key =f'{experiment},{key_kwargs}'
            if key in results:
                continue
       #     print(experiment, kwargs)
            try:
                config, model = load_path(experiment, **kwargs)
                results[key] = _test(config, model, EbNo_range_test=EbNo_range_test)
            except MissingStateException:
                print(f'{experiment=}, checkpoint is missing the state dict')
            except Exception as err:
                print(f'{experiment=}, failed to run for an unknown reason {err}')
            with open(os.path.join(experiment, 'validation_pure_mamba.json'), 'w') as f:
                json.dump(results, f)

def parse_args():
    parser = ArgumentParser('validate')
    parser.add_argument('--path', dest='path', type=str)
    return parser.parse_args()


def main():
    print('Start validation')
    args = parse_args()
    return validate(args.path)
    
    
if __name__ == "__main__":
    main()

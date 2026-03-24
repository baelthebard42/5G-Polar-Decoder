from dataset import BER, FER, bin_to_sign, EbN0_to_std, ECC_Dataset
from initialization import initialize, MissingStateException
from models.ECCM import ECCM, ECCM_only_mamba
from models.AECCT import ECC_Transformer_original
from torch.utils.data import DataLoader
import torch

import numpy as np

from argparse import ArgumentParser
from typing import List
from tqdm import tqdm
import logging
import random
import json
import os, time

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def test(model, device, test_loader_list, EbNo_range_test, is_ecct, min_FER=100):
    model.eval()
    results = {}

    total_ber = 0
    total_inference_time = 0.0
    total_codewords = 0
    total_batches = 0
    num_inferences = 0

    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):

            model.total_num_full_iterations = 0

            test_loss = 0.0
            test_ber = 0.0
            test_fer = 0.0
            cum_count = 0

            inference_time = 0.0
            batch_count = 0

            with tqdm(total=len(test_loader.dataset),
                      unit='codewords',
                      unit_scale=True,
                      desc=f"Testing {EbNo_range_test[ii]}") as pbar:

                for m, x, z, y, magnitude, syndrome in test_loader:

                    num_inferences += 1
                    code_length = x.shape[1]

                    # ---- timing start ----
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()

                    # ---- forward ----
                    z_pred = model(magnitude.to(device), syndrome.to(device))

                    if is_ecct:
                        # ECCT-style processing
                        z_mul = (y * bin_to_sign(x)).to(device)
                        loss, x_pred = model.loss(-z_pred, z_mul, y.to(device))
                        test_loss += loss.item() * x.shape[0]
                    else:
                        # Mamba / normal decoding
                        x_pred = model.get_codeword(z_pred, y.to(device))

                    # ---- timing end ----
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()

                    batch_time = end_time - start_time
                    inference_time += batch_time
                    batch_count += 1

                    # ---- metrics ----
                    test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                    test_fer += FER(x_pred, x.to(device)) * x.shape[0]
                    cum_count += x.shape[0]

                    pbar.update(x.shape[0])

                    # ---- ECCT early stopping ----
                    # if is_ecct:
                    #     if ((min_FER > 0 and test_fer > min_FER and cum_count > 1e5)
                    #             or cum_count >= 1e9):
                    #         print(f"Stopping early at EbN0={EbNo_range_test[ii]}")
                    #         break

            # ---- normalize ----
            test_ber /= cum_count
            test_fer /= cum_count
            avg_time_per_codeword = inference_time / cum_count
            avg_time_per_batch = inference_time / batch_count

            ln_ber = -np.log(test_ber)

            logging.info(
                f'Test EbN0={EbNo_range_test[ii]}, '
                f'BER={test_ber:.2e}, '
                f'FER={test_fer:.2e}, '
                f'-ln(BER)={ln_ber:.2e}, '
                f'AvgTime/codeword={avg_time_per_codeword:.2e}s, '
                f'AvgTime/batch={avg_time_per_batch:.2e}s'
            )

            # ---- store results ----
            results[f"BER_{EbNo_range_test[ii]}"] = test_ber
            results[f"FER_{EbNo_range_test[ii]}"] = test_fer
            results[f"AvgTimePerCodeword_{EbNo_range_test[ii]}"] = avg_time_per_codeword
            results[f"AvgTimePerBatch_{EbNo_range_test[ii]}"] = avg_time_per_batch
            results[f"FullIters_{EbNo_range_test[ii]}"] = model.total_num_full_iterations

            total_ber += test_ber / len(test_loader_list)
            total_inference_time += inference_time
            total_codewords += cum_count
            total_batches += batch_count

    # ---- global stats ----
    results['test_ber'] = total_ber
    results['avg_inference_time_per_codeword'] = total_inference_time / total_codewords
    results['avg_inference_time_per_batch'] = total_inference_time / total_batches
    results['total_number_of_inferences'] = num_inferences

    return results

def _test(config, model, EbNo_range_test):

    code = config.code
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    test_dataloader_list = [DataLoader(ECC_Dataset(code, [std_test[ii]], len=int(config.test_batch_size)*1000, zero_cw=False),
                                        batch_size=config.test_batch_size, shuffle=False, num_workers=1) for ii in range(len(std_test))]
    return test(model, 'cuda', test_dataloader_list, EbNo_range_test, is_ecct=True)

TEST_BATCH_SIZE = 128
def load_path(path, best=False, best_ber=None):
    config, model, *rest = initialize(path, ECC_Transformer_original, experiment=True, summary=False, best=best, best_ber=best_ber)
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
    EbNo_range_test = range(4, 6)
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
            with open(os.path.join(experiment, 'validation_ecct_with_inf_time.json'), 'w') as f:
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


from initialization import dump_config, initialize, save_checkpoint
from dataset import BER, bin_to_sign
from configuration import Config
from validate import validate
from models.ECCM import ECCM, ECCM_only_mamba
from models.AECCT import ECC_Transformer_original

from torch.utils.tensorboard import SummaryWriter
import torch

from tqdm import tqdm
import numpy as np

import logging
import time
import  os


def train_epoch(model, device, train_loader, optimizer, epoch, LR, config: Config, tqdm=tqdm):
    model.train()
    cum_loss = cum_ber = cum_samples = cum_loss = 0.
    t = time.time()
    batch_idx = 0
    for m, x, z, y, magnitude, syndrome in tqdm(train_loader, position=0, leave=True, desc="Training"):
        z_mul = (y * bin_to_sign(x)) # x = 1, y = -1 => z_mul = -1
        z_pred = model(magnitude.to(device), syndrome.to(device))
        loss, x_pred = model.loss(z_pred, z_mul.to(device), y.to(device))
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clipping)
        optimizer.step()
        ###
        ber = BER(x_pred, x.to(device))

        cum_loss += loss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_samples += x.shape[0]
        if batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e}')
        batch_idx += 1
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples, cum_ber / cum_samples


def test(model, device, test_loader_list, EbNo_range_test, tqdm=tqdm):
    model.eval()
    results = {}
    total_ber = 0
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ber = cum_count = 0.
            for m, x, z, y, magnitude, syndrome in tqdm(test_loader, position=0, leave=True, desc="Testing"):
                z_pred = model(magnitude.to(device), syndrome.to(device))
                x_pred = model.get_codeword(z_pred, y.to(device))

                test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                cum_count += x.shape[0]
            test_ber /= cum_count
            ln_ber = -np.log(test_ber)
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, BER={test_ber:.2e} -ln(BER)={ln_ber:.2e}')
            results[f"BER_{EbNo_range_test[ii]}"] = test_ber
            total_ber += test_ber/len(test_loader_list)
    results['test_ber'] = total_ber
    return results


def update_training_state(training_state, epoch, loss, ber):
    training_state['epoch'] = epoch
    training_state['loss'] = loss
    training_state['BER'] = ber
    if ber < training_state.get('best_ber',float('inf')):
        training_state['best_ber'] = ber
        training_state['best_ber_epoch'] = epoch
    if loss < training_state.get('best_loss',float('inf')):
        training_state['best_loss'] = loss
        training_state['best_loss_epoch'] = epoch
    return training_state


def update_test_state(test_state: dict, resutls: dict, epoch: int):
    test_state.update(resutls)
    for key in resutls:
        best_key = f'best_{key}'
        best_result = test_state.get(best_key, float('inf'))
        if best_result <= resutls[key]:
            continue
        test_state[best_key] = resutls[key]
        test_state[f'{best_key}_epoch'] = epoch
    return test_state


def epoch_callback(
        config: Config,
        training_state: dict,
        *args,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        summary_writer: SummaryWriter,
        **kwargs
    ):
    checkpoint = {
        'config': config,
        'state': training_state,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict()
    }

    if training_state['best_loss'] <=  training_state['loss']:
        checkpoint['best_model'] = model.state_dict()
    
    save_checkpoint(checkpoint)
    summary_writer.add_scalar('Train: Loss/Epoch', training_state['loss'], training_state['epoch'])
    summary_writer.add_scalar('Train: BER/Epoch', training_state['BER'], training_state['epoch'])
    summary_writer.add_scalar('Train: Best Loss/Epoch', training_state['best_loss'], training_state['epoch'])
    summary_writer.add_scalar('Train: Best BER/Epoch', training_state['best_ber'], training_state['epoch'])
    if (scheduler := kwargs.get('scheduler')):
        summary_writer.add_scalar('Train: LR/Epoch', scheduler.get_last_lr()[0], training_state['epoch'])

def test_callback(
        config: Config,
        training_state: dict,
        test_results: dict,
        *args,
        summary_writer: SummaryWriter,
        model,
        **kwargs
    ):
    run_name = os.path.basename(os.path.normpath(config.path))
    hparams_dir = os.path.join(config.path, run_name)
    if os.path.exists(hparams_dir):
        for filename in os.listdir(hparams_dir):
            os.remove(os.path.join(hparams_dir, filename))
    
    for key, value in test_results.items():
        if 'BER' in key and not key.endswith('epoch'):
            summary_writer.add_scalar(f'Test: -ln({key})/Epoch', -np.log(value), training_state['epoch'])
        else:
            summary_writer.add_scalar(f'Test: {key}/Epoch', value, training_state['epoch'])
        if (best_result := test_results.get(f'best_{key}')) is not None and best_result >= value:
            torch.save(model.state_dict(), os.path.join(config.path, f'best_model_{key}'))

    summary_writer.add_hparams(
        dump_config(config),
        {**training_state, **test_results},
        run_name=run_name,
        global_step=training_state['epoch']
    )


def train_model(
        args: Config,
        model: torch.nn.Module,
        optimizer,
        training_state,
        dataset,
        summary_writer: SummaryWriter,
        scheduler=None,
        epoch_callback=epoch_callback,
        test_callback=test_callback,
        epochs_per_test=10,
        tqdm=tqdm,
        scheduler_init=torch.optim.lr_scheduler.CosineAnnealingLR
    ):
    """
    Pass None to `epoch_callback` and `test_callback` to disable
    """
    T_max = args.T_max
    lr = args.warmup_lr
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader, test_dataloader_list, _, EbNo_range_test =  dataset
    epoch = training_state.get('epoch', 0)
    test_state = {}
    
    scheduler = None
    for epoch in range(epoch + 1, args.epochs + 1):
        if epoch >= args.warmup_length and scheduler is None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
                param_group['initial_lr'] = param_group['lr']
            scheduler = scheduler_init(optimizer, T_max=T_max, eta_min=args.eta_min, last_epoch=epoch - args.warmup_length)
        if epoch >= args.warmup_length and scheduler is not None:
            lr = scheduler.get_last_lr()[0]
        loss, ber = train_epoch(model, device, train_dataloader, optimizer,
                               epoch, LR=lr, config=args, tqdm=tqdm)
        update_training_state(training_state, epoch, loss, ber)
        if scheduler is not None:
            scheduler.step()

        if epoch_callback is not None:
            epoch_callback(args, training_state, model=model, optimizer=optimizer, scheduler=scheduler, summary_writer=summary_writer)

        if epoch % epochs_per_test == 0:
            results = test(model, device, test_dataloader_list, EbNo_range_test, tqdm=tqdm)
            update_test_state(test_state, results, epoch)
            if test_callback is not None:
                test_callback(args, training_state, test_state, model=model, optimizer=optimizer, scheduler=scheduler, summary_writer=summary_writer)

    return model


from argparse import ArgumentParser
import json

def parse_args(args=None):
    argparser = ArgumentParser('train')
    argparser.add_argument('--code-hint', dest='code_hint', type=str, required=True, help="String hint for code that the decoder will be trained on see the codes dir for available codes")
    argparser.add_argument('--path', dest='path', default='results', required=False, help="Path where the results are saved [Default: results]")
    argparser.add_argument('--config-file', dest='config', type=str, default='train.json', required=False, help="Path to a config file see `configuration.py` for further options")
    argparser.add_argument('--epochs-per-test', dest='epochs_per_test', type=int, default=10, required=False, help="Controls after how many epochs a test will be performed [Default: 10]")
    return argparser.parse_args(args=args)

def load_tune_config(config):
    with open(config, 'r') as f:
        return json.load(f)


DEFAULT_PARAMETERS = dict(
    code_hint="LDPC_N49_K24",
    d_model=128,
    d_state=128,
    N_dec=8,
    warmup_lr=1.0e-3,
    warmup_length=10,
    epochs=20000,
    eta_min=1e-10,
    batch_size=128,
)

def get_next_dir(path):
    try:
        _, runs, _ = next(os.walk(path))
        runs = tuple(filter(lambda name: name.startswith('run'), runs))
        i = len(runs)
    except StopIteration:
        i = 0
    return os.path.join(path, f'run_{i}')

def main():
    args = parse_args()
    training_config = load_tune_config(args.config)
    
    parameters = {
        **DEFAULT_PARAMETERS,
        **training_config,
        'code_hint': args.code_hint
    }



    

    path = get_next_dir(args.path, )            
    config, model, optimizer, training_state, dataset, summary_writer = \
        initialize(path, model_cls=ECC_Transformer_original, resume=False, **parameters)
    model = train_model(
        config,
        model,
        optimizer,
        training_state,
        dataset,
        summary_writer,
        epoch_callback=epoch_callback,
        test_callback=test_callback,
        tqdm=tqdm,
        epochs_per_test=args.epochs_per_test
    )

    validate(config.path)

if __name__ == "__main__":
    main()

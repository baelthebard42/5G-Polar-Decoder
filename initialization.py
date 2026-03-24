import json
from configuration import Code, Config
from dataset import ECC_Dataset, EbN0_to_std, get_generator_and_parity
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
import os, sys
import logging
from hashlib import md5
from dataclasses import fields, MISSING
from models.AECCT import ECC_Transformer_original
import models.AECCT

sys.modules['Model'] = models.AECCT




import torch.serialization
torch.serialization.add_safe_globals([ECC_Transformer_original])


CODES_PATH = os.path.join(os.path.dirname(__file__), "codes")


def config_hash(config):
    config_dict = dump_config(config)
    temp_string = (''.join([f'{k}{v}' for k,v in config_dict.items()])).encode()
    hash_string = md5(temp_string).hexdigest()
    return hash_string.upper()


def code_to_hint(code: Code) -> str:
    return f"{code.code_type.upper()}_N{code.n}_K{code.k}"

def non_default_fields(instance):
    diffs = {}
    for field in fields(instance):
        current = getattr(instance, field.name)
        
        # Automatically add required fields (no default)
        if field.default is MISSING and field.default_factory is MISSING:
            diffs[field.name] = current
        # Automatically add fields with default_factory (likely non-comparable)
        elif field.default_factory is not MISSING:
            diffs[field.name] = current
        # Add fields with default values only if they differ
        elif field.default is not MISSING and current != field.default:
            diffs[field.name] = current

    return diffs

def dump_config(config: Config):
    config_dump = non_default_fields(config)
    config_dump['code_hint'] = code_to_hint(config_dump.pop('code'))
    return config_dump

def update_journal(path, config):
    path = os.path.normpath(path)
    output_dir = os.path.dirname(path)
    experiment_name = os.path.basename(path)
    journal_file = os.path.join(output_dir, 'journal.json')
    if not os.path.isfile(journal_file):
        journal = {}
    else:
        with open(journal_file, 'r') as f:
            journal = json.load(f)
    journal[experiment_name] = dump_config(config)
    with open(journal_file, 'w') as f:
        json.dump(journal, f)
    

def code_from_hint(hint: str,):
    hint = hint.upper()
    code_files = os.listdir(CODES_PATH)
    code_files = [f for f in code_files if hint in f][0]
    code_n = int(code_files.split('_')[1][1:])
    code_k = int(code_files.split('_')[-1][1:].split('.')[0])
    code_type = code_files.split('_')[0]
    code = Code(code_n, code_k, code_type)
    G,H = get_generator_and_parity(code, standard_form=True)
    code.generator_matrix = torch.from_numpy(G).transpose(0,1).long()
    code.pc_matrix = torch.from_numpy(H).long()
    return code


def create_config(
        output_path=".output",
        code_hint="LDPC_N49_K24",
        d_model=128,
        N_dec=8,
        warmup_lr=1.0e-3,
        warmup_length=10,
        lr=5e-4,
        epochs=1000,
        eta_min=1e-10,
        batch_size=64,
        gradient_clipping=1.0,
        resume=False,
        **kwargs
    ):
    code = code_from_hint(code_hint)
    config = Config(
        code=code,
        d_model=d_model, # example_code.n + H.shape[0],
        N_dec=N_dec,
        warmup_lr=warmup_lr,
        warmup_length=warmup_length,
        lr=lr,
        epochs=epochs,
        eta_min=eta_min,
        batch_size=batch_size,
        gradient_clipping=gradient_clipping,
        **kwargs
    )
    if config.experiment_type:
        path = os.path.join(output_path, config.experiment_type, config_hash(config))
    else:
        path = os.path.join(output_path, config_hash(config))
    print(path)

    if not resume:
     config.path = path

    return config


def load_checkpoint(path):
    checkpoint = {}
    config_path = os.path.join(path, 'config.json')
    with open(config_path, 'r') as f:
        config_dict: dict = json.load(f)
    
    config_dict.pop('path', None)
    checkpoint['config'] = Config(
        code=code_from_hint(config_dict.pop('code_hint')),
        path=path,
        **config_dict
    )
    if os.path.isfile(model_path := os.path.join(path, 'model')):
        checkpoint['model'] = torch.load(model_path, weights_only=False, map_location="cuda")
 
    if os.path.isfile(model_path := os.path.join(path, 'best_model')):
        checkpoint['best_model'] = torch.load(model_path, weights_only=False)
    if os.path.isfile(optimizer_path := os.path.join(path, 'optimizer')):
        checkpoint['optimizer'] = torch.load(optimizer_path)
    if os.path.isfile(scheduler_path := os.path.join(path, 'scheduler')):
        checkpoint['scheduler'] = torch.load(scheduler_path)
    if os.path.isfile(state_path := os.path.join(path, 'state.json')):
        with open(state_path) as f:
            checkpoint['state'] = json.load(f)
    return checkpoint


def save_checkpoint(checkpoint):    
    config: Config = checkpoint['config']
    config_file = os.path.join(config.path, 'config.json')
    if not os.path.isfile(config_file):
        with open(os.path.join(config_file),'w') as f:
            json.dump(dump_config(config), f)
    
    if 'model' in checkpoint:
        torch.save(checkpoint['model'], os.path.join(config.path, 'model'))
    if 'best_model' in checkpoint:
        torch.save(checkpoint['best_model'], os.path.join(config.path, 'best_model'))
    
    if 'optimizer' in checkpoint:
        torch.save(checkpoint['optimizer'], os.path.join(config.path, 'optimizer'))
    
    if 'state' in checkpoint:
        with open(os.path.join(config.path, 'state.json'), 'w') as f:
            json.dump(checkpoint['state'], f)
        



def create_dataset(args: Config):
    code = args.code
    train_batch_count=args.train_batch_count
    test_batch_count=args.test_batch_count
    #################################
    EbNo_range_test = range(3, 7)
    EbNo_range_train = range(2, 8)
    std_train = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_train]
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    train_dataloader = DataLoader(ECC_Dataset(code, std_train, len=args.batch_size * train_batch_count, zero_cw=args.zero_cw), batch_size=int(args.batch_size),
                                  shuffle=True, num_workers=args.workers)
    test_dataloader_list = [DataLoader(ECC_Dataset(code, [std_test[ii]], len=int(args.test_batch_size*test_batch_count), zero_cw=False),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers) for ii in range(len(std_test))]
    #################################
    return train_dataloader, test_dataloader_list, EbNo_range_train, EbNo_range_test

def resolve_model(checkpoint, best=False, best_ber=None):
    if not best and 'model' in checkpoint:
        return checkpoint['model']
    if best:
        value = None
        if 'best_model' in checkpoint:
            value = checkpoint['best_model']
        if best_ber is not None:
            target_path = os.path.join(checkpoint['config'].path, f'best_model_{best_ber}')
            if os.path.isfile(target_path):
                print(f'Using {target_path}')
                value = torch.load(target_path)
            else:
                print(f'{target_path} is not a file')
        if value is None:
            print(f'best configured but no file was found')
            return resolve_model(checkpoint, best=False)
        else:
            return value
    return None


class MissingStateException(Exception):
    def __init__(self,):
        super().__init__("Checkpoint missing state")

def default_optimizer_init(model: torch.nn.Module, config: Config):
    return torch.optim.Adam(model.parameters(), lr=config.warmup_lr)

def initialize(path, model_cls, optimizer_init=default_optimizer_init, experiment=None, summary=True, best=False, best_ber=None, resume=False, **parameters):
    """ Initialize model for the training loop.

    This function creates the required objects for the training loop: config, model and optimizer.

    path:
        Path to the outputs directory.
    model_cls:
        Model class, an instance will be created.
        the constructor must expect a single input named conifg meaning: `model_cls(config=config)`.
    optimizer_init:
        A function that creates an optimizer.
        should expect two named parameters model and config: `optimizer_init(model=model, config=config)`
    experiment:
        optional, if supplied will load an existing experiment from `Path(path)`.
    **parameters:
        will be passed to the `Config` object constructor, ignored if loading an experiment!
    """
    print("using path", path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    config = create_config(output_path=path, resume=resume, **parameters)

    if experiment:
        load_path = path
    elif resume:
        print("The path entered is: ", config.path)
        load_path = config.path
    else:
        load_path = None
    
    if load_path is None:
        os.makedirs(config.path, exist_ok=True)
        with open(os.path.join(config.path, 'config.json'), 'w') as f:
            json.dump(dump_config(config), f)
        
        checkpoint = {'config': config, 'state': {}}
    else:
        checkpoint = load_checkpoint(load_path)
        config = checkpoint['config']
    
    handlers = [
            logging.FileHandler(os.path.join(config.path, 'logging.txt')),
            logging.StreamHandler()
        ]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    
    
    
    
    model = model_cls(config=config).to(device)
    model_state_dict = resolve_model(checkpoint, best, best_ber)
    if model_state_dict is not None:
        try:
            model.load_state_dict(model_state_dict,)
        except:
            logging.error(model_state_dict.keys())
            raise


        
    optimizer = optimizer_init(model=model, config=config)
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    try:
        state = checkpoint['state']    
    except KeyError:
        raise MissingStateException()

    training_state = {
        'epoch': state.pop('epoch', 0 ),
        'best_loss': state.pop('best_loss', float('inf')),
        'best_ber': state.pop('best_ber', float('inf')),
        **state
    }

    summary_writer = None
    if summary:
        summary_writer = SummaryWriter(config.path)
    dataset = create_dataset(config)

    logging.info(f'Model {model_cls} initialized size={sum(p.numel() for p in model.parameters())}. {config.layout=} {config.N_dec=} {config.d_model=} {config.code=}')

    return config, model, optimizer, training_state, dataset, summary_writer
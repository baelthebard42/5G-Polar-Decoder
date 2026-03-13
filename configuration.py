from dataclasses import dataclass
from typing import Any, Literal, Optional

@dataclass
class Code():
    n: int
    k: int
    code_type: str
    pc_matrix: Any = None
    generator_matrix: Any = None


@dataclass
class Config():

    # training param
    epochs: int = 20000
    workers: int = 4
    warmup_lr: float = 1e-3
    warmup_length: int = 10
    lr: float = 2.5e-4
    gpus: str = '-1'
    batch_size: int = 128
    test_batch_size: int = 128
    train_batch_count: int = 1000
    test_batch_count: int = 1000
    seed: int = 42
    eta_min: float =1e-6
    gradient_clipping: float = 1.0
    zero_cw: bool = True
    T_max: int = 20000
    layout: str = 'MT'
    experiment_type: str = None
    attention_type: Literal['aecct'] = 'aecct'
    mask_type: Literal['pc_matrix', 'aecct_method'] = 'pc_matrix'
    enable_multi_loss: bool = True

    # code params
    standardize: bool = True

    # dimensions
    N_dec: int = 8
    N_dec_mamba: int = 4
    d_model: int = 64
    d_state: int = 64
    h: int = 8
    code: Code = None

    # other
    path: str = None

    # lpe
    lpe_dim: int = 8
    lpe_num_heads: int = 8

    # head partitioning
    num_heads_for_one_ring: int = 4

    # quantization
    use_aap_linear_training: bool = False
    use_aap_linear_inference: bool = False

    act_bits: int = 8
    initial_percentile: Optional[float] = 0.45

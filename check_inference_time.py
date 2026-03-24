import time
import torch, os
from torch.utils.data import DataLoader
from dataset import ECC_Dataset, EbN0_to_std
from configuration import Code, Config
from initialization import code_from_hint, config_hash
from models.ECCM import ECCM, ECCM_only_mamba
from models.AECCT import ECC_Transformer_original
from typing import Any, Literal, Optional



def create_config(
        output_path=".output",
        code_hint="POLAR_N128_K86",
        d_model=128,
        N_dec=8,
        N_dec_mamba=4,
        warmup_lr=1.0e-3,
        warmup_length=10,
        lr=5e-4,
        epochs=1000,
        eta_min=1e-10,
        batch_size=64,
        gradient_clipping=1.0,
        resume=False,
        mask_type: Literal['pc_matrix', 'aecct_method'] = 'pc_matrix'
    
    ):
    code = code_from_hint(code_hint)
    config = Config(
        code=code,
        d_model=d_model, # example_code.n + H.shape[0],
        N_dec=N_dec,
        N_dec_mamba=N_dec_mamba,
        warmup_lr=warmup_lr,
        warmup_length=warmup_length,
        lr=lr,
        epochs=epochs,
        eta_min=eta_min,
        batch_size=batch_size,
        gradient_clipping=gradient_clipping,
        mask_type=mask_type
    
    )
    if config.experiment_type:
        path = os.path.join(output_path, config.experiment_type, config_hash(config))
    else:
        path = os.path.join(output_path, config_hash(config))
    print(path)

    if not resume:
     config.path = path

    return config
def check_inference_time(
    model_class,
    config,
    device="cuda",
    EbNo=4,
    num_batches=100000 ,
    batch_size=128,
    is_ecct=False   # ✅ NEW FLAG
):
    import torch
    import time
    from dataset import bin_to_sign

    # ---- model ----
    model = model_class(config).to(device)
    model.eval()

    # ---- dataset ----
    code = config.code
    std = EbN0_to_std(EbNo, code.k / code.n)

    dataset = ECC_Dataset(
        code,
        [std],
        len=batch_size * num_batches,
        zero_cw=False
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    total_time = 0.0
    total_codewords = 0
    total_batches = 0

    with torch.no_grad():
        for i, (_, x, z, y, magnitude, syndrome) in enumerate(loader):
            if i >= num_batches:
                break

            magnitude = magnitude.to(device)
            syndrome = syndrome.to(device)
            y = y.to(device)
            x = x.to(device)

            # Only needed for ECCT
            if is_ecct:
                z_mul = (y * bin_to_sign(x))

            # ---- timing start ----
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            z_pred = model(magnitude, syndrome)

            if is_ecct:
                # ✅ ECCT decoding (real inference)
                _loss, x_pred = model.loss(-z_pred, z_mul, y)
            else:
                # ✅ Mamba decoding
                x_pred = model.get_codeword(z_pred, y)

            # ---- timing end ----
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            batch_time = end - start
            total_time += batch_time
            total_codewords += x.shape[0]
            total_batches += 1

    return {
        "avg_time_per_codeword": total_time / total_codewords,
        "avg_time_per_batch": total_time / total_batches,
        "total_time": total_time,
        "num_batches": total_batches,
        "num_codewords": total_codewords,
    }

def main():
    config_mamba_cnn = create_config(d_model=64, N_dec_mamba=4)
    t1 = check_inference_time(model_class=ECCM_only_mamba, config=config_mamba_cnn)
    print(f"Average Inference Time for Mamba-CNN: {t1["avg_time_per_codeword"]*1000000} microseconds\n\n")

    # config_ecct = create_config(d_model=64, N_dec=8, mask_type='aecct_method')
    # t2 = check_inference_time(model_class=ECC_Transformer_original, config=config_ecct, is_ecct=True)
    # print(f"Average Inference Time for ECCT: {t2["avg_time_per_codeword"]*1000000} microseconds\n\n")

main()





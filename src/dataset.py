from torch.utils.data import Dataset
import torch
import numpy as np
from generate_dataset import generate_data
import math, os


DATASET_PATH = '../scripts/data_32bits_polar.csv'

def get_parity_matrix(seq_len, fixed_msg_bit_size):

    PCM_MATRIX_PATH = f'./parity_check_matrices/PolarCode_N{seq_len}_K{fixed_msg_bit_size}.txt'
    pcm = []

    if not os.path.exists(PCM_MATRIX_PATH):
        raise FileNotFoundError(f"Parity check matrix not available for ({seq_len}, {fixed_msg_bit_size}). Add it as a txt file in parity_check_matrices directory.")

    with open(PCM_MATRIX_PATH, "r") as f:
        for line in f:
           
            numbers = line.strip().split()
            numbers_int = [int(x) for x in numbers]

            curr_row = torch.tensor(numbers_int, dtype=torch.long)
            pcm.append(curr_row)

    pcm = torch.stack(pcm)
    return pcm




class PolarDecDataset(Dataset):

    def __init__(self, snr_db, num_samples, seq_length, device,
                  snr_noise_std=0.1,
                 fixed_msg_bit_size=None, transform=None):
        super().__init__()
        self.snr_db = snr_db + np.random.normal(0, snr_noise_std)
        self.num_samples = num_samples
        self.fixed_msg_bit_size = fixed_msg_bit_size
        self.seq_length = seq_length
        self.device = device

        
        # ------------------------------------------------------------
        # PAPER REQUIRED: parity-check matrix
        # H ∈ {0,1}^{(n-k) × n}
        # ------------------------------------------------------------
        self.H = get_parity_matrix(seq_length, fixed_msg_bit_size)

        assert(self.H.shape[0]==seq_length-fixed_msg_bit_size)
        assert(self.H.shape[1]==seq_length)
        self.H = self.H.float()


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

      

        channel_observation_vector, frozen_bit_prior, target = generate_data(
            message_bit_size=self.fixed_msg_bit_size,
            SNRs_db=[self.snr_db]
        )

        # ------------------------------------------------------------
        # Original tensors (KEPT)
        # ------------------------------------------------------------
        channel_tensor = torch.tensor(channel_observation_vector, dtype=torch.float32)
        frozen_tensor = torch.tensor(frozen_bit_prior, dtype=torch.float32)
        snr_tensor = torch.tensor(self.snr_db, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        # code_rate = self.fixed_msg_bit_size/ self.seq_length

        # ------------------------------------------------------------
        # ORIGINAL (NOT paper-faithful): LLR computation
        # ------------------------------------------------------------
        # llrs = 2 * channel_observation_vector * math.pow(10, self.snr_db / 10) * code_rate
        # llrs = torch.tensor(llrs, dtype=torch.float32)

        # ============================================================
        # ================= PAPER-FAITHFUL PIPELINE ==================
        # ============================================================

        # ------------------------------------------------------------
        # y = raw channel observation vector (paper definition)
        # ------------------------------------------------------------
        y = channel_tensor

        # ------------------------------------------------------------
        # |y|  (paper input magnitude)
        # ------------------------------------------------------------
        y_abs = torch.abs(y)

        # ------------------------------------------------------------
        # Hard decision
        # y_b = (1 - sign(y)) / 2
        # ------------------------------------------------------------
        y_hard = (1.0 - torch.sign(y)) / 2.0

        # ------------------------------------------------------------
        # PAPER SYNDROME COMPUTATION (Eq. 3 in paper)
        # s = H * y_b (mod 2)
        # ------------------------------------------------------------
        # NOTE: frozen_bit_prior is NOT used
        # ------------------------------------------------------------
        syndrome = torch.matmul(self.H, y_hard) % 2.0

        # ------------------------------------------------------------
        # FINAL PAPER INPUT
        # y_in = [ |y| ; syndrome ]
        # ------------------------------------------------------------
        yin = torch.cat([y_abs, syndrome], dim=0)

        # ------------------------------------------------------------
        # RETURNS (original + paper-faithful)
        # ------------------------------------------------------------
        # return {
        #     # ---------------- ORIGINAL (kept) ----------------
        #     # "llrs": llrs,                      # NOT paper-faithful
        #     # "frozen_prior": frozen_tensor,     # NOT used anymore
        #     # "snr": snr_tensor,
        #     "target": target_tensor,

        #     # ---------------- PAPER-FAITHFUL ----------------
        #     "y": y,                            # raw channel
        #     "y_abs": y_abs,                    # |y|
        #     "y_hard": y_hard,                  # hard decision
        #     "syndrome": syndrome,              # H y_b
        #     "yin": yin                         # EXACT paper input
        # }
        
        self.H.to(device=self.device)
        return yin, y, target_tensor, syndrome

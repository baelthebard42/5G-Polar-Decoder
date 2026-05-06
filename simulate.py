from PIL import Image
import numpy as np
import io, os
from initialization import code_from_hint, initialize
from configuration import Code
from argparse import ArgumentParser
import json
from dataset import EbN0_to_std
from models.ECCM import ECCM_only_mamba
from models.AECCT import ECC_Transformer_original
import torch, random
from datetime import datetime


def create_new_dir(root_path: str) -> str:
    """
    Creates a new directory inside root_path using current timestamp
    and returns the full path.
    """

    dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = os.path.join(root_path, dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def load_col_perm(perm_path):
    """
    Loads col_perm and col_perm_inv from the .npz artifact produced by
    precompute_systematic_transform.py.

    Returns
    -------
    col_perm : list[int]  — original→permuted mapping
    inv_perm : list[int]  — permuted→original mapping (inverse)
    """
    data = np.load(perm_path)
    col_perm = data["col_perm"].tolist()
    inv_perm = data["col_perm_inv"].tolist()
    return col_perm, inv_perm


def load_image(path: str):
    img = Image.open(path).convert("RGB")
    return img

def load_path(path, best=False, best_ber=None):
    config, model, dataset, *rest = initialize(path, ECC_Transformer_original, experiment=True, summary=False, best=best, best_ber=best_ber)
    return config, model


def sign_to_bin(x):
    # [-1, 1] -> [1, 0]
    return 0.5 * (1 - x)

def bin_to_sign(x):
    # [0,1] -> [1,-1]
    return 1 - 2 * x
# (1 - 2 * x)  > 0, x < 0.5
def EbN0_to_std(EbN0, rate):
    snr =  EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()


def decode_systematic(model_output, inv_perm, k):
    """
    Recover message bits from a non-systematic codeword.

    Steps:
      1. Apply inv_perm to reorder codeword columns into systematic form.
         After reordering, the first k positions correspond to the identity
         block of G_sys, i.e. they ARE the message bits.
      2. Slice the first k bits.

    Parameters
    ----------
    model_output : torch.Tensor, shape (n,)  — hard-decision codeword (0s and 1s)
    inv_perm     : list[int], length n       — precomputed inverse column permutation
    k            : int                       — number of message bits

    Returns
    -------
    u_hat : torch.Tensor, shape (k,)
    """
    inv_perm_tensor = torch.tensor(inv_perm, dtype=torch.long)
    c_perm = model_output[:, inv_perm_tensor]   # reorder into systematic column space
    return c_perm[:, :k]                        # first k positions = message bits




def img_to_bitstream(img:Image):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    byte_data = buffer.getvalue()
    #print(f"byte_data: {byte_data}\n\n")
    packed_bits = np.frombuffer(byte_data, dtype=np.uint8) # packed bits means in byte form, in decimal system that represents 8 bits
    #print(f"packed bits: {packed_bits.shape}\n\n")
    bitstream = np.unpackbits(packed_bits) # these are actual bits
    #print(f"bitstream: {bitstream.shape}\n\n")
    return bitstream, len(bitstream)


def bitarray_to_chunks(k: int, bit_array):
    bit_array = torch.tensor(bit_array, dtype=torch.int64)
    pad_len = (-bit_array.numel()) % k
    if pad_len != 0:
        padding = torch.zeros(pad_len, dtype=torch.int64)
        bit_array = torch.cat([bit_array, padding])
    chunks = bit_array.view(-1, k)
    return chunks



def chunks_to_bitarray(chunks):
    bit_array = chunks.reshape(-1)
    return bit_array
    


def bitstream_to_img(bitstream, original_length=None):
    bitstream = np.array(bitstream, dtype=np.uint8)
    if original_length is not None:
        bitstream = bitstream[:original_length]
    byte_array = np.packbits(bitstream)
    byte_data = byte_array.tobytes()
    try:
        img = Image.open(io.BytesIO(byte_data))
        img = img.convert("RGB")  
        return img
    except Exception as e:
        print("Image reconstruction failed:", e)
        return None
    


def decode_ldpc(x, code):
    return x[:, :code.k]
    

def encoded_to_msg_bits(encoded_msg_vector, code,  inv_perm, systematic=True):

    if code.code_type == 'LDPC' and not systematic:
        return decode_systematic(encoded_msg_vector, inv_perm, code.k)
    if code.code_type == 'LDPC' and systematic:
        return decode_ldpc(encoded_msg_vector, code)
    if code.code_type == 'POLAR':
        raise NotImplementedError
    if code.code_type == 'BCH':
        raise NotImplementedError

    

def create_data_from_chunks(chunks, code, sigma):
    m =  chunks
    x = torch.matmul(m, code.generator_matrix.transpose(0,1)) % 2
    z = torch.randn(x.shape[0], x.shape[1]) * sigma
    y = bin_to_sign(x) + z
    magnitude = torch.abs(y)
    syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(),
                            code.pc_matrix.transpose(0,1)) % 2
    syndrome = bin_to_sign(syndrome)
    return m.float(), x.float(), z.float(), y.float(), magnitude.float(), syndrome.float()




def get_codeword_from_model(model, y, mag, syn, device, batch_size=1024):
    model.eval()
    outputs = []
    n = y.shape[0]
    with torch.no_grad():
     #   print(f"Starting model inference...\n\n")
        for i in range(0, n, batch_size):
     #       print(f"Processing step {i}\n")
            y_batch = y[i:i+batch_size].to(device)
            mag_batch = mag[i:i+batch_size].to(device)
            syn_batch = syn[i:i+batch_size].to(device)
            z_pred = model(mag_batch, syn_batch)
            x_pred = model.get_codeword(z_pred, y_batch)
            outputs.append(x_pred.cpu())  
    x_pred = torch.cat(outputs, dim=0)
    return x_pred

def frange(start, stop, step):
    while start <= stop:
        yield round(start, 6)  
        start += step


def bp_decode_llr(y, H, max_iter=20):
    """
    Belief Propagation (Min-Sum) LDPC decoder.

    Args:
        y: (N, n) received BPSK symbols
        H: (m, n) parity-check matrix (0/1)
        max_iter: iterations

    Returns:
        x_hat: decoded bits (N, n)
    """

    device = y.device
    N, n = y.shape
    m = H.shape[0]

    H = H.to(device)

    # BPSK LLR initialization: L = 2y/sigma^2 (sigma absorbed already in y scaling)
    # since y = ±1 + noise, we approximate:
    L_ch = 2 * y

    # messages: variable -> check and check -> variable
    # shape: (m, n)
    msg_vc = torch.zeros((N, m, n), device=device)
    msg_cv = torch.zeros((N, m, n), device=device)

    for _ in range(max_iter):

        # -------------------------
        # Check node update
        # -------------------------
        for i in range(m):
            idx = (H[i] == 1).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue

            for j in idx:
                others = idx[idx != j]

                if len(others) == 0:
                    msg_cv[:, i, j] = 0
                    continue

                signs = torch.prod(torch.sign(msg_vc[:, i, others]), dim=1)
                min_vals = torch.min(torch.abs(msg_vc[:, i, others]), dim=1).values

                msg_cv[:, i, j] = signs * min_vals

        # -------------------------
        # Variable node update
        # -------------------------
        for j in range(n):
            idx = (H[:, j] == 1).nonzero(as_tuple=True)[0]

            if len(idx) == 0:
                continue

            incoming = msg_cv[:, idx, j].sum(dim=1)
            for i in idx:
                msg_vc[:, i, j] = L_ch[:, j] + incoming - msg_cv[:, i, j]

    # Final LLR
    L_final = L_ch + msg_cv.sum(dim=1)
    x_hat = (L_final < 0).long()

    return x_hat

def decode_with_bp(y, code, max_iter=20):
    """
    BP decoder entry point matching your model interface.
    """

    return bp_decode_llr(y, code.pc_matrix, max_iter=max_iter)


def parse_args(args=None):
    argparser = ArgumentParser('simulate')
    argparser.add_argument('--code-hint', dest='code_hint', type=str, required=True, help="String hint for code that the decoder will be trained on see the codes dir for available codes")
    argparser.add_argument('--path', dest='path', default='results', required=False, help="path of model")
    argparser.add_argument("--snr_lower", dest="snr_lower", type=float, required=True, help="decides lower noise level to corrupt the transmitted bits during simulation")
    argparser.add_argument("--snr_upper", dest="snr_upper", type=float, required=True, help="decides upper noise level to corrupt the transmitted bits during simulation")
    argparser.add_argument("--snr_step", dest="snr_step", type=float, required=True, help="step increment in snr for simulation")
    argparser.add_argument("--img_data_path", dest="img_data_path", type=str, required=True, help="a folder full of image files to transmit data")
    argparser.add_argument("--results_path", dest="results_path", type=str, required=True, help="resulting images and evaluations will be saved in the folder")
    argparser.add_argument(
    "--decoder",
    type=str,
    default="model",
    choices=["model", "bp"],
    help="Choose decoding method: model or belief propagation"
)
    argparser.add_argument('--transform', dest='transform', type=str,
                           required=True,
                           help='Path to the .npz artifact from precompute_systematic_transform.py')
    return argparser.parse_args(args=args)



    
def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n\n")
    args = parse_args()
    saving_directory_path = create_new_dir(args.results_path)

    config, model = load_path(args.path)
    code = code_from_hint(args.code_hint)
    col_perm, inv_perm = load_col_perm(args.transform)
    test_results = {}
    test_results['model_type'] = config.path
    test_results['code_type'] = args.code_hint
    test_results['snr_lower'] = args.snr_lower
    test_results['snr_upper'] = args.snr_upper

 
    
    for i, each_img_path in enumerate(os.listdir(args.img_data_path)):
        print(f"Processing image {i}: {each_img_path}...\n\n")
        
        test_results[i]={}
        test_results[i]['image_name']=each_img_path
        bitstream, original_length = img_to_bitstream(load_image(os.path.join(args.img_data_path, each_img_path)))
    
      

        for snr in frange(args.snr_lower, args.snr_upper, args.snr_step):
            snr_str = f"{snr:.2f}".replace('.', '_')
            snr_folder_path = os.path.join(saving_directory_path, f'snr_{snr_str}_db_transmitted')
            os.makedirs(snr_folder_path, exist_ok=True)
            test_results[i][f'snr_{snr_str}'] = {}
            print(f"Simulating transmission for SNR={snr_str} dB...\n")
            sigma = EbN0_to_std(EbN0=snr, rate=code.k/code.n)
            chunks = bitarray_to_chunks(k=code.k, bit_array=bitstream)
            _, x, _, y, mag, syn = create_data_from_chunks(chunks, code, sigma)


            if args.decoder == "model":
             predicted_codewords = get_codeword_from_model(model, y, mag, syn, device)
            elif args.decoder == "bp":
             predicted_codewords = decode_with_bp(y, code, max_iter=30)
            else:
             raise ValueError("Unknown decoder")
          

            recovered_message_bits = encoded_to_msg_bits(predicted_codewords, code, inv_perm)
            recovered_bitstream = chunks_to_bitarray(recovered_message_bits)

            recovered_image = bitstream_to_img(recovered_bitstream, original_length)

          #  print(f"recovered bitstream shape: {recovered_bitstream.shape}\n\noriginal bitstream shape: {bitstream.shape}")

            ber = BER(recovered_bitstream[:original_length], torch.tensor(bitstream))
            test_results[i][f'snr_{snr_str}']['BER'] = ber
            print(f"Bit error rate(BER) = {ber}")

            if recovered_image is not None:
                recovered_image.save(os.path.join(snr_folder_path, f"{each_img_path.split(".")[0]}_recovered.jpg"))

      

     


    overall_metrics = {}

    num_images = len(os.listdir(args.img_data_path))

    for snr in frange(args.snr_lower, args.snr_upper + 1, args.snr_step):
        snr_str = f"{snr:.2f}".replace('.', '_')
        ber_values = []

        for i in range(num_images):
            try:
                ber = test_results[i][f'snr_{snr_str}']['BER']
                ber_values.append(ber)
            except KeyError:
                continue

        if ber_values:
            avg_ber = sum(ber_values) / len(ber_values)
        else:
            avg_ber = None

        overall_metrics[f'snr_{snr_str}'] = {
            "average_BER": avg_ber
        }

    test_results['overall_metrics'] = overall_metrics


    json_path = os.path.join(saving_directory_path, f"test_results_{args.decoder}.json")

    with open(json_path, "w") as f:
        json.dump(test_results, f, indent=4)

    print(f"\nSaved all test statistics and recovered data in directory {saving_directory_path}")
    





main()
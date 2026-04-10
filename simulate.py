from PIL import Image
import numpy as np
import io
from initialization import code_from_hint, initialize
from configuration import Code
from argparse import ArgumentParser
import json
from dataset import EbN0_to_std
from models.ECCM import ECCM_only_mamba
import torch


def load_image(path: str):
    img = Image.open(path).convert("RGB")
    return img

def load_path(path, best=False, best_ber=None):
    config, model, dataset, *rest = initialize(path, ECCM_only_mamba, experiment=True, summary=False, best=best, best_ber=best_ber)
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
    

def encoded_to_msg_bits(encoded_msg_vector, code):

    if code.code_type == 'LDPC':
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
        print(f"Starting model inference...\n\n")
        for i in range(0, n, batch_size):
            print(f"Processing step {i}\n")
            y_batch = y[i:i+batch_size].to(device)
            mag_batch = mag[i:i+batch_size].to(device)
            syn_batch = syn[i:i+batch_size].to(device)
            z_pred = model(mag_batch, syn_batch)
            x_pred = model.get_codeword(z_pred, y_batch)
            outputs.append(x_pred.cpu())  
    x_pred = torch.cat(outputs, dim=0)
    return x_pred


def parse_args(args=None):
    argparser = ArgumentParser('simulate')
    argparser.add_argument('--code-hint', dest='code_hint', type=str, required=True, help="String hint for code that the decoder will be trained on see the codes dir for available codes")
    argparser.add_argument('--path', dest='path', default='results', required=False, help="Path where the results are saved [Default: results]")
    argparser.add_argument("--snr_lower", dest="snr_lower", type=float, required=True, help="decides lower noise level to corrupt the transmitted bits during simulation")
    argparser.add_argument("--snr_upper", dest="snr_upper", type=str, required=True, help="decides lower noise level to corrupt the transmitted bits during simulation")
    return argparser.parse_args(args=args)



    
def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n\n")
    args = parse_args()

    config, model = load_path(args.path)

    bitstream, original_length = img_to_bitstream(load_image("./bird_original.jpeg"))
    code = code_from_hint(args.code_hint)

    for snr in range(args.snr_lower, args.snr_upper+1):
        print(f"\n\nSimulating transmission for SNR={snr} dB...\n\n")
        sigma = EbN0_to_std(EbN0=snr, rate=code.k/code.n)
        chunks = bitarray_to_chunks(k=code.k, bit_array=bitstream)
        _, x, _, y, mag, syn = create_data_from_chunks(chunks, code, sigma)
        predicted_codewords = get_codeword_from_model(model, y, mag, syn, device)
        print(f"Bit error rate for model output: {BER(predicted_codewords, x)}")
        recovered_message_bits = encoded_to_msg_bits(predicted_codewords, code)
        recovered_bitstream = chunks_to_bitarray(recovered_message_bits)
        recovered_image = bitstream_to_img(recovered_bitstream)
        recovered_image.save(f"recovered_snr_{args.snr}.jpg")
   





main()
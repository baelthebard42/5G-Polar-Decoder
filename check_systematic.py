import torch
import random
import numpy as np
from argparse import ArgumentParser
from initialization import code_from_hint


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





def parse_args(args=None):
    argparser = ArgumentParser('simulate')
    argparser.add_argument('--code-hint', dest='code_hint', type=str, required=True)
    argparser.add_argument('--k', dest='k', type=int, required=True)
    argparser.add_argument('--num_trials', dest="num_trials", type=int, required=True)
    argparser.add_argument('--transform', dest='transform', type=str,
                           required=True,
                           help='Path to the .npz artifact from precompute_systematic_transform.py')
    return argparser.parse_args(args=args)


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
    c_perm = model_output[inv_perm_tensor]   # reorder into systematic column space
    return c_perm[:k]                        # first k positions = message bits


def main():
    args = parse_args()

    code = code_from_hint(args.code_hint)
    col_perm, inv_perm = load_col_perm(args.transform)

    print(f"col_perm[:10]:  {col_perm[:10]}")
    print(f"inv_perm[:10]:  {inv_perm[:10]}")

    G = code.generator_matrix.to(torch.int64)

    same_count = 0
    diff_count = 0
    bitwise_mismatch_total = 0

    for _ in range(args.num_trials):
        # Generate random message bits
        u = torch.randint(0, 2, (args.k,), dtype=torch.int64)

        # Encode with original G
        encoded = (u @ G.T) % 2

        # Permute to systematic order and slice first k bits
        u_hat = decode_systematic(encoded, inv_perm, args.k)

        if torch.equal(u_hat, u):
            same_count += 1
        else:
            diff_count += 1
            bitwise_mismatch_total += torch.sum(u_hat != u).item()

    print("===== REPORT =====")
    print(f"Total trials:                       {args.num_trials}")
    print(f"Exact matches (systematic):         {same_count}")
    print(f"Mismatches:                         {diff_count}")

    if diff_count > 0:
        avg_bit_errors = bitwise_mismatch_total / diff_count
        print(f"\nIt is NOT systematic")
        print(f"Average bit errors per mismatch:    {avg_bit_errors:.2f}")
    else:
        print(f"\nIt IS systematic ✅")
        print(f"Average bit errors per mismatch:    0")

    print("==================")


main()
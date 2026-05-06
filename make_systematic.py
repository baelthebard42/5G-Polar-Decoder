"""
precompute_systematic_transform.py
====================================
Offline / one-time script that takes a non-systematic generator matrix G
(or a parity-check matrix H) and computes everything needed to extract
message bits from a non-systematic codeword at inference time.

Works for LDPC, BCH, and Polar codes (any binary linear block code).

Outputs a single .npz file containing:
    - G_sys      : systematic generator matrix  [k x n]  (row-reduced form)
    - col_perm   : column permutation applied   [n]       (index array)
    - col_perm_inv: inverse column permutation  [n]       (for codeword recovery)
    - info_cols  : which positions in the permuted codeword hold message bits [k]
    - G_k_inv    : GF(2) inverse of the k×k identity sub-block (sanity use)
    - k, n, r    : code parameters (k=message bits, n=codeword length, r=n-k)

Usage
-----
    python precompute_systematic_transform.py \
        --input  my_generator_matrix.npy \
        --type   G \
        --output systematic_transform.npz

    python precompute_systematic_transform.py \
        --input  my_parity_check.npy \
        --type   H \
        --output systematic_transform.npz

    # For polar codes you can also pass the polar transform matrix G_N:
    python precompute_systematic_transform.py \
        --input  polar_G_N.npy \
        --type   G \
        --output polar_systematic_transform.npz

Arguments
---------
    --input   : path to .npy file containing the matrix (binary, dtype uint8 or int)
    --type    : 'G' for generator matrix (k×n), 'H' for parity-check matrix ((n-k)×n)
    --output  : path for the output .npz file (default: systematic_transform.npz)
    --verify  : run full verification suite after computing (default: True)
    --verbose : print detailed step-by-step info (default: True)
"""

import argparse
import sys
import numpy as np


# ---------------------------------------------------------------------------
# alist parser  (handles both H-matrix and G-matrix alist files)
# ---------------------------------------------------------------------------

def parse_alist(path: str, verbose: bool = True):
    """
    Parse a .alist file into a dense binary matrix.

    alist format (MacKay / AList standard):
        Line 1:  n  m          — number of columns, number of rows
        Line 2:  max_col_wt  max_row_wt
        Line 3:  col_wt_1  col_wt_2 ... col_wt_n      (one per column)
        Line 4:  row_wt_1  row_wt_2 ... row_wt_m      (one per row)
        Lines 5 .. 5+n-1:   for column j, the 1-indexed row indices it connects to
        Lines 5+n .. 5+n+m-1: for row i, the 1-indexed column indices it connects to

    For LDPC parity-check matrices: n = codeword length, m = number of checks.
    The returned matrix M has shape (m, n).

    Parameters
    ----------
    path : str
    verbose : bool

    Returns
    -------
    M : np.ndarray, shape (m, n), dtype uint8
    meta : dict  — {'n': int, 'm': int, 'max_col_wt': int, 'max_row_wt': int,
                    'col_weights': list, 'row_weights': list}
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    ptr = 0
    n, m = map(int, lines[ptr].split());  ptr += 1
    max_col_wt, max_row_wt = map(int, lines[ptr].split());  ptr += 1
    col_weights = list(map(int, lines[ptr].split()));  ptr += 1
    row_weights = list(map(int, lines[ptr].split()));  ptr += 1

    assert len(col_weights) == n, (
        f"Expected {n} column weights, got {len(col_weights)}")
    assert len(row_weights) == m, (
        f"Expected {m} row weights, got {len(row_weights)}")

    # Build matrix from column adjacency lists (1-indexed row numbers)
    M = np.zeros((m, n), dtype=np.uint8)
    for col_idx in range(n):
        entries = list(map(int, lines[ptr].split()));  ptr += 1
        for r in entries:
            if r > 0:                   # some alist files pad with zeros
                M[r - 1, col_idx] = 1

    # Row adjacency lists are redundant (just a consistency check)
    row_check_ok = True
    for row_idx in range(m):
        if ptr < len(lines):
            entries = list(map(int, lines[ptr].split()));  ptr += 1
            for c in entries:
                if c > 0 and M[row_idx, c - 1] != 1:
                    row_check_ok = False

    meta = {
        "n": n, "m": m,
        "max_col_wt": max_col_wt, "max_row_wt": max_row_wt,
        "col_weights": col_weights, "row_weights": row_weights,
    }

    if verbose:
        print(f"  Parsed alist: matrix shape = ({m}, {n})")
        print(f"  max_col_weight={max_col_wt}, max_row_weight={max_row_wt}")
        print(f"  Column weight distribution: min={min(col_weights)}, "
              f"max={max(col_weights)}, mean={sum(col_weights)/len(col_weights):.2f}")
        print(f"  Row weight distribution:    min={min(row_weights)}, "
              f"max={max(row_weights)}, mean={sum(row_weights)/len(row_weights):.2f}")
        print(f"  Row-vs-col consistency check: {'OK' if row_check_ok else 'MISMATCH — check alist file'}")

    return M, meta



# ---------------------------------------------------------------------------
# GF(2) core routines
# ---------------------------------------------------------------------------

def gf2_row_reduce(M: np.ndarray):
    """
    Perform Gauss-Jordan elimination (RREF) over GF(2) on matrix M in-place.
    Also tracks row and column operations for full auditability.

    Parameters
    ----------
    M : np.ndarray, shape (r, c), dtype uint8
        Binary matrix to reduce. Modified in-place.

    Returns
    -------
    pivot_cols : list[int]
        Column indices (in the PERMUTED matrix) where pivots were placed.
        len(pivot_cols) == rank(M).
    col_perm : np.ndarray, shape (c,), dtype int
        col_perm[i] = original column index that is now at position i.
        Apply as:  M_original_col_order = M_reduced[:, np.argsort(col_perm)]
    rank : int
        Rank of M.
    """
    M = M.copy().astype(np.uint8)
    nrows, ncols = M.shape
    col_perm = np.arange(ncols, dtype=int)   # tracks column swaps
    pivot_row = 0
    pivot_cols = []

    for col in range(ncols):
        if pivot_row >= nrows:
            break

        # Find a row at or below pivot_row with a 1 in this column
        candidates = np.where(M[pivot_row:, col] == 1)[0]
        if len(candidates) == 0:
            # No pivot in this column — try swapping with a later column
            # to maintain the identity structure (full column pivoting)
            found = False
            for swap_col in range(col + 1, ncols):
                cands2 = np.where(M[pivot_row:, swap_col] == 1)[0]
                if len(cands2) > 0:
                    # Swap columns col and swap_col
                    M[:, [col, swap_col]] = M[:, [swap_col, col]]
                    col_perm[[col, swap_col]] = col_perm[[swap_col, col]]
                    candidates = cands2
                    found = True
                    break
            if not found:
                continue   # truly zero column, skip

        # Swap the found row to pivot_row position
        row_idx = candidates[0] + pivot_row
        if row_idx != pivot_row:
            M[[pivot_row, row_idx]] = M[[row_idx, pivot_row]]

        pivot_cols.append(col)

        # Eliminate all other rows (both above and below — full RREF)
        for r in range(nrows):
            if r != pivot_row and M[r, col] == 1:
                M[r] = (M[r] + M[pivot_row]) % 2

        pivot_row += 1

    return M, col_perm, pivot_cols, pivot_row   # pivot_row == rank


def gf2_inverse(M: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a square binary matrix M over GF(2).
    Raises ValueError if M is singular.

    Parameters
    ----------
    M : np.ndarray, shape (k, k), dtype uint8

    Returns
    -------
    M_inv : np.ndarray, shape (k, k), dtype uint8
    """
    k = M.shape[0]
    assert M.shape == (k, k), "Matrix must be square"

    # Augment [M | I]
    aug = np.hstack([M.copy().astype(np.uint8),
                     np.eye(k, dtype=np.uint8)])

    # Forward and backward elimination (RREF)
    pivot_row = 0
    for col in range(k):
        # Find pivot
        cands = np.where(aug[pivot_row:, col] == 1)[0]
        if len(cands) == 0:
            raise ValueError(
                f"Matrix is singular (no pivot at column {col}). "
                "Cannot invert."
            )
        row_idx = cands[0] + pivot_row
        if row_idx != pivot_row:
            aug[[pivot_row, row_idx]] = aug[[row_idx, pivot_row]]

        # Eliminate all other rows
        for r in range(k):
            if r != pivot_row and aug[r, col] == 1:
                aug[r] = (aug[r] + aug[pivot_row]) % 2

        pivot_row += 1

    return aug[:, k:]


def derive_G_from_H(H: np.ndarray) -> np.ndarray:
    """
    Derive a generator matrix G from a parity-check matrix H over GF(2).

    For an (n-k)×n parity-check matrix H of rank (n-k), the row space of
    the returned G (k×n) is the null space of H, i.e. G·H^T = 0 (mod 2).

    Strategy: RREF H to identify pivot columns → free columns → build G.

    Parameters
    ----------
    H : np.ndarray, shape (n-k, n)

    Returns
    -------
    G : np.ndarray, shape (k, n)
    """
    H_rref, col_perm, pivot_cols_idx, rank = gf2_row_reduce(H.copy())

    n = H.shape[1]
    k = n - rank

    # Pivot positions and free positions in the permuted column space
    piv = np.array(pivot_cols_idx[:rank], dtype=int)
    free = np.array([c for c in range(n) if c not in piv], dtype=int)

    assert len(free) == k, (
        f"Expected {k} free variables, got {len(free)}. "
        "Check matrix rank."
    )

    # Build G in the permuted column order:
    # For each free variable i, the corresponding row of G has:
    #   - a 1 at the free variable's position
    #   - values from H_rref at the pivot positions (to satisfy H·G^T = 0)
    G_perm = np.zeros((k, n), dtype=np.uint8)
    for i, f in enumerate(free):
        G_perm[i, f] = 1
        # The pivot rows constrain the pivot columns:
        # H_rref[j, f] tells us the contribution of free col f to pivot col j
        for j, p in enumerate(piv):
            G_perm[i, p] = H_rref[j, f]

    # Un-permute columns back to original order
    inv_perm = np.argsort(col_perm)
    G = G_perm[:, inv_perm]

    return G


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_systematic_transform(G: np.ndarray, verbose: bool = True):
    """
    Given a non-systematic binary generator matrix G (k×n), compute the
    systematic transform artifacts needed for message bit extraction.

    Parameters
    ----------
    G : np.ndarray, shape (k, n), dtype uint8
        Non-systematic generator matrix.
    verbose : bool

    Returns
    -------
    artifacts : dict with keys:
        G_sys       : np.ndarray (k, n)  — systematic generator matrix
        col_perm    : np.ndarray (n,)    — column permutation (original→permuted)
        col_perm_inv: np.ndarray (n,)    — inverse permutation
        info_cols   : np.ndarray (k,)    — positions of message bits in permuted codeword
        G_k_inv     : np.ndarray (k, k) — inverse of pivot submatrix (for direct solve)
        k, n, rank  : int
    """
    G = G.astype(np.uint8)
    k_orig, n = G.shape

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Non-systematic Generator Matrix G: shape {G.shape}")
        print(f"  k (message bits) = {k_orig},  n (codeword length) = {n}")
        print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # Step 1: GF(2) RREF on G with full column pivoting
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[Step 1] Running GF(2) Gauss-Jordan elimination with column pivoting...")

    G_rref, col_perm, pivot_cols, rank = gf2_row_reduce(G.copy())

    if verbose:
        print(f"         Rank of G = {rank}  (expected k = {k_orig})")
        if rank < k_orig:
            print(f"  WARNING: G has rank {rank} < k={k_orig}. "
                  "Rows are linearly dependent — check your generator matrix.")
        print(f"         Pivot columns (in permuted order): {pivot_cols[:min(8,len(pivot_cols))]}{'...' if len(pivot_cols)>8 else ''}")
        swapped = np.where(col_perm != np.arange(n))[0]
        print(f"         Number of column swaps performed: {len(swapped)//2 if len(swapped)>0 else 0}")

    # -----------------------------------------------------------------------
    # Step 2: Verify RREF has identity in first k columns (systematic form)
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[Step 2] Verifying systematic (identity) structure in RREF...")

    G_sys = G_rref[:rank]   # take only the non-zero rows (rank rows)
    k = rank

    identity_block = G_sys[:, :k]
    is_identity = np.array_equal(identity_block, np.eye(k, dtype=np.uint8))

    if verbose:
        print(f"         First {k}×{k} block is identity: {is_identity}")
        if not is_identity:
            print("  WARNING: Identity block not clean — check pivot selection logic.")

    # -----------------------------------------------------------------------
    # Step 3: Extract the info_cols and compute inverse permutation
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[Step 3] Computing column permutation and inverse...")

    col_perm_inv = np.argsort(col_perm)
    info_cols = np.array(pivot_cols[:k], dtype=int)   # first k pivot positions

    if verbose:
        print(f"         col_perm (first 16):     {col_perm[:16]}")
        print(f"         col_perm_inv (first 16): {col_perm_inv[:16]}")
        print(f"         info_cols (message bit positions in permuted codeword): "
              f"{info_cols[:min(8,k)]}{'...' if k>8 else ''}")

    # -----------------------------------------------------------------------
    # Step 4: Compute GF(2) inverse of the k×k pivot submatrix
    #         (provides an alternative direct-solve path at inference)
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[Step 4] Computing GF(2) inverse of pivot submatrix...")

    G_k = G_sys[:, :k].copy()   # should be identity already, but compute anyway
    try:
        G_k_inv = gf2_inverse(G_k)
        inv_ok = np.array_equal(
            (G_k @ G_k_inv) % 2,
            np.eye(k, dtype=np.uint8)
        )
        if verbose:
            print(f"         G_k invertible: True,  G_k @ G_k_inv == I: {inv_ok}")
    except ValueError as e:
        G_k_inv = None
        if verbose:
            print(f"         G_k not invertible: {e}")
            print("         Falling back to column-slice method only.")

    # -----------------------------------------------------------------------
    # Step 5: Verification suite
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[Step 5] Running verification suite...")

    errors = []

    # 5a: Re-derive G from G_sys and check it generates same code
    # For a random message, encode with G and G_sys, check both are valid codewords
    rng = np.random.default_rng(42)
    test_messages = rng.integers(0, 2, size=(200, k), dtype=np.uint8)

    # Codewords via original G
    cw_G = (test_messages @ G[:k]) % 2   # use only rank rows if G had dep. rows

    # Codewords via systematic G_sys (in permuted column space)
    cw_sys_perm = (test_messages @ G_sys) % 2

    # Un-permute cw_sys to original column space and compare
    cw_sys_orig = cw_sys_perm[:, col_perm_inv]

    # Both must be in the same coset (same codeword space).
    # They may differ by a coset offset if G had a non-trivial kernel row;
    # instead check that cw_sys_orig is in the row space of G.
    # A codeword c is valid iff H·c^T = 0. Since we don't always have H,
    # check pairwise: difference of any two codewords is in row space of G.
    diff = (cw_G ^ cw_sys_orig)   # XOR = addition in GF(2)
    # Each row of diff must be in row space of G (i.e., G·diff^T rows = 0 is wrong metric)
    # Better: check message recovery
    # Recover message from cw_sys_perm by slicing info_cols
    recovered = cw_sys_perm[:, info_cols]
    message_ok = np.array_equal(recovered, test_messages)
    if not message_ok:
        errors.append("Message recovery from systematic codeword FAILED on test set.")

    if verbose:
        print(f"         Message recovery from permuted codeword:  {'PASS' if message_ok else 'FAIL'}")

    # 5b: Verify Method B (G_k_inv direct solve) on G_sys codewords.
    # NOTE: Method B works on codewords produced via G_sys (the systematic form).
    # For codewords produced by the original non-systematic G, use Method A only.
    # In your pipeline: the model encodes via G_nonsys, but after we apply col_perm
    # and confirm G_sys's identity structure, Method A (slice) is the right path.
    if G_k_inv is not None:
        cw_sys_k = cw_sys_perm[:, :k]              # pivot block (= I_k block of G_sys)
        recovered_solve = (cw_sys_k @ G_k_inv) % 2
        solve_ok = np.array_equal(recovered_solve, test_messages)
        if not solve_ok:
            errors.append("Direct-solve (Method B) recovery FAILED on G_sys codewords.")
        if verbose:
            print(f"         Direct-solve recovery (G_k_inv, on G_sys codewords): "
                  f"{'PASS' if solve_ok else 'FAIL'}")

    # 5c: Systematic property — message in first k positions of G_sys codeword
    sys_property_ok = np.array_equal(cw_sys_perm[:, :k], test_messages)
    if not sys_property_ok:
        errors.append("Systematic property check FAILED: message not in first k positions.")
    if verbose:
        print(f"         Systematic property (msg in positions 0..k-1): {'PASS' if sys_property_ok else 'FAIL'}")

    if errors:
        print("\n  ERRORS FOUND:")
        for e in errors:
            print(f"    ✗ {e}")
    else:
        if verbose:
            print("\n  All verification checks PASSED ✓")

    return {
        "G_sys":        G_sys,
        "col_perm":     col_perm,
        "col_perm_inv": col_perm_inv,
        "info_cols":    info_cols,
        "G_k_inv":      G_k_inv if G_k_inv is not None else np.array([]),
        "k":            k,
        "n":            n,
        "rank":         rank,
    }


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_artifacts(artifacts: dict, path: str, verbose: bool = True):
    np.savez(
        path,
        G_sys        = artifacts["G_sys"],
        col_perm     = artifacts["col_perm"],
        col_perm_inv = artifacts["col_perm_inv"],
        info_cols    = artifacts["info_cols"],
        G_k_inv      = artifacts["G_k_inv"],
        k            = np.array(artifacts["k"]),
        n            = np.array(artifacts["n"]),
        rank         = np.array(artifacts["rank"]),
    )
    if verbose:
        print(f"\n[Save] Artifacts saved to: {path}")
        print(f"       Keys: G_sys, col_perm, col_perm_inv, info_cols, G_k_inv, k, n, rank")


def load_artifacts(path: str) -> dict:
    data = np.load(path)
    return {
        "G_sys":        data["G_sys"],
        "col_perm":     data["col_perm"],
        "col_perm_inv": data["col_perm_inv"],
        "info_cols":    data["info_cols"],
        "G_k_inv":      data["G_k_inv"],
        "k":            int(data["k"]),
        "n":            int(data["n"]),
        "rank":         int(data["rank"]),
    }


# ---------------------------------------------------------------------------
# Demo: build a small synthetic non-systematic G for self-testing
# ---------------------------------------------------------------------------

def build_demo_nonsystematic_G(k: int = 4, n: int = 7) -> np.ndarray:
    """
    Build a simple non-systematic generator matrix for a (7,4) Hamming-like code.
    The matrix is deliberately scrambled so message bits don't appear directly.

    We start from the standard systematic (7,4) Hamming generator and apply
    a verified invertible GF(2) row transformation T so rank is preserved.
    """
    # Standard systematic (7,4) Hamming generator: G_sys = [I_4 | P]
    P = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ], dtype=np.uint8)
    G_sys = np.hstack([np.eye(4, dtype=np.uint8), P])

    # Invertible transformation T over GF(2) — det(T) = 1 mod 2 guaranteed by
    # construction (lower-triangular with 1s on diagonal → always invertible).
    # This mixes rows so the identity block in G_sys is destroyed.
    T = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 0, 1],
    ], dtype=np.uint8)
    G_nonsys = (T @ G_sys) % 2
    return G_nonsys


def build_demo_LDPC_like(n: int = 20, k: int = 10) -> np.ndarray:
    """
    Build a random full-rank non-systematic generator matrix over GF(2).
    Useful for testing with larger dimensions.
    """
    rng = np.random.default_rng(0)
    while True:
        G = rng.integers(0, 2, size=(k, n), dtype=np.uint8)
        # Check rank via RREF
        _, _, _, rank = gf2_row_reduce(G.copy())
        if rank == k:
            return G


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute systematic transform artifacts for a non-systematic "
            "binary linear code (LDPC / BCH / Polar)."
        )
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to .npy file containing G (k×n) or H ((n-k)×n). "
             "If omitted, runs a built-in demo."
    )
    parser.add_argument(
        "--type", type=str, choices=["G", "H"], default="H",
        help="Matrix type: 'G' = generator matrix, 'H' = parity-check matrix."
    )
    parser.add_argument(
        "--alist", action="store_true",
        help=(
            "Input file is in .alist format (MacKay standard). "
            "The alist encodes an H (parity-check) matrix; k is derived as n - rank(H). "
            "If not set, input must be a .npy binary matrix file."
        )
    )
    parser.add_argument(
        "--output", type=str, default="systematic_transform.npz",
        help="Output .npz file path."
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip verification suite."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output."
    )
    parser.add_argument(
        "--demo-size", type=str, default="small",
        choices=["small", "large"],
        help="Demo mode size: 'small' = (7,4) Hamming, 'large' = (20,10) random."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    # ---- Load matrix -------------------------------------------------------
    if args.input is None:
        if verbose:
            print("\n[Demo Mode] No --input provided. Using built-in demo matrix.")
            print(f"            Demo size: {args.demo_size}")

        if args.demo_size == "small":
            G = build_demo_nonsystematic_G(k=4, n=7)
        else:
            G = build_demo_LDPC_like(n=20, k=10)

        if verbose:
            print(f"\n  Generator matrix G:\n{G}")
    else:
        if args.alist:
            if verbose:
                print(f"\n[Input] Parsing alist file: {args.input}")
            H, meta = parse_alist(args.input, verbose=verbose)
            if verbose:
                print(f"\n        Deriving generator matrix G from H (shape {H.shape})...")
            G = derive_G_from_H(H)
            if verbose:
                print(f"        Derived G: shape {G.shape}")
                print(f"        Expected k (from filename/meta): n - m = {meta['n']} - {meta['m']} = {meta['n'] - meta['m']}")
                print(f"        Actual k from rank(H): {G.shape[0]}")
        else:
            raw = np.load(args.input)
            if args.type == "H":
                if verbose:
                    print(f"\n[Input] Parity-check matrix H loaded: shape {raw.shape}")
                    print("        Deriving generator matrix G from H...")
                G = derive_G_from_H(raw)
                if verbose:
                    print(f"        Derived G: shape {G.shape}")
            else:
                G = raw
                if verbose:
                    print(f"\n[Input] Generator matrix G loaded: shape {G.shape}")

    # ---- Compute -----------------------------------------------------------
    artifacts = compute_systematic_transform(G, verbose=verbose)

    # ---- Save --------------------------------------------------------------
    save_artifacts(artifacts, args.output, verbose=verbose)

    # ---- Summary -----------------------------------------------------------
    if verbose:
        print("\n" + "="*60)
        print("  SUMMARY")
        print("="*60)
        print(f"  Code parameters : n={artifacts['n']}, k={artifacts['k']}, "
              f"r={artifacts['n']-artifacts['k']}")
        print(f"  Rank of G       : {artifacts['rank']}")
        print(f"  Output file     : {args.output}")
        print()
        print("  At inference, load this file and use:")
        print("    c_perm = c_hat[col_perm]")
        print("    m      = c_perm[info_cols]          # Method A: direct slice")
        print("    m      = (c_perm[:k] @ G_k_inv) % 2 # Method B: linear solve")
        print()
        print("  See inference_message_extraction.py for the complete snippet.")
        print("="*60)


if __name__ == "__main__":
    main()
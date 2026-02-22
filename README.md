# Hybrid Mamba–Transformer Decoder for Error-Correcting Codes
This repository provides the official implementation of "Hybrid Mamba–Transformer Decoder for Error-Correcting Codes".

# Installation

The list of required packages are in `requirements.txt`

For optimal setup, you may want to check the original sources for installation for:
* [torch](https://pytorch.org/get-started/locally/)
* [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)
* [mamba-ssm](https://github.com/state-spaces/mamba)

# Running the code
The following example will train a model on the `LDPC(49,24)` code:\
`python training.py --code-hint LDPC_N49_K24`

For further command line options run:
`python training.py --help`

and see the `configuration.py` file for other configuration options.

# License

This repository is licensed under Apache 2.0. See `LICENSE`


# Acknowledgements

This repository is based on the [AECCT](https://github.com/mlaetvayn/AECCT) repository,\
and uses code from the [Mamba](https://github.com/state-spaces/mamba) repository.

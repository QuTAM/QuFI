# QuFI-Pennylane

This repository contains the Pennylane code adaptation work of the original Qiskit based QuFI (Quantum Fault Injector) proposed by Oliveira _et al._ in the paper [QuFI: a Quantum Fault Injector to Measure the Reliability of Qubits and Quantum Circuits](https://arxiv.org/abs/2203.07183).  

## Installation

All the dependencies of this project can be installed through the [requirements.txt](requirements.txt) file.

### Creating a virtual environment

You can create a virtual environment by installing [miniconda](https://docs.conda.io/en/latest/miniconda.html) and then running the following code:
```
conda create -n qufi_tutorial python=3.9
conda activate qufi_tutorial
pip3 install -r tutorial_requirements.txt
```


## Usage

Simply import qufi as a library and call its methods.  

```python
import qufi
from qufi import execute, save_results, IQFT
```

A simple usage example is available in [run_circuits.py](run_circuits.py).  
A load distributed campaign example is available in [launch_campaign.py](launch_campaign.py).  

## Contributing
Contribution to the project is welcome, however for opening issues please refer to the original repository by Oliveira.  

## Authors and Acknowledgement

The project has been developed by Marzio Vallero as part of the [Computer Engineering Masters Degree](https://didattica.polito.it/pls/portal30/sviluppo.offerta_formativa.corsi?p_sdu_cds=37:18&p_lang=EN) thesis work _Quantum Machine Learning Fault Injection_ under the guidance of professors B. Montrucchio, P. Rech and assistant researcher Edoardo Giusto, during the second semester of the academic year 2021/2022 at the [Politecnico di Torino University](https://www.polito.it/).

## License
For right to use, copyright and warranty of this software, refer to this project's [License](LICENSE).

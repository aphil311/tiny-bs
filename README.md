# Tiny-BS: Reference-less and transformer-less translation scoring

[![CLicense](https://img.shields.io/badge/License%20-%20MIT%20-%20%23ff6863?style=flat)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE) [![Python 3.10](https://img.shields.io/badge/Python%20-%203.10%20-%20?style=flat&logo=python&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Issues](https://img.shields.io/github/issues/aphil311/tiny-bs?style=flat&logo=github&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)


<!-- ABOUT THE PROJECT -->
## About The Project
Tiny-BS is a lightweight reference-less translation scoring model designed with low-compute environments in mind. We hope to achieve moderate to high correlation with BERTScore (and human judgement) while avoiding the use of transformers.

At the moment we only support the German-English language pair.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started


### Installation
1. Clone this repository with `git clone https://github.com/aphil311/tiny-bs.git`.
2. Install the dependencies with `pip install -r requirements.txt`.


### Usage 
1. You can run the model in demo mode with `python run_model {reference} {candidate}`.


<p align="right">(<a href="#readme-top">back to top</a>)</p> 



<!-- ROADMAP -->
## Roadmap

- [ ] **Build the model**
- [ ] **Training**
- [ ] **Validation**
- [ ] **Evaluation**


See the [open issues](https://github.com/aphil311/talos/issues) for a full list of proposed features (and known issues).



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
I would like to thank Professor Srinivas Bangalore as well as the TRA 301 TAs their for their invaluable guidance, feedback, and support.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
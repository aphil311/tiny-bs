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
   - You must downgrade to `pip < 24.1` with `pip install pip=24.0` to install `laser_encoders`.
   - You can upgrade after installing.


### Usage 
1. You can run the model in demo mode with `python run_model {reference} {candidate}`.
2. Log into huggingface with `huggingface-cli login`.
   - You can check your huggingface account status with `huggingface-cli whoami`.
   - If you do not have an account please make one at huggingface.com
3. Download the dataset by running the training code on a node with internet access.
   - To set the install/cache directory set the HF HOME environment variable with `export HF_HOME /your/scratch/dir`.
   - On Adriot that would be `/scratch/network/<netid>` and on Della that would be `/scratch/gpfs/<netid>`.
   - You can cancel the run before training. Future runs on nodes without internet access will find the dataset locally in your cache directory.
4. Run the training loop with `python train_score.py /your/output/dir`.
   - For details on optional parameters (dataset, epochs, debug) you can run `python train_score.py -h`


<p align="right">(<a href="#readme-top">back to top</a>)</p> 



<!-- ROADMAP -->
## Roadmap

- [X] **Build the model**
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

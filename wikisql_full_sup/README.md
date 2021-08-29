## Fully-Supervised WikiSQL

This directory contains code used for GraPPa experiments on WikiSQL. We employ [NL2SQL-RULE](https://github.com/guotong1988/NL2SQL-RULE) as our base model. Please follow their instructions to set up the environment.


### Data Download

Download the [processed WikiSQL data](https://drive.google.com/file/d/1YWviJdL6-BcpVQXEslVbGSdFbSJFcl73/view?usp=sharing), and unzip it.

### Run the Experiments

Please run `./run.sh`. Notice: you need to change `GRAPPA_PATH` in `run.sh` to use a different [GraPPa checkpoint](https://drive.google.com/file/d/1WHTrcCqNNxSdIJCpddBApKIDFKvYscPB/view?usp=sharing).

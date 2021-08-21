## Reprouduce Experiments on WikiSQL

### 1. Download data

You can download the processed data from [here](https://drive.google.com/file/d/1dXsx8WJolMDB2AE6QpZ62nDyqwm8tPiL/view), uncompress it to `processed/`. 
Your `processed` directory should look like:

```
processed
├── dev.pkl
├── productions.txt
├── sketch.actions
├── test.pkl
├── train.pkl
└── wikisql_glove_42B_minfreq_3.pkl
```

### 2. Adapt configurations

The main config file is `train_config/`, which contains a set of model configurations specified via instantiations of 
the Config object. We will use "2gpu_run_grappa_1080" for example. 
The results in the paper were obtained via the config "4gpu_run_grappa_V100".

To adapt the default config, you need to specify the path of grappa, by changing the respective variable "roberta_path".

## 3. Train the parser

    ./run_scripts/run_2gpu_grappa.sh

The run script have two variables: experiment id, model config name, e.g., "exp0", "2gpu_run_grappa_1080". 
They will be used to point to the corresponding checkpoints during evaluation.

### 4. Evaluation

Run the command:

    python eval_seq.py exp_id model_config_name checkpoint_path

For example:

    python eval_seq.py exp0 2gpu_run_grappa_1080 checkpoints/

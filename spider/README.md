## RAT-SQL for Spider
This directory contains code used for GraPPa experiments on Spider We employ [RAT-SQL](https://github.com/microsoft/rat-sql) as our base model.


### Data Download

Download the [processed Spider data](https://drive.google.com/file/d/1QIM1VswEvbYhUK3mP8f3ed0ZDGrCj1SK/view?usp=sharing), unzip it and put them under `data/spider_roberta` dir. The data directory structure looks like the following:

```
spider
 - data
  - spider_roberta
   - nl2code,output_from=true,fs=2,emb=bert,cvlink
    - ...
```

If you want to process the data by yourself, follow instructions on [RAT-SQL](https://github.com/microsoft/rat-sql).

``` bash
python run.py preprocess experiments/spider-bert-run.jsonnet # notice you first need to download the original spider data and put them in data/spider_roberta dir
```

### Build and Run the Docker Image

``` bash
docker build -t ratsql .
run -d --gpus 2 -it -v ~/grappa/spider:/app ratsql
```
Notice: use `--gpus number_of_gpus_to_use` to use GPUs in docker container.


### Step 3: Run the Experiments

Please follow the instructions on [the official RAT-SQL README](https://github.com/microsoft/rat-sql#step-3-run-the-experiments) to train and evaluate a model on Spider.

Notice: you need to change `GRAPPA_PATH` in `models/spider/spider_enc.py` to use a different [GraPPa checkpoint](https://drive.google.com/file/d/1WHTrcCqNNxSdIJCpddBApKIDFKvYscPB/view?usp=sharing).

Running training:
``` bash
python run.py train experiments/spider-bert-run.jsonnet --logdir checkpoints
```

You can download a [trained RAT-SQL+GraPPa checkpoint](https://drive.google.com/file/d/1f8YJ2A1l9tat8PtZ6MukTG19RH_PcT5-/view?usp=sharing), create a dir named `checkpoints/bs=8,lr=1.0e-04,bert_lr=1.0e-05,end_lr=0e0,att=1`, and put the checkpint under the new dir.

Running inteference:
``` bash
python run.py eval experiments/spider-bert-run.jsonnet --logdir checkpoints
```

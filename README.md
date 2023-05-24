# Needleman-Wunsch Attention

This is Pytorch implementation for ["Needleman-Wunsch Attention: A Novel Framework for Enhancing DNA Sequence Embedding"]()

## 1. Installation

Create virtual environment and install depenedencies

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

build Cython

```
cd utils
python setup.py build_ext --inplace
cd ..
```

Download data

- Place string DNA sequences into `./data/`
- each DNA string data should be separated with linebreak(`'\n'`)

```

```

[**_Qiita_**](https://cmi-workshop.readthedocs.io/en/latest/index.html)
can be downloaded here.

---

## 2. Preprocessing

- `generate_seq.py` divide(train_size/val_size/test_size) and save sequences into pkl file.
- `generate_nw_dataset.py` generates `(sequences, edit_distance, nw_matrix)` from pkl generated in the previous step with index and gap.

```
python -m preprocessing.generate_seq --out_dir ./data/qiita/seq.pkl --train_size 7000 --val_size 700 --test_size 1500 --src_dir ./data/qiita.txt
python -m preprocessing.generate_nw_dataset --out_dir ./data/qiita --src_dir ./data/qiita/seq.pkl --key train --index 0 --gap 350
```

**_Note that_**

- Sequence should have `out_dir` with path specified with filename e.g. `./data/sampledataset/seq.pkl`
- Whole dataset `out_dir` should end with directory. e.g. `./data/sampledataset`

Feeding arg `--num_attn_data` saves small portion of datas from test set to `attn.pkl`
Run this code to plot attention map (Figure 4. from the paper)

```
python -m preprocessing.generate_nw_dataset --out_dir ./data/qiita --src_dir ./data/qiita/seq.pkl --key test --index 0 --gap 300 --num_attn_data 10
```

---

## 3. Train

### 3.1 Baseline

- CNN (NeuroSEED)

```
python train.py --task vanilla --cuda 0 --seed 405 --data_path ./data/qiita --encoder cnn --distance hyperbolic --scaling True --enc_d_model 64 --n_enc_layers 4 --read_layers 1 --d_embedding 128 --layer_norm True --ed_loss mse --epochs 150 --ed_lr 0.01 --lr_warmup_step 10 --batch_size 2048 --pad_masking False --homologous_K 40 --print_every 5 --plot True
```

- Transformer (NeuroSEED)

```
python train.py --task vanilla --cuda 0 --seed 405 --data_path ./data/qiita --encoder trans --distance hyperbolic --scaling True --segment_size 4 --enc_nhead 2 --enc_d_model 16 --enc_d_ff 16 --n_enc_layers 2 --read_layers 1 --d_embedding 128 --layer_norm True --ed_loss mse --epochs 150 --ed_lr 0.01 --lr_warmup_step 10 --batch_size 2048 --pad_masking False --homologous_K 40 --print_every 5 --plot True
```

- CNN chi-square (DSEE)

```
python train.py --task vanilla --cuda 0 --seed 405 --data_path ./data/qiita --encoder cnn --distance square --scaling False --enc_d_model 64 --n_enc_layers 5 --read_layers 1 --d_embedding 128 --layer_norm True --ed_loss chi2 --epochs 150 --ed_lr 0.01 --lr_warmup_step 10 --batch_size 1024 --pad_masking False --homologous_K 40 --print_every 5 --plot True
```

### 3.2 Needleman-Wunsch Attention

- CNN + NWA

```
python train.py --task ednw --cuda 0 --seed 405 --data_path ./data/qiita --encoder cnn --distance hyperbolic --scaling True --enc_d_model 64 --n_enc_layers 4 --read_layers 1 --d_embedding 128 --dec_nhead 4 --dec_d_model 64 --dec_d_ff 1024 --n_dec_layers 2 --layer_norm True --ed_loss mse --nw_loss jsd --mat_lambda 1e-5 --epochs 80 --ed_lr 0.01 --nw_lr 0.001 --lr_warmup_step 10 --batch_size 1024 --pad_masking False --homologous_K 40 --print_every 5 --plot True
```

- Transformer + NWA

```
python train.py --task ednw --cuda 0 --seed 405 --data_path ./data/qiita --encoder trans --distance hyperbolic --scaling True --segment_size 4 --enc_nhead 2 --enc_d_model 16 --enc_d_ff 16 --n_enc_layers 2 --read_layers 1 --d_embedding 128 --layer_norm True --dec_nhead 4 --dec_d_model 64 --dec_d_ff 1024 --n_dec_layers 2 --ed_loss mse --nw_loss jsd --mat_lambda 1e-5 --epochs 80 --ed_lr 0.01 --nw_lr 0.001 --lr_warmup_step 10 --batch_size 1024 --pad_masking False --homologous_K 40 --print_every 5 --plot True
```

CNN chi-square + NWA

```
python train.py --task ednw --cuda 0 --seed 405 --data_path ./data/qiita --encoder cnn --distance square --scaling False --enc_d_model 64 --n_enc_layers 5 --read_layers 1 --d_embedding 128 --dec_nhead 4 --dec_d_model 64 --dec_d_ff 1024 --n_dec_layers 2 --layer_norm True --ed_loss chi2 --nw_loss jsd --mat_lambda 1e-6 --epochs 80 --ed_lr 0.01 --nw_lr 0.001 --lr_warmup_step 10 --batch_size 1024 --pad_masking False --homologous_K 40 --print_every 5 --plot True
```

---

## Validation

Test saved model

feed args corresponding to the saved model path.

```
python val.py --task ednw --cuda 0 --data_path ./data/qiita --encoder cnn --encoder_path ./path/to/enc_ckpt* --decoder_path ./decoder_path ./path/to/dec_ckpt* --ed_loss mse --nw_loss jsd --mat_lambda 1e-5 --batch_size 1024 --distance hyperbolic --homologous_K 40 --plot True
```

- model checkpoints will be saved in `./model/params/task*/`
- input filename of the model to use

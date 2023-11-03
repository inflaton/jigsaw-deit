# Code for cs712

## Run fine-tune exps
I give template config in `./exps/dgx/jigsaw_base_p56_336_f101_shuffle_in1ke80fte100_e50_1e-3_1024.sh`

Please run `bash ./exps/dgx/jigsaw_base_p56_336_f101_shuffle_in1ke80fte100_e50_1e-3_1024.sh` to run the exps.

Put your cs712 train data and test data under `./data`, each image is of size 336*336, and put into category subfolder:
```bash
./data/cs/
├── train
│   ├── 0
│   │   ├── 83.jpg
│   │   ├── ...
│   │   └── 2703.jpg
|   ├── ...
│   └── 49
└── val
    ├── 0
    │   ├── 0.jpg
    │   ├── ...
    │   └── 2606.jpg
    ├── ...
    └── 49
```

Key items need to change: epochs, batch size, learning rate, **freeze backbone or not (by --freeze)**, **classifier head** and **pretrained backbones (by --finetune)**.

When you try to change classifierhead, simly modify `./models_jigsaw.py/JigsawClassifierHead(nn.Module).__init__()`.
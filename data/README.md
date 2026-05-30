# Data


```text
data/sample_images/
```

Expected detector dataset format for local training:

```text
data/detector/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Expected classifier dataset format:

```text
data/classifier/
├── train/
│   ├── bad/
│   └── good/
├── val/
│   ├── bad/
│   └── good/
└── test/
    ├── bad/
    └── good/
```

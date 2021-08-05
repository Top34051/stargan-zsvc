# StarGAN-ZSVC: Unofficial PyTorch Implementation

This repository is an unofficial PyTorch implementation of [StarGAN-ZSVC](https://arxiv.org/pdf/2106.00043.pdf) by Matthew Baas and Herman Kamper. This repository provides both model architectures and the code to inference or train them. 

One of the StarGAN-ZSVC advantages is that it works on zero-shot settings and can be trained on unparallel audio data (different audio content by different speakers). Also, the model inference time is real-time or faster.

![](https://i.imgur.com/4cGK4UM.png)

**Disclaimer**: I implement this repository for educational purpose only. All credits go to the original authors. Also, it may contains different details as described in the paper. If there is a room for improvement, please feel free to contact me.

## Set up
```bash
git clone git@github.com:Top34051/stargan-zsvc.git
cd stargan-zsvc
conda env create -f environment.yml
conda activate stargan-zsvc
```

## Usage

### Voice conversion

Given two audio files, `source.wav` and `target.wav`, you can generate a new audio file with the same speaking content as in `source.wav` spoken by the speaker in `target.wav` as follow.

First, load my pretrained [model weights (`best.pt`)](https://drive.google.com/drive/folders/1IGZuuwbEvE-G68noOhlyPeZeAEFe2Lsc?usp=sharing) and put it in `checkpoints` folder.

Next, we need to embed both speaker identity.

```bash
python embed.py --path path_to_source.wav --name src
python embed.py --path path_to_target.wav --name trg
```
This will generate `src.npy` and `trg.npy`, the source and target speaker embeddings.

To perform voice conversion,

```bash
python convert.py \
  --audio_path path_to_source.wav \
  --src_id src \
  --trg_id trg  
```

That's it! :tada: You can check out the result at `results/output.wav`.

### Training

To train the model, you have to download and preprocess the dataset first. Since your data might be different from mine, I recommend you to read and fix the logic I used in `preprocess.py` (the dataset I used is [here](https://github.com/nii-yamagishilab/VCC2020-database)). 

The fixed size utterances from each speaker will be extracted, resampled to 22,050 Hz, and converted to Mel-spectrogram with window and hop length of size 1024 and 256. This will preprocess the speaker embeddings as well, so that you don't have to embed them one-by-one.

The processed dataset will look like this
```bash
data/
    train/
        spk1.npy # contains N samples of (80, 128) mel-spectrogram
        spk2.npy
        ...
    test/
        spk1.npy
        spk2.npy
        ...
        
embeddings/
    spk1.npy # a (256, ) speaker embedding vector
    spk2.npy
    ...
```

You can customize some of the training hyperparameters or select resuming checkpoint in `config.json`. Finally, train the models by

```bash
python main.py \ 
  --config_file config.json 
  --num_epoch 3000
```

You will now see new checkpoint pops up in the `checkpoints` folder. 

Please check out my code and modify them for improvement. Have fun training! :v:
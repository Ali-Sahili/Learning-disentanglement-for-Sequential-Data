## TIMIT Dataset

#### Introduction
The TIMIT corpus of read speech is designed to provide speech data for acoustic-phonetic studies and for the development and evaluation of automatic speech recognition systems. TIMIT contains broadband recordings of 630 speakers of eight major dialects of American English, each reading ten phonetically rich sentences. The TIMIT corpus includes time-aligned orthographic, phonetic and word transcriptions as well as a 16-bit, 16kHz speech waveform file for each utterance. 

> This code is to done to use read and preprocess Timit Dataset in order to be used in a Deep Learning task simply.

#### Setup
To download timit dataset, put into your terminal:
```
chmod +x download_dataset.sh
./download_dataset.sh
```

#### Requirements
The experiments were performed using Python 3.8.5 with the following Python packages:
- [numpy](http://www.numpy.org/) == 1.18.5
- [torch](https://pytorch.org/) == 1.5.1
- [matplotlib](https://pypi.org/project/matplotlib/) == 3.3.3
- [scipy](https://pypi.org/project/scipy/) == 1.5.4
- [sklearn](https://pypi.org/project/scikit-learn/) == 0.24.1
- [pandas](https://pypi.org/project/pandas/) == 0.25.3

#### Usage
To read visualize and store data as a dataloader object ready to used in a training process, you can use one of two preprocessing methods by choosing the convenient parameters:
```
python3 preprocess.py [--data_path DATA_PATH] [--mode {train,test}]
                      [--batch_size BATCH_SIZE] [--verbose VERBOSE]
                      [--nb_samples NB_SAMPLES] [--feature_len FEATURE_LEN]
                      [--win_len WIN_LEN] [--win_step WIN_STEP]

python3 preprocess_2.py [--data_path DATA_PATH] [--mode {train,test}]
                        [--mode_feat {mfcc,fbank}] [--batch_size BATCH_SIZE]
                        [--verbose VERBOSE] [--nb_samples NB_SAMPLES]
                        [--feature_len FEATURE_LEN] [--win_len WIN_LEN]
                        [--win_step WIN_STEP]
```

In addition, you can visualize the waveforms, extracted features, words, phonemes and other useful characteristics.
```
python3 visualize.py [--data_path DATA_PATH] [--scale_wav SCALE_WAV]
                     [--scale_words SCALE_WORDS]
                     [--scale_phones SCALE_PHONES] [--phn PHN] [--idx IDX]
                     [--save SAVE] [--out_path OUT_PATH] [--show SHOW]
```

#### Acknowledgements
- [timit_utils](https://github.com/colinator/timit_utils/tree/21592c362c6e441bc830f53b90d4a80af747456c) library.
- Some useful functions implemented by [zzw922cn](https://github.com/zzw922cn/Automatic_Speech_Recognition).

# METTS - PyTorch Implementation (wip)

A PyTorch implementation of [**METTS: Multilingual Emotional Text-to-Speech by Cross-Speaker and Cross-Lingual Emotion Transfer**](https://ieeexplore.ieee.org/document/10423864). 


![](./img/model.png)

# Updates
- 2025/6/30: First Initial Commit

## Dependencies
We recommend using a Python 3.10 environment.

You can install the Python dependencies with
```
pip install -r requirements.txt
```

## PreProcessing
```
python prepare_align.py
```
```
mfa
```
```
python preprocess_esd.py
```

## Training
```
python train.py
```

## Inference
```
python inference.py
```

## Todo

- [x] DelightfulTTS backbone
- [x] Multi-scale emoton modeling

- [ ] Information perturbation distanglement module
- [ ] VQ module

# References
- [ming024's FastSpeech2 implementation]https://github.com/ming024/FastSpeech2)
- [KinglittleQ's GST implementation]https://github.com/KinglittleQ/GST-Tacotron)
- [jaywalnut310's vits implementation]https://github.com/jaywalnut310/vits)

# Citation
```
@article{ren2020fastspeech2,
  title={Fastspeech 2: Fast and high-quality end-to-end text to speech},
  author={Ren, Yi and Hu, Chenxu and Tan, Xu and Qin, Tao and Zhao, Sheng and Zhao, Zhou and Liu, Tie-Yan},
  journal={arXiv preprint arXiv:2006.04558},
  year={2020}
}

@inproceedings{Wang2018GST,
  author={Wang, Yuxuan and Stanton, Daisy and Zhang, Yu and Ryan, RJ-Skerry and Battenberg, Eric and Shor, Joel and Xiao, Ying and Jia, Ye and Ren, Fei and Saurous, Rif A},
  title     = {Style tokens: Unsupervised style modeling, control and transfer in end-to-end speech synthesis},
  booktitle = {Proc. 35th Int. Conf. Mach. Learn.},
  pages     = {5167--5176},
  year      = {2018}
}

@inproceedings{Kim2021VITS,
  author    = {J. Kim and J. Kong and J. Son},
  title     = {Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech},
  booktitle = {Proc. 38th Int. Conf. Machine Learning},
  pages     = {5530--5540},
  year      = {2021},
}
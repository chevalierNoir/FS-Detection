# Fingerspelling Detection in American Sign Language (CVPR'2021)
This repo contains the code for paper: Fingerspelling Detection in American Sign Language [Arxiv Preprint](https://arxiv.org/abs/2104.01291).


## Requirements
* Pytorch 1.1.0
* Warp-CTC

Run `./setup.sh` to set up environment

## Usage:
0. Go to `src` directory: `cd src/`.

1. Download raw ASL videos of [ChicagoFSWild/ChicagoFSWild+](https://ttic.uchicago.edu/~klivescu/ChicagoFSWild.htm). Due to copyright issues, we cannot distribute the videos. The scripts to set up the dataset from scratch with URLs will be released soon. A sample dataset can be downloaded [here](https://drive.google.com/file/d/1KFmtiwZh7ehuAdiCQYoivtK__f9Rej81/view?usp=sharing) for debugging purpose. Uncompress it into `data/`. The data folder should look like:

```sh
   data/fswild/
   |-- dev.json
   |-- test.json
   |-- train.json
   |-- pose/
   `-- video/
```

2.  Training
```sh
./scripts/train.sh --help  # show arguments
./scripts/train.sh --step 1
```
See training script for details.

3. Evaluation
```sh
./scripts/eval.sh --help  # show arguments
./scripts/eval.sh --stage 2
```
See evaluation script for details. Note computing MSA/AP@Acc requires an off-the-shelf fingerspelling recognizer, which can be downloaded [here](https://drive.google.com/file/d/1M4hdgZNlEVqkRZW75ItWg0seReq7GGAq/view?usp=sharing).

## ToDo:
- [x] Code for fingerspelling detector
- [x] Code for evaluation
- [ ] Code for ASL data preparation from scratch


## Reference

    @inproceedings{shi2021fsdet,
      author = {Bowen Shi and Diane Brentari and Greg Shakhnarovich and Karen Livescu},
      title = {Fingerspelling Detection in American Sign Language},
      booktitle = {CVPR},
      year = {2021}
    }

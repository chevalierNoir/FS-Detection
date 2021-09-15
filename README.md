# Fingerspelling Detection in American Sign Language (CVPR'2021)
This repo contains the code for paper: Fingerspelling Detection in American Sign Language [[Arxiv Preprint]](https://arxiv.org/abs/2104.01291).


## Requirements
* Pytorch 1.1.0
* Warp-CTC
* Youtube-dl
* FFmpeg

Run `./setup.sh` to set up environment. Youtube-dl and FFmpeg are only required for data preparation.

## Usage:
0. Go to `src` directory: `cd src/`.

1. Download csv files of [ChicagoFSWild/ChicagoFSWild+](https://drive.google.com/file/d/1rDahGBMj0v-28mxyHJiZRwiFbhWF9PN_/view?usp=sharing). Use `preproc/pipeline.sh` to set up the dataset. For example, to set up the ChicagoFSWild in folder `data/fswild`, put the csv file `ChicagoFSwild.csv` in `data/fswild` and run the following command:

```sh
for step in {1..6};do ./preproc/pipeline.sh -d ./data/fswild/ -t ChicagoFSWild -s $step;done
```

It will generate the following subfolder `data/fswild/loader`, where the training and evaluation are based. 

```sh
   data/fswild/loader
   |-- dev.json
   |-- test.json
   |-- train.json
   `-- video
```

More concretely, the script will do the following: (1) downloading videos from Youtube. (2) creating csv files for downloaded videos. (3) resizing. (4) extracting optical flow. (5) generating label files. (6) spliting videos for data loading. In total, those steps take ~1 minute per 1-minute video clip on a common single 12-core CPU, where most time is consumed by step 1,3,4,6. The scripts for parallelizing those steps on slurm can be found in `scripts/slurm_fswild.sh` (for ChicagoFSWild) and `scripts/slurm_fswildplus.sh` (for ChicagoFSWild+). 

Note the above script only downloads and processes videos from youtube which are still available online. Thus the following experimental results can vary from original paper.

2.  Training
```sh
./scripts/train.sh  --help  # show arguments
./scripts/train.sh --data .data/fswild/loader/ --step 1
```
See training script for details.

3. Evaluation
```sh
./scripts/eval.sh --help  # show arguments
./scripts/eval.sh --data .data/fswild/loader/ --stage 1
```
See evaluation script for details. Note computing MSA/AP@Acc requires an off-the-shelf fingerspelling recognizer, which can be downloaded [here](https://drive.google.com/file/d/1HgweY-H24vM-5b67Tu2GBCWA4Pz1-bC-/view?usp=sharing).

## Pretrained models:
The fingerspelling detector trained on ChicagoFSWild+ (AP@0.5: 0.448) can be downloaded [here](https://drive.google.com/file/d/1zl1Pq54E55mrYQHE4qwZzVz-mBr3U0w5/view?usp=sharing).

## ToDo:
- [x] Code for fingerspelling detector
- [x] Code for evaluation
- [x] Code for ASL data preparation from scratch


## Reference

    @inproceedings{shi2021fsdet,
      author = {Bowen Shi and Diane Brentari and Greg Shakhnarovich and Karen Livescu},
      title = {Fingerspelling Detection in American Sign Language},
      booktitle = {CVPR},
      year = {2021}
    }

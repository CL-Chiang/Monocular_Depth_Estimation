# Monocular Depth Estimation: Empirical Study of Loss Functions based on AdaBins

Modified version of [Adabins: Depth Estimation using adaptive bins](https://arxiv.org/abs/2011.14141)

Official implementation is [here](https://github.com/shariqfarooq123/AdaBins).

## Download links (original implementation)
* You can download the pretrained models "AdaBins_nyu.pt" and "AdaBins_kitti.pt" from [here](https://drive.google.com/drive/folders/1nYyaQXOBjNdUJDsmJpcRpu6oE55aQoLA?usp=sharing)
* You can download the predicted depths in 16-bit format for NYU-Depth-v2 official test set and KITTI Eigen split test set [here](https://drive.google.com/drive/folders/1b3nfm8lqrvUjtYGmsqA5gptNQ8vPlzzS?usp=sharing)

## Inference
Move the downloaded weights to a directory of your choice (we will use "./pretrained/" here). 
Specify three variables in the `infer.py`:
1. `img_path`: Path to the image, e.g., `"test_imgs/room.jpg"`
2. `weight_path`: Path to the weight of model, e.g., `"checkpoints/Model5_3.pt"`
3. `n_bins`: Number of bins, which depends on the training setting. In original implementation, this is set to 256. In our implementation, this is set to **80**.

After assigning these variables in `infer.py`, just run `python infer.py` to get the results.

## Download links (our implementation)
Please use UMich email to download the following weight files.
* [Baseline](https://drive.google.com/file/d/108MdTNekRyshtns0xv7Op3T6w43PBmw1/view?usp=sharing)
* [Model1.1](https://drive.google.com/file/d/1qi5vCxhUshHYccfsrS4ueTuX90mmUHa3/view?usp=sharing)
* [Model1.2](https://drive.google.com/file/d/1PHta4GuZI1h5VjxNk95mFdO8xkkRdNJD/view?usp=sharing)
* [Model2  ](https://drive.google.com/file/d/1Xe4zw7mRRCc0OpdKk3APKZqIF_rLipRV/view?usp=sharing)
* [Model3  ](https://drive.google.com/file/d/1R_IFKFX6Ic03QEI7RaCCqgfJQAFydWdQ/view?usp=sharing)
* [Model4  ](https://drive.google.com/file/d/1EVNomiJnqhVRnfpoONux3ZpdJoWKo3s0/view?usp=sharing)
* [Model5.1](https://drive.google.com/file/d/1Da8S5xjLhdSrtz4az4simuE2kJov_0BW/view?usp=sharing)
* [Model5.2](https://drive.google.com/file/d/1NICxtdC6T1g4AeD6RA5fcrBd3cEE6_fd/view?usp=sharing)
* [Model5.3](https://drive.google.com/file/d/1JVwmzABUvZXLgEEXeJ0Un9UcN4pG6hIM/view?usp=sharing) 

## Experiment results

### Different combination of loss function 
|           |  w1 | w2| w3 |  w4 |  w5 |
|-----------|-----|---|----|-----|-----|
| Baseline  | 0.1 | 1 |  0 | 0   | 0   |
| Model1.1  | 0   | 0 |  1 | 1   | 1   |
| Model1.2  | 0   | 0 | 10 | 1   | 1   |
| Model2    | 0   | 1 |  0 | 0   | 0   |
| Model3    | 0   | 1 |  0 | 1   | 0   |
| Model4    | 0   | 1 |  0 | 0   | 1   |
| Model5.1  | 0   | 1 |  0 | 1   | 0.1 |
| Model5.2  | 0   | 1 |  0 | 0.1 | 1   |
| Model5.3  | 0   | 1 |  0 | 1   | 1   |

### Comparison between Baseline and our methods.
|           |   a1  |   a2  |   a3  |  REL  | SqRel |  RMSE |RMSElog| Log10 |
|-----------|-------|-------|-------|-------|-------|-------|-------|-------|
| Baseline  | 0.827 | 0.973 | 0.996 | 0.134 | 0.083 | 0.463 | 0.166 | 0.057 |
| Model 1.1 | 0.857 | 0.977 | 0.996 | 0.124 | 0.076 | 0.433 | 0.154 | 0.052 |
| Model 1.2 | 0.876 | 0.982 | 0.996 | 0.115 | 0.066 | 0.399 | 0.144 | 0.049 |
| Model 2   | 0.633 | 0.902 | 0.977 | 0.218 | 0.187 | 0.691 | 0.255 | 0.09  |
| Model 3   | 0.855 | 0.985 | 0.997 | 0.11  | 0.059 | 0.389 | 0.138 | 0.047 |
| Model 4   | 0.888 | 0.985 | 0.997 | 0.108 | 0.059 | 0.384 | 0.137 | 0.047 |
| Model 5.1 | 0.88  | 0.985 | 0.997 | 0.113 | 0.064 | 0.395 | 0.14  | 0.048 |
| Model 5.2 | 0.885 | 0.986 | 0.997 | 0.11  | 0.061 | 0.393 | 0.138 | 0.047 |
| Model 5.3 | 0.879 | 0.985 | 0.998 | 0.111 | 0.059 | 0.392 | 0.14  | 0.048 |


## Instruction:
### Install necessary packages
Create virtual enviroment (`VENV`) and install packages.
```
$ virtualenv VENV
$ source VENV/bin/activate
$ pip install torch torchvision torchaudio wandb matplotlib scipy
```
### Create secrets.py and enter wandb key
refers to https://wandb.ai/home
create `secrets.py` and enter `WANDB_API_KEY="xxxxx"`, `xxxxx` is you key.

### Check and run train.sh (on GreatLakes)
* Modify the virtual enviroment name.
* Take a look at the parameters below, especially for `w_grad` and `w_norm`.
* Enter `$ sbatch train.sh` to submit the job.

### Sync Wandb manually
`$ wandb sync wandb/offline-run-2021xxxx_xxxxxx-xxxxxxx`
Then you can go to Wandb website to check training status.

## TODO
* Update instruction for intallation
* Add evaluation
#!/bin/sh

#SBATCH --job-name="Evaluate"
#SBATCH --mail-type=BEGIN,END
#SBATCH --time=24:00:00
#SBATCH --account=eecs545w21_class

#SBATCH --mem=7gb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1


module load python
module load cuda
module load cudnn
source ../depth_estimation/bin/activate

python evaluate.py --n-bins 80 \
                   --gpu 0 \
                   --save_dir predictions_nyu \
                   --root ./ \
                   --dataset nyu \
                   --data_path dataset/sync/ \
                   --gt_path dataset/sync/ \
                   --filenames_file train_test_inputs/nyudepthv2_test_files_with_gt.txt \
                   --input_height 480 \
                   --input_width 640 \
                   --data_path_eval dataset/official_splits/test/ \
                   --gt_path_eval dataset/official_splits/test/ \
                   --filenames_file_eval train_test_inputs/nyudepthv2_test_files_with_gt.txt \
                   --checkpoint_path checkpoints/Model5_3.pt \
                   --min_depth_eval 1e-3 \
                   --max_depth_eval 10 \
                   --max_depth 10 \
                   --min_depth 1e-3 \
                   --eigen_crop \

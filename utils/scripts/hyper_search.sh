#!/usr/bin/bash

#SBATCH --job-name="Hyperparam Search"
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=course
#SBATCH --mail-user=a.ibrahimi@tum.de
#SBATCH --mail-type=ALL

# Output and Error logs for the job
#SBATCH --output=PATH/TO/YOUR/OUTPUT/FILE
#SBATCH --error=PATH/TO/YOUR/ERROR/FILE

module load PATH/TO/YOUR_ENVIRONMENT
source activate YOUR_ENVIRONMENT

# Define your Python virtual environment and script path
SCRIPT_PATH="/PATH/TO/YOUR/TRAINING/SCRIPT"

# Base path for saving model weights
BASE_WEIGHTS_PATH="/PATH/TO/YOUR/WEIGHTS/DIRECTORY/"

# Number of models to train
NUM_MODELS=50

# Loop through different hyperparameters and train models
for ((i=1; i<=$NUM_MODELS; i++))
do
    loss_func=$(python -c "from random import choice; print(choice(['MSE', 'NCC', 'MI']))")
    lambda_param=$(python -c "from scipy.stats import uniform; print(uniform.rvs(0.02, 0.08))")
    gamma_param=$(python -c "from scipy.stats import uniform; print(uniform.rvs(1, 3))")
    learning_rate=$(python -c "import random; print('%.7f' % (10 ** random.uniform(-5, -3)))")

    # Create weights path using hyperparameters
    weights_path="$BASE_WEIGHTS_PATH""_loss-$loss_func""_lambda-$lambda_param""_gamma-$gamma_param""_lr-$learning_rate"".h5"

    # Execute your Python script with chosen hyperparameters
    python $SCRIPT_PATH --nb_epochs 2000 --loss "$loss_func" --gamma_param $gamma_param --lambda_param $lambda_param --learning_rate $learning_rate --batch_size 16 --weights_path "$weights_path"
done
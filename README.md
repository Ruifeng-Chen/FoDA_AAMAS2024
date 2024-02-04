
# FoDA (Foresight Distribution Adjustment for Off-policy Reinforcement Learning)

FoDA (Foresight Distribution Adjustment) is a framework for off-policy reinforcement learning, where the goal is to use the post-update policy distribution to update the Q network to look ahead at the policy learning. This approach involves deriving the gradient of the visitation distribution with respect to the policy parameter and obtaining an explicit expression to approximate the post-update policy distribution.

## Environment Setup

Before running the code, ensure that your environment is properly set up. Use the following command to set up the environment:

```bash
conda env create -f foda_environment.yml
conda activate foda
```

For development mode, use:

```bash
pip install -e .
```

## Running the Code

To run the code, execute the following command:

```bash
python examples/mujoco/mujocofoda.py --seed 40 >terminal_output/cheetahrun40.log 2>&1 &
```

This command runs the `mujocofoda.py` script with a seed of `40` and logs the output to `terminal_output/cheetahrun40.log`.

## Acknowledgments

The code for FoDA is mainly built upon [Tianshou](https://github.com/thu-ml/tianshou). Special thanks to the contributors and maintainers of Tianshou for their excellent framework.



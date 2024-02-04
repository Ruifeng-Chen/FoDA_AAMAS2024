source activate usb


export CUDA_VISIBLE_DEVICES=1
python examples/mujoco/mujoco_sac.py --seed 50 >terminal_output/chrun50.log 2>&1 &
sleep 2 

export CUDA_VISIBLE_DEVICES=1
python examples/mujoco/mujoco_sac.py --seed 40 >terminal_output/chrun40.log 2>&1 &
sleep 2 

# export CUDA_VISIBLE_DEVICES=2
# python examples/mujoco/mujoco_sac.py --seed 30 >terminal_output/chrun30.log 2>&1 &
# sleep 2 

export CUDA_VISIBLE_DEVICES=0
python examples/mujoco/mujoco_sac.py --task walker-run --seed 20 >terminal_output/warun20.log 2>&1 &
sleep 2 

export CUDA_VISIBLE_DEVICES=0
python examples/mujoco/mujoco_sac.py --task walker-run --seed 10 >terminal_output/warun10.log 2>&1 &
sleep 2 

wait
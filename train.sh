source activate usb


export CUDA_VISIBLE_DEVICES=1
python examples/mujoco/mujocofoda.py --seed 40 >terminal_output/chrun40.log 2>&1 &
sleep 2 

wait


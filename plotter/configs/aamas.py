overwrite_args = {
    "mode": "joint",
    "algos": [        
        'FoDA',
        'LFIW',
        'SAC',
        'ReMERT',
        ],
    "tasks":{
        'walker-run': 2000000,
        'fish-swim': 2000000,
        'hopper-hop': 2000000,
        'hopper-stand': 2000000,
        'cheetah-run': 2000000,
        'humanoid-stand': 2000000,
        # 'reacher-hard': 2000000,
        # 'finger-spin': 2000000,
    },
    'key_mapping':{
        'performance/eval_return':'Eval Return',
        'test/reward':'Eval Return',
    },
    "x_axis_sci_limit": (6,6),
    "row": 2,
    "col": 3,
    "plot_interval": 2, # Int: select one datapoint from each "plot_interval", can smoothen and speed up plotting
    "smooth_length": 3, # Int: use convolution to smooth data, the value means the radius for smoothing
    "output_dir": "results/dmc/6tab_new"
}
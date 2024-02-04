overwrite_args = {
    "mode": "joint",
    "algos": [        
        'FoDA',
        'LFIW',
        'SAC',
        'ReMERN',
        ],
    "tasks":{
        'stick-push-v2': 1500000,
        'pick-place-v2': 1500000,
        'basketball-v2': 1500000,
        'peg-insert-side-v2': 1500000,
        'sweep-v2': 1500000,
        'hammer-v2': 1500000,
    },
    'key_mapping':{
        'performance/eval_success':'Eval Success',
        # 'performance/train_return':'Train Return'
    },
    "x_axis_sci_limit": (6,6),
    "row": 2,
    "col": 3,
    "output_dir": "results/dmc/metaworld6"
}
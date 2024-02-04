overwrite_args = {
    "mode": "joint",
    "algos": [        
        'eta=5',
        'eta=10',
        'eta=15'
        ],
    "tasks":{
        'cheetah-run': 2000000,
        'walker-run': 2000000,
    },
    'key_mapping':{
        'performance/eval_return':'Eval Return',
        # 'performance/train_return':'Train Return'
    },
    "x_axis_sci_limit": (6,6),
    "row": 1,
    "col": 2,
    "output_dir": "results/dmc/sensitivityfontsize"
}
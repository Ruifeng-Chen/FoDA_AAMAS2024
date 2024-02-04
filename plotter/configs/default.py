default_args = {
    "algos": [ #List: Name of algorithms to plot
        ],
    "tasks":{ # Dict, "env_name": max timestep. Name of tasks to plot
    },
    'key_mapping':{ # Dict, tb key name: name to appear on figure.
    },
    "aspect": 1.2, # Float: length/height ratio for each figure/subfigure
    "mode": "joint", # Str[single/joint]: "single" to plot one figure for each key, "joint": plot multiple subfigures in one single figure, 
    "plot_interval": 1, # Int: select one datapoint from each "plot_interval", can smoothen and speed up plotting
    "smooth_length": 3, # Int: use convolution to smooth data, the value means the radius for smoothing
    "x_axis_sci_limit": (0,0),  # (Int, Int): scientific notification control, see https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html
    "output_dir": "results/" # Str: directory to store the plot results
}
from cv2 import COLORMAP_WINTER
import seaborn as sns
import matplotlib.pyplot as plt
import click
import ast
import os
from plot_helper import create_log_pdframe, load_logs
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.ticker import MultipleLocator

matplotlib.use('Agg')

from tqdm import tqdm
from util import load_config 

sns.set_theme()


sns.set_theme(style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True, rc=None)

def single_plot(df, value_keys, config, output_dir):
    tasks = config['tasks']
    x_axis_sci_limit = config['x_axis_sci_limit']
    for task_name in tqdm(tasks):
        task_output_dir = os.path.join(output_dir, task_name)
        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir)
        task_df = df.loc[df['task_name']==task_name]
        for value_key in value_keys:
            value_df = task_df[['timestep', 'algo_name', value_key]].dropna()
            sns.lineplot(data = value_df,x = 'timestep', y = value_key, hue = 'algo_name')
            
            plt.ticklabel_format(axis="x", style="sci", scilimits=x_axis_sci_limit)
            output_path = os.path.join(task_output_dir, value_key + ".svg")
            plt.savefig(output_path)
            plt.clf()


COLOR = ["tab:red", "tab:blue", "tab:green", "tab:orange"]

def joint_plot(df, value_keys, config, output_dir):
    x_axis_sci_limit = config['x_axis_sci_limit']
    
    row, col = config["row"], config["col"]


    for value_key in tqdm(value_keys):
        fig, axarr = plt.subplots(figsize=(18, 10), nrows=row, ncols=col, sharex=False, sharey=False)  # Adjust the figure size and layout as needed
        plt.subplots_adjust(bottom= 0.22, wspace=0.4, hspace=0.4)
        # fig, axarr = plt.subplots(figsize=(18, 10), nrows=row, ncols=col, sharex=False, sharey=False)  # Adjust the figure size and layout as needed
        # plt.subplots_adjust(bottom = 0.15, wspace=0.4, hspace=0.4)
        
        for i, task_name in enumerate(df['task_name'].unique()):
            task_group = df[(df['task_name'] == task_name)][['timestep', 'algo_name', value_key]]
            row_i, col_i = i // col, i % col
            if row == 1:
                ax = axarr[col_i]
            else: 
                ax = axarr[row_i][col_i]
            for algo, color in zip(task_group.groupby('algo_name'), COLOR):
                algo_name, algo_group = algo[0], algo[1]
                algo_group.dropna(inplace = True)

                mean_values = algo_group.groupby('timestep')[value_key].mean()
                timestep = algo_group["timestep"].unique()

                ax.plot(timestep, mean_values, label=algo_name, color = color, linewidth = 1.5)
                # ax.xaxis.set_major_locator(MultipleLocator(500000))

                # Calculate standard deviation or error here
                std_values = algo_group.groupby('timestep')[value_key].std()

                ax.fill_between(timestep, mean_values - 0.9 * std_values, mean_values + 0.9 * std_values, alpha=0.1, facecolor = color)

            ax.set_xlabel('Timestep', fontsize = 14)
            ax.set_ylabel(value_key, fontsize = 14)
            ax.ticklabel_format(axis="x", style="sci", scilimits=x_axis_sci_limit)
            x_ticklabels = ax.get_xticklabels()
            for label in x_ticklabels:
                label.set_fontsize(12) 

            ax.set_xlim(left=0, right=2000000)
            ax.set_title(f'{task_name}', fontsize = 14)
            ax.grid(True)
        if row == 1:
            axarr[0].legend(ncol = 3, loc = "lower center", bbox_to_anchor = (1.15, -0.35),  prop={'size': 16})            
        else:
            axarr[0][0].legend(ncol = 4, loc = "lower center", bbox_to_anchor = (1.9, -1.9), prop={'size': 16})
        
        # plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"{value_key}.pdf")
        plt.savefig(output_path)
        plt.clf()


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,)
)
@click.argument("config-path", type = str, default = os.path.join("configs", "aamas.py")) # path to plot config
@click.argument("log-dir", type = str, default = os.path.join("dmc_logs"))  # path to log dir, the logs follows the default format of usb
@click.argument('args', nargs=-1)   # args in the config to overwrite 
def main(config_path, log_dir, args):
    # load config
    config = load_config(config_path, args)
    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # load logs
    print("loading logs")
    logs = load_logs(log_dir, config) 
    df, value_keys = create_log_pdframe(logs, config['key_mapping'])

    # plot 
    print("plotting")
    mode = config['mode']
    if mode == "single": # one figure for each task
        single_plot(df, value_keys, config, output_dir)
    elif mode == "joint": # plot tasks together in one figure
        joint_plot(df, value_keys, config, output_dir)

if __name__ == "__main__":
    main()
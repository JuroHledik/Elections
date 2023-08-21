from scipy import stats
import seaborn as sns
import datetime as datetime
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import copy
from pandasql import sqldf


def create_directory(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

def yearly_plot(df, figures_path, latex_figures_path, data_type, trns_type, figure_name, figure_title, y_variable, y_label):
    sns.set_theme(style="dark")
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

    # Plot each year's time series in its own facet
    g = sns.relplot(
        data=df,
        x="week", y=y_variable, col="year", hue="year",
        kind="line", palette=cmap, linewidth=4, zorder=5,
        col_wrap=2, height=2, aspect=1.5, legend=False, alpha=0.7
    )

    # Iterate over each subplot to customize further
    for year, ax in g.axes_dict.items():
        # Plot every year's time series in the background
        sns.lineplot(
            data=df, x="week", y=y_variable, units="year",
            estimator=None, color=".7", linewidth=1, ax=ax,
        )
        # Add the title as an annotation within the plot
        ax.text(.8, .85, year, transform=ax.transAxes, fontweight="bold")

    # Reduce the frequency of the x axis ticks
    # ax.set_xticks(ax.get_xticks()[::1])
    ax.set_xlim(1, 52)
    ax.set_xticks([6.5, 13, 19.5, 26, 32.5, 39, 45.5])
    ax.set_xticklabels(['Q1', '|', 'Q2', '|', 'Q3', '|', 'Q4'])

    # Tweak the supporting aspects of the plot
    g.set_titles("")
    g.set_axis_labels("", y_label, fontsize=10)
    g.tight_layout(rect=[0, 0.03, 1, 0.95])
    g.fig.suptitle(figure_title)

    figure = plt.gcf()
    figure.set_size_inches(8, 6)

    plt.savefig(figures_path + data_type + "_" +  trns_type + "_" + figure_name + '.png', dpi=150, bbox_inches='tight')
    plt.savefig(latex_figures_path + data_type + "_" + trns_type + "_" + figure_name + '.png', dpi=150, bbox_inches='tight')
    plt.close()

def scatter_plot_A(df, figures_path, latex_figures_path, data_type, trns_type, figure_name, figure_title, x_variable,x_label, y_variable, y_label, time_variable, time_variable_label, time_variable_description, point_size_variable, point_size_variable_description, middle_text):

    eps = 0.1

    variable_x_min = min(df[x_variable])
    variable_x_max = max(df[x_variable])
    variable_y_min = min(df[y_variable])
    variable_y_max = max(df[y_variable])

    x_min_temp = min(variable_x_min, variable_y_min)
    x_max_temp = max(variable_x_max, variable_y_max)
    y_min_temp = x_min_temp
    y_max_temp = x_max_temp

    x_min = x_min_temp - eps * (x_max_temp - x_min_temp)
    x_max = x_max_temp + eps * (x_max_temp - x_min_temp)
    y_min = x_min
    y_max = x_max

    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.5, 0.75])
    ax = fig.add_subplot(111, aspect=2)
    g = sns.relplot(
        data=df,
        x=x_variable, y=y_variable,
        hue=time_variable, size=point_size_variable,
        palette=cmap, sizes=(5, 100), aspect=1.5
    )
    g.ax.set_aspect('equal')
    for ax in g.fig.axes:
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))

    plt.plot([x_min,x_max],[y_min,y_max], '-', linewidth=0.5, color='#527095')
    plt.text((x_min+x_max)*0.5 - (x_max-x_min)*0.015, (y_min+y_max)*0.5 + (y_max-y_min)*0.015, '45°:  ' + middle_text, fontsize=12, ha="center", rotation=45, rotation_mode="anchor", color="#527095")

    # g.set(xscale="log", yscale="log")
    g.ax.xaxis.grid(True, "minor", linewidth=.25)
    g.ax.yaxis.grid(True, "minor", linewidth=.25)
    # g.despine(left=True, bottom=True)
    g.ax.set(xlabel=x_label, ylabel=y_label)
    g.fig.suptitle(figure_title, y=1)
    g._legend.remove()
    L = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    current_legend="NA"
    try:
        for t in L.get_texts():
            match = re.search(r'.*?\'(.*)\'.*', str(t)).group(1)
            if match==time_variable:
                current_legend="A"
                new_label = time_variable_description
                t.set_text(new_label)
            elif match == point_size_variable:
                current_legend = "B"
                new_label = point_size_variable_description
                t.set_text(new_label)
            else:
                if current_legend=="A":
                    new_label = df.loc[df[time_variable] == int(match), time_variable_label].values[0]
                    t.set_text(new_label)
    except:
        print('Exception in scatter_plot_A legend.')
    # plt.title(figure_title)
    plt.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig(figures_path + data_type + "_" +  trns_type + "_" + figure_name + '_A.png', dpi=150, bbox_inches='tight')
    plt.savefig(latex_figures_path + data_type + "_" + trns_type + "_" + figure_name + '_A.png', dpi=150, bbox_inches='tight')
    plt.close()











def scatter_plot_B(df, figures_path, latex_figures_path, data_type, trns_type, figure_name, figure_title, x_variable,x_label, y_variable, y_label, time_variable, time_variable_label, time_variable_description, point_size_variable, point_size_variable_description):
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.5, 0.75])
    ax = fig.add_subplot(111, aspect=2)
    g = sns.relplot(
        data=df,
        x=x_variable, y=y_variable,
        hue=time_variable, size=point_size_variable,
        palette=cmap, sizes=(10, 200), aspect=1.5
    )
    # g.set(xscale="log", yscale="log")
    for n in range(0,df.shape[0]-1):
        plt.plot([df[x_variable].iloc[n], df[x_variable].iloc[n+1]], [df[y_variable].iloc[n], df[y_variable].iloc[n+1]], '-', linewidth=0.5, color='#527095')


    g.ax.xaxis.grid(True, "minor", linewidth=.25)
    g.ax.yaxis.grid(True, "minor", linewidth=.25)
    g.ax.set(xlabel=x_label, ylabel=y_label)
    g.fig.suptitle(figure_title, y=1)
    g._legend.remove()
    L = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    current_legend="NA"
    try:
        for t in L.get_texts():
            match = re.search(r'.*?\'(.*)\'.*', str(t)).group(1)
            if match==time_variable:
                current_legend="A"
                new_label = time_variable_description
                t.set_text(new_label)
            elif match == point_size_variable:
                current_legend = "B"
                new_label = point_size_variable_description
                t.set_text(new_label)
            else:
                if current_legend=="A":
                    new_label = df.loc[df[time_variable] == int(match), time_variable_label].values[0]
                    t.set_text(new_label)
    except:
        print('Exception in scatter_plot_B legend.')
    # plt.title(figure_title)
    plt.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig(figures_path + data_type + "_" +  trns_type + "_" + figure_name + '_B.png', dpi=150, bbox_inches='tight')
    plt.savefig(latex_figures_path + data_type + "_" + trns_type + "_" + figure_name + '_B.png', dpi=150, bbox_inches='tight')
    plt.close()


def line_plot_A(df, figures_path, latex_figures_path, data_type, trns_type, figure_name, figure_title, x_variable, x_variable_label, x_label, y_variable, y_label):
    sns.set_theme(style="darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = sns.lineplot(data=df, x=x_variable, y=y_variable, )
    plt.style.use('seaborn')
    ax = df[y_variable].plot.area(alpha=0.7)

    ax.xaxis.grid(True, "minor", linewidth=.25)
    ax.yaxis.grid(True, "minor", linewidth=.25)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(0, (ax.get_xticks()).size):
        tick = ax.get_xticks()[i]
        try:
            new_label = df.loc[df[x_variable] == int(tick) + 1, x_variable_label].values[0]
        except:
            new_label = ""
        labels[i] = new_label
    ax.set_xticklabels(labels)
    plt.ylim(bottom=0)
    plt.title(figure_title)
    # current_legend="NA"
    # for t in L.get_texts():
    #     match = re.search(r'.*?\'(.*)\'.*', str(t)).group(1)
    #     if match==time_variable:
    #         current_legend="A"
    #         new_label = time_variable_description
    #         t.set_text(new_label)
    #     elif match == point_size_variable:
    #         current_legend = "B"
    #         new_label = point_size_variable_description
    #         t.set_text(new_label)
    #     else:
    #         if current_legend=="A":
    #             new_label = df.loc[df[time_variable] == int(match), time_variable_label].values[0]
    #             t.set_text(new_label)
    # plt.title(figure_title)
    plt.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig(figures_path + data_type + "_" +  trns_type + "_" + figure_name + '_A.png', dpi=150, bbox_inches='tight')
    plt.savefig(latex_figures_path + data_type + "_" + trns_type + "_" + figure_name + '_A.png', dpi=150, bbox_inches='tight')
    plt.close()



def line_plot_B(df, figures_path, latex_figures_path, data_type, trns_type, figure_name, figure_title, x_variable, x_variable_label, x_label, y_variable, y_label):
    sns.set_theme(style="darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sns.lineplot(data=df, x=x_variable, y=y_variable)
    # plt.style.use('seaborn')
    # ax = df[y_variable].plot.area(alpha=0.7)

    ax.xaxis.grid(True, "minor", linewidth=.25)
    ax.yaxis.grid(True, "minor", linewidth=.25)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(0, (ax.get_xticks()).size):
        tick = ax.get_xticks()[i]
        try:
            new_label = df.loc[df[x_variable] == int(tick) + 1, x_variable_label].values[0]
        except:
            new_label = ""
        labels[i] = new_label
    ax.set_xticklabels(labels)

    plt.title(figure_title)
    # current_legend="NA"
    # for t in L.get_texts():
    #     match = re.search(r'.*?\'(.*)\'.*', str(t)).group(1)
    #     if match==time_variable:
    #         current_legend="A"
    #         new_label = time_variable_description
    #         t.set_text(new_label)
    #     elif match == point_size_variable:
    #         current_legend = "B"
    #         new_label = point_size_variable_description
    #         t.set_text(new_label)
    #     else:
    #         if current_legend=="A":
    #             new_label = df.loc[df[time_variable] == int(match), time_variable_label].values[0]
    #             t.set_text(new_label)
    # plt.title(figure_title)
    plt.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig(figures_path + data_type + "_" +  trns_type + "_" + figure_name + '_B.png', dpi=150, bbox_inches='tight')
    plt.savefig(latex_figures_path + data_type + "_" + trns_type + "_" + figure_name + '_B.png', dpi=150, bbox_inches='tight')
    plt.close()

def bar_plot(df, figures_path, latex_figures_path, data_type, trns_type, figure_name, figure_title, x_variable, x_variable_label, x_label, y_variable, y_label):
    sns.set_theme(style="darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sns.barplot(data=df, x=x_variable, y=y_variable, color='b', alpha=0.7)
    plt.style.use('seaborn')

    ax.xaxis.grid(True, "minor", linewidth=.25)
    ax.yaxis.grid(True, "minor", linewidth=.25)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    # for i in range(0, (ax.get_xticks()).size):
    #     tick = ax.get_xticks()[i]
    #     try:
    #         new_label = df.loc[df[x_variable] == int(tick) + 1, x_variable_label].values[0]
    #     except:
    #         new_label = ""
    #     labels[i] = new_label
    # ax.set_xticklabels(labels)
    plt.ylim(bottom=0)
    plt.title(figure_title)
    plt.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig(figures_path + data_type + "_" + trns_type + "_" + figure_name + '_A.png', dpi=150,
                bbox_inches='tight')
    plt.savefig(latex_figures_path + data_type + "_" + trns_type + "_" + figure_name + '_A.png', dpi=150,
                bbox_inches='tight')
    plt.close()

def box_plot(df, figures_path, figure_name, figure_title, x_variable, x_variable_label, x_label, y_variable, y_label):
    sns.set_palette("tab10")
    sns.set_color_codes("muted")
    sns.set_theme(style="darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sns.boxplot(data=df, x=x_variable, y=y_variable, color='b', boxprops=dict(alpha=.7), whis=[1, 99], showfliers = False)
    # plt.style.use('seaborn')
    # ax = df[y_variable].plot.area(alpha=0.7)

    ax.xaxis.grid(True, "minor", linewidth=.25)
    ax.yaxis.grid(True, "minor", linewidth=.25)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(0, (ax.get_xticks()).size):
        tick = ax.get_xticks()[i]
        try:
            new_label = df.loc[df[x_variable] == int(tick) + 1, x_variable_label].values[0]
        except:
            new_label = ""
        labels[i] = new_label
    ax.set_xticklabels(labels)

    plt.title(figure_title)
    # current_legend="NA"
    # for t in L.get_texts():
    #     match = re.search(r'.*?\'(.*)\'.*', str(t)).group(1)
    #     if match==time_variable:
    #         current_legend="A"
    #         new_label = time_variable_description
    #         t.set_text(new_label)
    #     elif match == point_size_variable:
    #         current_legend = "B"
    #         new_label = point_size_variable_description
    #         t.set_text(new_label)
    #     else:
    #         if current_legend=="A":
    #             new_label = df.loc[df[time_variable] == int(match), time_variable_label].values[0]
    #             t.set_text(new_label)
    # plt.title(figure_title)
    plt.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig(figures_path + figure_name + '.png', dpi=150,
                bbox_inches='tight')
    plt.close()

def boxplot_vertical(df, figures_path, latex_figures_path, data_type, trns_type, figure_name, figure_title, x_variable, x_label, y_variable, y_label, log_scale, show_obs, figsize_x, figsize_y):
    sns.set_theme(style="darkgrid")
    sns.set_theme(style="ticks")

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(8, 6))
    if log_scale == True:
        ax.set_xscale("log")

    # Plot the orbital period with horizontal boxes
    df = df.sort_values(y_variable)
    sns.boxplot(x=x_variable, y=y_variable, data=df,
                    whis=[1, 99], width=.6, color='b', boxprops=dict(alpha=.7), showfliers = False)

    ax.xaxis.grid(True, "minor", linewidth=.25)
    ax.yaxis.grid(True, "minor", linewidth=.25)
    ax.set(xlabel=x_label, ylabel=y_label)
    f.suptitle(figure_title, y=1)

    # Add in points to show each observation
    if show_obs > 0:
        df = df.groupby(y_variable).apply(lambda s: s.sample(min(len(s), show_obs)))
        quantile_right = df[x_variable].quantile(0.99)
        quantile_left = df[x_variable].quantile(0.01)
        df = df[df[x_variable] < quantile_right]
        df = df[df[x_variable] > quantile_left]
        sns.stripplot(x=x_variable, y=y_variable, data=df,
                      size=4, color=".3", linewidth=0)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)

    plt.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(figsize_x, figsize_y)
    plt.savefig(figures_path + data_type + "_" + trns_type + "_" + figure_name + '.png', dpi=150,
                bbox_inches='tight')
    plt.savefig(latex_figures_path + data_type + "_" + trns_type + "_" + figure_name + '.png', dpi=150,
                bbox_inches='tight')
    plt.close()

def heatmap_continuous_A(df, figures_path, latex_figures_path, data_type, trns_type, figure_name, figure_title, x_variable, x_label, y_variable, y_label, quantiles):
    [q_x_min, q_x_max, q_y_min, q_y_max]=quantiles
    x_max = df[x_variable].quantile(q_x_max)
    x_min = df[x_variable].quantile(q_x_min)
    y_max = df[y_variable].quantile(q_y_max)
    y_min = df[y_variable].quantile(q_y_min)
    df = df.loc[df[x_variable] >= x_min,]
    df = df.loc[df[x_variable] <= x_max,]
    df = df.loc[df[y_variable] >= y_min,]
    df = df.loc[df[y_variable] <= y_max,]

    df = df.sample(n = min(df.shape[0],10000))

    sns.set_theme(style="white")

    g = sns.JointGrid(data=df, x=x_variable, y=y_variable, space=0, xlim=(x_min, x_max), ylim=(y_min, y_max))

    #It's not possible to draw the figure for number of contour levels that is too high, so we gradually lower it:
    success = False
    contour_levels = 100
    while success == False:
        if contour_levels < 3:
            break
        try:
            g.plot_joint(sns.kdeplot, fill=True, thresh=0, levels=contour_levels, cmap="rocket")
        except:
            print("Could not print the figure with " + str(contour_levels) + " levels, trying with half of that.")
            contour_levels = int(round(contour_levels / 2))
        else:
            print("Figure printed successfullly.")
            success = True

    g.ax_marg_x.hist(df[x_variable], color="#03051A", bins=np.arange(x_min, x_max))
    g.ax_marg_y.hist(df[y_variable], color="#03051A", orientation="horizontal", bins=100)
    #g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=int(x_max-x_min+1), ax=g.ax_marg_x)

    g.set_axis_labels(x_label, y_label)
    plt.gcf().subplots_adjust(bottom=.15)
    plt.gcf().subplots_adjust(left=.15)
    g.fig.suptitle(figure_title, y=1)
    figure = plt.gcf()
    figure.set_size_inches(8, 8)
    plt.savefig(figures_path + data_type + "_" + trns_type + "_" + figure_name + '_A.png', dpi=150,
                bbox_inches='tight')
    plt.savefig(latex_figures_path + data_type + "_" + trns_type + "_" + figure_name + '_A.png', dpi=150,
                bbox_inches='tight')
    plt.close()



def heatmap_continuous_B(df, figures_path, latex_figures_path, data_type, trns_type, figure_name, figure_title, x_variable, x_label, y_variable, y_label, quantiles):
    [q_x_min, q_x_max, q_y_min, q_y_max] = quantiles
    x_max = df[x_variable].quantile(q_x_max)
    x_min = df[x_variable].quantile(q_x_min)
    y_max = df[y_variable].quantile(q_y_max)
    y_min = df[y_variable].quantile(q_y_min)
    df = df.loc[df[x_variable] >= x_min,]
    df = df.loc[df[x_variable] <= x_max,]
    df = df.loc[df[y_variable] >= y_min,]
    df = df.loc[df[y_variable] <= y_max,]

    df = df.sample(n = min(df.shape[0],10000))

    # sns.set_theme(style="white")
    #
    # g = sns.JointGrid(data=df, x=x_variable, y=y_variable, space=0)
    # g.plot_joint(sns.kdeplot,
    #  fill=True,
    #  thresh=0, levels=100, cmap="rocket")
    # g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=25)
    #
    # g.set_axis_labels(x_label, y_label)
    # g.fig.suptitle(figure_title, y=1)

    sns.set_theme(style="whitegrid")
    f = plt.figure()
    ax = f.add_subplot(111)
    g = sns.JointGrid(df[x_variable], df[y_variable], xlim=(x_min, x_max), ylim=(y_min, y_max))

    # It's not possible to draw the figure for number of contour levels that is too high, so we gradually lower it:
    success = False
    contour_levels = 100
    while success == False:
        if contour_levels < 3:
            break
        try:
            g.plot_joint(sns.kdeplot, shade=True, cmap="Greys", n_levels=contour_levels)
        except:
            print("Could not print the figure with " + str(contour_levels) + " levels, trying with half of that.")
            contour_levels = int(round(contour_levels / 2))
        else:
            print("Figure printed successfullly.")
            success = True

    g.plot_joint(plt.scatter, color='#e74c3c', s=1.5)
    g.plot_marginals(sns.kdeplot, color="black", shade=True)
    g.ax_joint.collections[0].set_alpha(0)
    g.set_axis_labels(x_label,y_label)
    plt.gcf().subplots_adjust(bottom=.15)
    plt.gcf().subplots_adjust(left=.15)
    g.fig.suptitle(figure_title, y=1)
    figure = plt.gcf()
    figure.set_size_inches(8, 8)
    plt.savefig(figures_path + data_type + "_" + trns_type + "_" + figure_name + '_B.png', dpi=150,
    bbox_inches='tight')
    plt.savefig(latex_figures_path + data_type + "_" + trns_type + "_" + figure_name + '_B.png', dpi=150,
                bbox_inches='tight')
    plt.close()

def line_plot_numbers(df, figures_path, latex_figures_path, data_type, figure_name, figure_title, x_variable, x_variable_label, x_label, y_variable, y_label, line_labels):
    sns.set_theme(style="darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = sns.lineplot(data=df, x=x_variable, y=y_variable, )
    plt.style.use('seaborn')

    ax.xaxis.grid(True, "minor", linewidth=.25)
    ax.yaxis.grid(True, "minor", linewidth=.25)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    #Find the number of agents:
    N = sqldf("select count(distinct agent) as X from df")
    N=N.iloc[0]['X']

    agents = sqldf("select distinct agent from df")

    plt.title(figure_title)

    for agent_index in range(N):
        agent_number = agents.iloc[agent_index]['agent']
        df_temp = sqldf('select * from df where agent="' + str(agent_number) + '"')
        plt.plot(df_temp[x_variable], df_temp[y_variable], alpha=0.5, marker='B${}$'.format(agent_number), markersize=10, label=agent_number)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(0, (ax.get_xticks()).size):
        tick = ax.get_xticks()[i]
        try:
            new_label = df.loc[df[x_variable] == int(tick), x_variable_label].values[0]
        except:
            new_label = ""
        labels[i] = new_label
    ax.set_xticklabels(labels)

    figure = plt.gcf()
    figure.set_size_inches(8, 6)

    plt.savefig(figures_path + data_type + "_" + figure_name + '.png', dpi=150, bbox_inches='tight')
    plt.savefig(latex_figures_path + data_type + "_" + figure_name + '.png', dpi=150, bbox_inches='tight')
    plt.close()

def hist_plot(df, figures_path, figure_name, figure_title, x_variable, x_variable_label, x_label,mu,sigma):
    sns.set_palette("tab10")
    sns.set_color_codes("muted")
    sns.set_theme(style="darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = sns.displot(x=df[x_variable], color='b', binwidth=0.001)
    ax = sns.histplot(x=df[x_variable], color='b', bins=50)
    # plt.style.use('seaborn')
    # ax = df[y_variable].plot.area(alpha=0.7)

    # ax.xaxis.grid(True, "minor", linewidth=.25)
    # ax.yaxis.grid(True, "minor", linewidth=.25)

    xx = np.linspace(*ax.get_xlim(), 100)
    ax.plot(xx, stats.norm.pdf(xx, mu, np.sqrt(sigma)));

    plt.xlabel(x_label)

    plt.title(figure_title)
    # current_legend="NA"
    # for t in L.get_texts():
    #     match = re.search(r'.*?\'(.*)\'.*', str(t)).group(1)
    #     if match==time_variable:
    #         current_legend="A"
    #         new_label = time_variable_description
    #         t.set_text(new_label)
    #     elif match == point_size_variable:
    #         current_legend = "B"
    #         new_label = point_size_variable_description
    #         t.set_text(new_label)
    #     else:
    #         if current_legend=="A":
    #             new_label = df.loc[df[time_variable] == int(match), time_variable_label].values[0]
    #             t.set_text(new_label)
    # plt.title(figure_title)
    plt.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig(figures_path + figure_name + '.png', dpi=150,
                bbox_inches='tight')
    plt.close()
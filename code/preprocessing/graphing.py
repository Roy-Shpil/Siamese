import numpy as np
import matplotlib.pyplot as plt

def graph_histogram(x: np.ndarray, num_bins: int, show_mean: bool=True, show_std: bool=True, show_max: bool=True,
                    show_gaussian_fit: bool = True, title: str or None=None, is_log_x: bool = False,
                    is_log_y: bool = False, xlabel: str=None,
                    ylabel: str=None) -> tuple[float, float, tuple[float, float], list|np.ndarray]:
    """
    Graphing a histogram of the data
    :param x: input data
    :param num_bins: number of bins
    :param show_mean: if True, displays the mean on the graph
    :param show_std: if True, displays the standard deviation on the graph
    :param show_max: if True, displays the maximum value on the graph
    :param show_gaussian_fit: if True, displays the best gaussian fit
    :param title: title to display
    :return:
        - mu: mean
        - sigma: standard deviation
        - (x_max, y_max): maximum point
        - hist_vals: values of histogram bins
    """

    # Calculating mean and standard deviation
    mu=np.mean(x)
    sigma=np.std(x)

    # Adding title
    plt.figure()
    if is_log_x:
        plt.xscale('log')
    if is_log_y:
        plt.yscale('log')
    if title is not None:
        plt.title(title)
    else:
        plt.title('histogram')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # Calculating histogram
    hist_vals, bins, _ = plt.hist(x, bins=num_bins, label='histogram')
    x_max = bins[np.argmax(hist_vals)]
    y_max = np.max(hist_vals)

    # Adding the mean as a vertical line
    if show_mean:
        plt.plot(mu*np.ones(2), np.array([np.min(hist_vals), np.max(hist_vals)]), linewidth=2, label='mean')

    # Adding standard deviation as a horizontal line
    if show_std:
        plt.hlines(np.max(hist_vals)/2, mu - sigma, mu + sigma, linewidth=2, color='olive', label='standard deviation')
        cap_height = (np.max(hist_vals)-np.min(hist_vals))/18
        plt.vlines(x=mu-sigma, ymin=np.max(hist_vals)/2-cap_height/2, ymax=np.max(hist_vals)/2+cap_height/2, linewidth=2, color='olive')
        plt.vlines(x=mu+sigma, ymin=np.max(hist_vals)/2-cap_height/2, ymax=np.max(hist_vals)/2+cap_height/2, linewidth=2, color='olive')

    # Adding the maximum as an annotated point
    if show_max:
        x_offset = sigma/2
        plt.plot(x_max, y_max, marker='x', markersize=10, markeredgewidth=3, color='gold', label='max')
        plt.annotate(f'{(np.round(x_max, 2), np.round(y_max, 2))}', xy = (x_max+x_offset, y_max), color='gold')

    # Adding the gaussian fit graph
    if show_gaussian_fit:
        x_gauss = np.linspace(np.min(x), np.max(x))
        y_gauss = y_max*np.exp(-1/2*((x_gauss-x_max)/sigma)**2)
        plt.plot(x_gauss, y_gauss, linewidth=2, color='green', label='gaussian fit', ls='--')

    plt.legend()
    plt.grid()
    plt.show()

    return mu, sigma, (x_max, y_max), hist_vals

def graph_pie(sizes: list, labels: list, title: str = None, xlabel: str=None, ylabel: str=None) -> None:
    """
    Plots a pie chart.
    :param sizes: Input array (num_categories,)
    :param labels: label array (num_categories,)
    :param title: title to display
    """

    # Input array and label array must be of the same size!
    if len(sizes) != len(labels):
        raise ValueError('sizes and labels must have same length')

    # Converts to percentage
    sizes_pct = 100*np.array(sizes)/np.array(sizes).sum()

    # Creates labels
    labels_combined = [f'{labels[i]}: {format(sizes_pct[i], ".3f")}%' for i in range(len(labels))]
    plt.figure()

    # Adds title
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # Creates pie chart
    plt.pie(sizes_pct, labels=labels_combined, radius=1, explode = np.ones_like(sizes_pct) * 0.05)
    plt.show()

def graph_box(sizes: list, labels: list, title: str = None) -> None:
    """
    Plotting a box chart
    :param sizes: a list of samples, each belonging to a different category
    :param labels: category labels
    :param title: title to display
    """

    # Sample list and label list must be of the same length!
    if len(sizes) != len(labels):
        raise ValueError('sizes and labels must have same length')

    # Creating box plot
    plt.figure()
    plt.boxplot(sizes, labels=labels, showfliers=False)

    # Adding title
    if title is not None:
        plt.title(title)
    plt.show()

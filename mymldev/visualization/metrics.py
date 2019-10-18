import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as mt
from scipy import interp
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix():
    raise NotImplementedError


def plot_roc(
    y_true: np.array,
    y_probas: np.array,
    title: str = "ROC Curve",
    plot_micro: bool = False,
    plot_macro: bool = False,
    classes_to_plot: list = None,
    figsize: tuple = None,
    cmap: str = "gist_ncar",
    title_fontsize: str = "large",
    text_fontsize: str = "medium",
):
    """Plot ROC curve.

    Parameters
    ----------
    y_true : array_like, (n_samples,)
        Actual taget values.
    y_probas : array_like, (n_samples, n_classes)
        Predicted probabilities of each class.
    title : str
        Title for the ROC.
    plot_micro : bool, optional
        Plot micro averaged ROC curve (the default is False)
    plot_macro : bool, optional
        Plot macro averaged ROC curve (the default is False)
    classes_to_plot : list, optional
        Classes for which the ROC curve should be plotted.
        If the class doesn't exists it will be ignored.
        If ``None``, all classes will be plotted
        (the default is ``None``).
    figsize : tuple
        Size of the plot (the default is ``None``)
    cmap : str or `matplotlib.colors.Colormap` instance, optional
        Colormap used for plotting.
        https://matplotlib.org/tutorials/colors/colormaps.html
    title_fontsize : str or int, optional
        Use 'small', 'medium', 'large' or integer-values
        (the default is 'large')
    text_fontsize : str or int, optional
        Use 'small', 'medium', 'large' or integer-values
        (the default is 'medium')

    Returns
    -------
    ax : `matplotlib.axes.Axes` object
        The axes on which plot was drawn.

    References
    ----------
    .. [1] https://github.com/reiinakano/scikit-plot
    """
    classes = np.unique(y_true)
    if not classes_to_plot:
        classes_to_plot = classes
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(label=title, fontsize=title_fontsize)
    fpr_dict = {}
    tpr_dict = {}
    indices_to_plot = np.in1d(classes_to_plot, classes)
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = mt.roc_curve(y_true, y_probas[:, i], pos_label=classes[i])
        if to_plot:
            roc_auc = mt.auc(fpr_dict[i], tpr_dict[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(
                fpr_dict[i],
                tpr_dict[i],
                lw=2,
                color=color,
                label=f"ROC curve of class {classes[i]} (area = {roc_auc:.2f})",
            )
    if plot_micro:
        binarized_y_true = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            binarized_y_true = np.hstack((1 - binarized_y_true, binarized_y_true))
        fpr, tpr, _ = mt.roc_curve(binarized_y_true.ravel(), y_probas.ravel())
        roc_auc = mt.auc(tpr, fpr)
        ax.plot(
            fpr,
            tpr,
            label=f"micro-average ROC curve (area = {roc_auc:.2f})",
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )
    if plot_macro:
        # Compute macro-average ROC curve and it's area.
        # First aggregate all the false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i, _ in enumerate(classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i, _ in enumerate(classes):
            mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
        # Finally average it and compute AUC
        mean_tpr /= len(classes)
        roc_auc = mt.auc(all_fpr, mean_tpr)
        ax.plot(
            all_fpr,
            mean_tpr,
            label=f"macro-average ROC curve (area = {roc_auc:.2f})",
            color="navy",
            linestyle=":",
            linewidth=4,
        )
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05])
    ax.set_xlabel(f"False Positive Rate", fontsize=text_fontsize)
    ax.set_ylabel(f"True Positive Rate", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc="lower right", fontsize=text_fontsize)
    return ax


def plot_precision_recall():
    raise NotImplementedError


def plot_cumulative_gain():
    raise NotImplementedError


def plot_lift_curve():
    raise NotImplementedError


def plot_ks_statistic():
    raise NotImplementedError

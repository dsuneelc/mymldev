import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional
import sklearn.metrics as mt
from scipy import interp
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix():
    raise NotImplementedError


def plot_roc(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    labels: Optional[dict] = None,
    classes_to_plot: Optional[list] = None,
    plot_micro: Optional[bool] = False,
    plot_macro: Optional[bool] = False,
    title: str = "ROC Curve",
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: Optional[tuple] = None,
    cmap: Union[str, matplotlib.colors.Colormap] = "Blues",
    title_fontsize: Union[str, int] = "large",
    text_fontsize: Union[str, int] = "medium",
):
    """Plot ROC curve.

    Parameters
    ----------
    y_true : numpy.array, (n_samples,)
        Actual target values.
    y_probas : numpy.array, (n_samples, n_classes)
        Predicted probabilities of each class.
    labels: Optional[dict]
        labels for y.
    classes_to_plot : Optional[list]
        Classes for which the ROC curve should be plotted.
        If the class doesn't exists it will be ignored.
        If ``None``, all classes will be plotted
        (the default is ``None``).
    plot_micro : Optional[bool]
        Plot micro averaged ROC curve (the default is False)
    plot_macro : Optional[bool]
        Plot macro averaged ROC curve (the default is False)
    title : str
        Title for the ROC.
    ax: Optional[`matplotlib.axes.Axes`] object
        The axes on which plot was drawn.
    figsize : Optional[tuple]
        Size of the plot.
    cmap : Union[str, `matplotlib.colors.Colormap`]
        Colormap used for plotting.
        https://matplotlib.org/tutorials/colors/colormaps.html
    title_fontsize : Union[str, int]
        Use 'small', 'medium', 'large' or integer-values
        (the default is 'large')
    text_fontsize : Union[str, int]
        Use 'small', 'medium', 'large' or integer-values
        (the default is 'medium')

    Returns
    -------
    `matplotlib.axes.Axes` object
        The axes on which plot was drawn.

    References
    ----------
    .. [1] https://github.com/reiinakano/scikit-plot
    """
    classes = np.unique(y_true)
    if not classes_to_plot:
        classes_to_plot = classes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(label=title, fontsize=title_fontsize)
    fpr_dict = {}
    tpr_dict = {}
    indices_to_plot = np.isin(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = mt.roc_curve(
            y_true, y_probas[:, i], pos_label=classes[i]
        )
        if to_plot:
            roc_auc = mt.auc(fpr_dict[i], tpr_dict[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            class_name = labels[classes[i]] if labels else classes[i]
            ax.plot(
                fpr_dict[i],
                tpr_dict[i],
                lw=2,
                color=color,
                label=f"ROC curve of class {class_name} (area = {roc_auc:.2f})",
            )
    if plot_micro:
        binarized_y_true = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            binarized_y_true = np.hstack(
                (1 - binarized_y_true, binarized_y_true)
            )
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
        all_fpr = np.unique(
            np.concatenate([fpr_dict[i] for i, _ in enumerate(classes)])
        )
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


def plot_precision_recall(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    labels: Optional[dict] = None,
    classes_to_plot: Optional[list] = None,
    plot_micro: Optional[bool] = False,
    title: str = "Precision-Recall Curve",
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: Optional[tuple] = None,
    cmap: Union[str, matplotlib.colors.Colormap] = "Blues",
    title_fontsize: Union[str, int] = "large",
    text_fontsize: Union[str, int] = "medium",
):
    """Plots precision-recall curve.

    Parameters
    ----------
    y_true : numpy.array, (n_samples,)
        Actual target values.
    y_probas : numpy.array, (n_samples, n_classes)
        Predicted probabilities of each class.
    labels: Optional[dict] 
        labels for y, eg: {0: 'negative', 1: 'positive'}.
    classes_to_plot : Optional[list]
        Classes for which the Precision-Recall curve should be plotted.
        If the class doesn't exists it will be ignored.
        If ``None``, all classes will be plotted
        (the default is ``None``).
    plot_micro : Optional[bool]
        Plot micro averaged Precision-Recall curve (the default is False)
    title : str
        Title for the Precision-Recall curve.
    ax: Optional[matplotlib.axes.Axes]
        The axes on which plot was drawn.
    figsize : Optional[tuple]
        Size of the plot.
    cmap : Union[str, matplotlib.colors.Colormap]
        Colormap used for plotting.
        https://matplotlib.org/tutorials/colors/colormaps.html
    title_fontsize : Union[str, int]
        Use 'small', 'medium', 'large' or integer-values
        (the default is 'large')
    text_fontsize : Union[str, int]
        Use 'small', 'medium', 'large' or integer-values
        (the default is 'medium')

    Returns
    -------
    `matplotlib.axes.Axes` object
        The axes on which plot was drawn.

    References
    ----------
    .. [1] https://github.com/reiinakano/scikit-plot
    """
    classes = np.unique(y_true)
    if not classes_to_plot:
        classes_to_plot = classes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(label=title, fontsize=title_fontsize)
    binarized_y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        binarized_y_true = np.hstack((1 - binarized_y_true, binarized_y_true))
    indices_to_plot = np.isin(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        if to_plot:
            average_precision = mt.average_precision_score(
                binarized_y_true[:, i], y_probas[:, i]
            )
            precision, recall, _ = mt.precision_recall_curve(
                y_true, y_probas[:, i], pos_label=classes[i]
            )
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            class_name = labels[classes[i]] if labels else classes[i]
            ax.plot(
                recall,
                precision,
                lw=2,
                color=color,
                label=(
                    f"Precision-recall curve of class {class_name} "
                    f"(area = {average_precision:.2f})"
                ),
            )
    if plot_micro:
        precision, recall, _ = mt.precision_recall_curve(
            binarized_y_true.ravel(), y_probas.ravel()
        )
        average_precision = mt.average_precision_score(
            binarized_y_true, y_probas, average="micro"
        )
        ax.plot(
            recall,
            precision,
            color="deeppink",
            linestyle=":",
            linewidth=4,
            label=(f"micro-average Precision-recall curve "
                   f"(area = {average_precision:.2f})"),
        )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='best', fontsize=text_fontsize)
    return ax 


def plot_cumulative_gain():
    raise NotImplementedError


def plot_lift_curve():
    raise NotImplementedError


def plot_ks_statistic():
    raise NotImplementedError

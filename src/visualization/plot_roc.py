from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics
import seaborn as sns

sns.set_theme(style="ticks")


def compute_and_plot_roc_curves(path, id_sigma, ood_sigma, pre_fix=""):

    id_sigma = np.reshape(id_sigma, (id_sigma.shape[0], -1))
    ood_sigma = np.reshape(ood_sigma, (ood_sigma.shape[0], -1))

    id_sigma, ood_sigma = id_sigma.sum(axis=1), ood_sigma.sum(axis=1)

    pred = np.concatenate([id_sigma, ood_sigma])
    target = np.concatenate([[0] * len(id_sigma), [1] * len(ood_sigma)])

    # plot roc curve
    roc = torchmetrics.ROC(num_classes=1)
    fpr, tpr, thresholds = roc(torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1))

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(fpr, tpr)
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    fig.tight_layout()
    fig.savefig(f"figures/{path}/{pre_fix}ood_roc_curve.png")

    # save data
    # data = pd.DataFrame(
    #     np.concatenate([pred[:, None], target[:, None]], axis=1),
    #     columns=["sigma", "labels"],
    # )
    # data.to_csv(f"figures/{path}/{pre_fix}ood_roc_curve_data.csv")

    # plot precision recall curve
    pr_curve = torchmetrics.PrecisionRecallCurve(pos_label=1)
    precision, recall, thresholds = pr_curve(torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1))

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(recall, precision)
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
    )
    fig.tight_layout()
    fig.savefig(f"figures/{path}/{pre_fix}ood_precision_recall_curve.png")

    metrics = {}

    # compute auprc (area under precission recall curve)
    auc = torchmetrics.AUC(reorder=True)
    auprc_score = auc(recall, precision)
    metrics["auprc"] = float(auprc_score.numpy())

    # compute false positive rate at 80
    num_id = len(id_sigma)

    for p in range(0, 100, 10):
        # if there is no difference in variance
        try:
            metrics[f"fpr{p}"] = float(fpr[int(p / 100.0 * num_id)].numpy())
        except:
            metrics[f"fpr{p}"] = "none"
        else:
            continue

    # compute auroc
    auroc = torchmetrics.AUROC(num_classes=1)
    auroc_score = auroc(torch.tensor(pred).unsqueeze(1), torch.tensor(target).unsqueeze(1))
    metrics["auroc"] = float(auroc_score.numpy())

    return metrics

    # # save metrics
    # with open(f"../figures/{path}/{pre_fix}ood_metrics.json", "w") as outfile:
    #     json.dump(metrics, outfile)


if __name__ == "__main__":
    id_title = "CIFAR-10"
    id_label = id_title.lower()

    method = "post_hoc"

    ood_title = "SVHN"
    # ood_title = "CIFAR-100"
    ood_label = ood_title.lower()

    mu_id = np.load(f"results/{method}/{id_label}/id_laplace_mu.npy")
    var_id = np.load(f"results/{method}/{id_label}/id_laplace_sigma_sq.npy")
    mu_ood = np.load(f"results/{method}/{id_label}/{ood_label}/ood_laplace_mu.npy")
    var_ood = np.load(f"results/{method}/{id_label}/{ood_label}/ood_laplace_sigma_sq.npy")

    metrics = compute_and_plot_roc_curves("", var_id, var_ood, pre_fix=f"{id_label}_{ood_label}_")
    print(metrics)
    # fig, ax = plot_ood(mu_id, var_id, mu_ood, var_ood)
    # fig.suptitle(f"Trained on {id_title}, OOD {ood_title}")
    # fig.tight_layout()
    # fig.savefig(f"figures/ood_plot_{id_label}_{ood_label}.png")

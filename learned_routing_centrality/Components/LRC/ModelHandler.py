import datetime
from tqdm import tqdm
import torch
import scipy.spatial
import scipy.stats
import numpy as np
from Components.Utils.ParamsManager import ParamsManager
from Components.Utils.CommonStr import HyperParams

max_pearson = 0
pearson_threshold = 0.9
expected_rbcs_df, actual_rbcs_df = [], []


def train_model(nn_model, optimizer, p_man, train_ldr, val_ldr, test_ldr):
    start_train_time = datetime.datetime.now()
    train_result = train_nn(nn_model, p_man, optimizer, train_ldr, val_ldr)
    best_model, train_error, validation_error = train_result
    total_train_time = datetime.datetime.now() - start_train_time
    print(f'train time: {total_train_time}')

    return best_model, train_error, validation_error, total_train_time


def train_nn(nn_model, p_man: ParamsManager, optimizer, train_ldr, val_ldr):
    print(f'starting training')
    start_time = datetime.datetime.now()
    epochs = p_man.hyper_params[HyperParams.epochs]
    n_batches, n_batches_val = len(train_ldr), len(val_ldr)

    pearson_avg_train, pearson_avg_test = compute_pearson_corr(nn_model, train_ldr, val_ldr)
    print(f'\n[Before Training] pearson avg corr score - training set: {pearson_avg_train} , test set: {pearson_avg_test}')

    for epoch in range(0, epochs):
        train_running_loss = 0.0
        for i, samples in enumerate(tqdm(train_ldr)):
            X, expected_rbcs = samples
            train_batch_loss = compute_batch_loss(nn_model, X, expected_rbcs)
            train_running_loss += train_batch_loss.item()

            optimizer.zero_grad()
            train_batch_loss.backward()
            optimizer.step()

        train_err = train_running_loss / n_batches

        if epoch % 1 == 0:
            val_running_loss = 0.0
            nn_model.eval()
            with torch.no_grad():
                for i, samples in enumerate(val_ldr):
                    Xs, ys = samples[0], samples[1]
                    val_batch_loss = compute_batch_loss(nn_model, Xs, ys)
                    val_running_loss += val_batch_loss.item()

                val_err = val_running_loss / n_batches_val
            pearson_avg_train, pearson_avg_test = compute_pearson_corr(nn_model, train_ldr, val_ldr)
            print(f'\n[{epoch}] pearson avg corr score - training set: {pearson_avg_train} , test set: {pearson_avg_test}')
            # print(f'\n[{epoch}] train loss: {train_err} , validation loss: {val_err}')
            nn_model.train()

    return nn_model, train_err, val_err


def compute_batch_loss(nn_model, X, expected_rbcs):
    actual_rbcs, pearson_scores = nn_model(X, expected_rbcs)
    pearson_loss = torch.sum((1 - torch.stack(pearson_scores, dim=0))) / len(expected_rbcs)

    return pearson_loss


def compute_pearson_corr(nn_model, train_ldr, val_ldr):
    nn_model.eval()
    with torch.no_grad():
        test_corr = compute_pearson_avg_corr(nn_model, val_ldr)
        train_corr = compute_pearson_avg_corr(nn_model, train_ldr)
    nn_model.train()

    return train_corr, test_corr


def compute_pearson_avg_corr(model, ldr):
    n_instances = ldr.dataset.__len__()
    dist_avg, kendall_avg, pearson_avg, spearman_avg = 0, 0, 0, 0
    for i, samples in enumerate(ldr):
        Xs, ys = samples[0], samples[1]
        actual_rbcs, expected_rbcs = model(Xs), ys
        e_a = list(zip(expected_rbcs, actual_rbcs))
        pearsonr_arr = np.array([scipy.stats.pearsonr(expected, actual.cpu())[0] for expected, actual in e_a])
        pearson_avg += (pearsonr_arr / n_instances).sum()

    return pearson_avg

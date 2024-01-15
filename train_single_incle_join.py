import torch
import itertools
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from layers import ZINBLoss, MeanAct, DispAct, GaussianNoise
import numpy as np
from sklearn.cluster import KMeans
import math, os
from sklearn import metrics
from preprocessing import *
import argparse
import random
from itertools import cycle
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pandas as pd
from augmentation import *


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target):
    num_correct = np.sum(output == target)
    res = num_correct / len(target)
    return res


def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def auxilarly_dis(pred):
    weight = (pred ** 2) / torch.sum(pred, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def entropy(x):
    """
    Helper function to compute the entropy over the batch
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8
    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)

    if len(b.size()) == 2: # Sample-wise entropy
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


def buildNetwork(layers, activation="relu", noise=False, batchnorm=False):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if noise:
            net.append(GaussianNoise())
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if batchnorm:
            net.append(nn.BatchNorm1d(layers[i]))
    return nn.Sequential(*net)


class Prototype(nn.Module):
    def __init__(self, num_classes, input_size, tau=0.05):
        super(Prototype, self).__init__()
        self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.tau = tau
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc(x) / self.tau
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, z_dim, encodeLayer=[], decodeLayer=[], activation="relu"):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.activation = activation
        self.encoder = buildNetwork([self.input_dim] + encodeLayer, activation=activation, noise=True, batchnorm=False)
        self.decoder = buildNetwork([self.z_dim] + decodeLayer, activation=activation, batchnorm=False)
        self._enc_mu = nn.Linear(encodeLayer[-1], self.z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], self.input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], self.input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], self.input_dim), nn.Sigmoid())

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)
        h = self.decoder(z)
        mean = self._dec_mean(h)
        disp = self._dec_disp(h)
        pi = self._dec_pi(h)
        return z, mean, disp, pi


def extractor(model, test_loader, device):
    model.eval()
    test_embedding = []
    test_label = []
    test_index = []
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            x_t, label_t, index_t = data[0].to(device), data[3].to(device), data[4].to(device)
            z_t, _, _, _ = model(x_t)
            test_embedding.append(z_t.detach())
            test_label.append(label_t)
            test_index.append(index_t)
    test_embedding = torch.cat(test_embedding, dim=0)
    test_label = torch.cat(test_label)
    test_index = torch.cat(test_index)
    _, test_indexes = torch.sort(test_index, descending=False)
    test_embedding = test_embedding[test_indexes]
    test_label = test_label[test_indexes]
    test_embedding = test_embedding.cpu().numpy()
    test_label = test_label.cpu().numpy()
    return test_embedding, test_label


def test(model, labeled_num, device, test_loader, cluster_mapping, epoch):
    model.eval()
    preds = np.array([])
    preds_open = np.array([])
    targets = np.array([])
    confs = np.array([])
    confs_open = np.array([])
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            x_t, label_t, index_t, batch_t = data[0].to(device), data[3].to(device), data[4].to(device), data[5].to(device)
            z, _, _, _, _, output_s, output_t = model(x_t, batch_t)
            conf, pred = output_t.max(1)
            targets = np.append(targets, label_t.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            preds_open = np.append(preds_open, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
            confs_open = np.append(confs_open, conf.cpu().numpy())
    for i in range(len(cluster_mapping)):
        preds[preds == cluster_mapping[i]] = i
    k = 0
    for j in np.unique(preds_open):
        if j not in cluster_mapping:
            preds[preds == j] = len(cluster_mapping) + k
            k += 1
    targets = targets.astype(int)
    preds = preds.astype(int)
    preds_open = preds_open.astype(int)
    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    overall_acc2 = cluster_acc(preds_open, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    seen_acc2 = cluster_acc(preds_open[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_acc2 = cluster_acc(preds_open[unseen_mask], targets[unseen_mask])
    print('In the old {}-th epoch, Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(epoch, overall_acc,
                                                                                                   seen_acc,
                                                                                                   unseen_acc))
    print('In the old {}-th epoch, Test overall acc2 {:.4f}, seen acc2 {:.4f}, unseen acc2 {:.4f}'.format(epoch, overall_acc2,
                                                                                                    seen_acc2,
                                                                                                    unseen_acc2))
    return overall_acc, seen_acc, unseen_acc, overall_acc2, seen_acc2, unseen_acc2


def dataset_spliting(X, count_X, cellname, size_factor, class_set, stage_number, labeled_ratio=0.5, random_seed=8888):
    train_X_set = []
    train_count_X_set = []
    train_cellname_set = []
    train_size_factor_set = []
    train_Y_set = []
    test_X_set = []
    test_count_X_set = []
    test_cellname_set = []
    test_size_factor_set = []
    test_Y_set = []

    for i in range(stage_number):
        if i != stage_number - 1:
            current_class_set = class_set[((len(class_set) // stage_number) * i):((len(class_set) // stage_number) * (i + 1))]
            print("For the {}-th stage, the class set is {} and the class number is {}".format(i, current_class_set, len(current_class_set)))
        else:
            current_class_set = class_set[((len(class_set) // stage_number) * i):]
            print("For the {}-th stage, the class set is {} and the class number is {}".format(i, current_class_set, len(current_class_set)))

        train_index = []
        test_index = []
        np.random.seed(random_seed)

        for j in range(X.shape[0]):
            if cellname[j] in current_class_set:
                if np.random.rand() < labeled_ratio:
                    train_index.append(j)
                else:
                    test_index.append(j)

        train_X = X[train_index]
        train_count_X = count_X[train_index]
        train_cellname = cellname[train_index]
        train_size_factor = size_factor[train_index]
        test_X = X[test_index]
        test_count_X = count_X[test_index]
        test_cellname = cellname[test_index]
        test_size_factor = size_factor[test_index]
        train_Y = np.array([0] * len(train_cellname))
        test_Y = np.array([0] * len(test_cellname))

        for k in range(len(class_set)):
            if class_set[k] in current_class_set:
                train_Y[train_cellname == class_set[k]] = k
                test_Y[test_cellname == class_set[k]] = k
        print("For the {}-th stage, the train cell class number is {} and the test class number "
              "is {}".format(i, len(np.unique(train_Y)), len(np.unique(test_Y))))
        train_X_set.append(train_X)
        train_count_X_set.append(train_count_X)
        train_cellname_set.append(train_cellname)
        train_size_factor_set.append(train_size_factor)
        train_Y_set.append(train_Y)
        test_X_set.append(test_X)
        test_count_X_set.append(test_count_X)
        test_cellname_set.append(test_cellname)
        test_size_factor_set.append(test_size_factor)
        test_Y_set.append(test_Y)

    return train_X_set, train_count_X_set, train_cellname_set, train_size_factor_set, train_Y_set, \
           test_X_set, test_count_X_set, test_cellname_set, test_size_factor_set, test_Y_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scUDA')
    parser.add_argument('--random-seed', type=int, default=8888, metavar='S')
    parser.add_argument('--gpu-id', default='0', type=int)
    parser.add_argument('--num', default=0, type=int)
    parser.add_argument('--ra', type=float, default=0.5)
    parser.add_argument('--age', default=2, type=int)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--highly-genes', type=int, default=2000)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--pretrain', type=int, default=200)
    parser.add_argument('--finetune', type=int, default=200)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--structure', type=int, default=1)

    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu", args.gpu_id)

    filename_set = ["Cao", "Quake_10x", "Quake_Smart-seq2", "Zeisel_2018"]

    result_list = []

    for i in range(args.num, args.num + 1):
        filename = filename_set[i]
        dataname = filename
        X, cell_name, gene_name = read_real_with_genes(filename, batch=False)
        class_set = class_splitting_single(filename)
        total_classes = len(class_set)

        index = []
        for i in range(len(cell_name)):
            if cell_name[i] in class_set:
                index.append(i)
        X = X[index]
        cell_name = cell_name[index]

        count_X = X.astype(np.int)
        adata = sc.AnnData(X)
        adata.var["gene_id"] = gene_name
        adata.obs["cellname"] = cell_name
        adata = normalize(adata, highly_genes=args.highly_genes, size_factors=True, normalize_input=True,
                          logtrans_input=True)
        X = adata.X.astype(np.float32)
        cell_name = np.array(adata.obs["cellname"])
        gene_name = np.array(adata.var["gene_id"])
        print("after preprocessing, the cell number is {} and the gene dimension is {}".format(len(cell_name), len(gene_name)))

        if args.highly_genes != None:
            high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
            count_X = count_X[:, high_variable]
        else:
            select_genes = np.array(adata.var.index, dtype=np.int)
            select_cells = np.array(adata.obs.index, dtype=np.int)
            count_X = count_X[:, select_genes]
            count_X = count_X[select_cells]
        assert X.shape == count_X.shape
        size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)

        labeled_ratio = args.ra  # 0.5
        stage_number = args.age
        source_X_set, source_count_X_set, source_cellname_set, source_size_factor_set, source_Y_set, \
        target_X_set, target_count_X_set, target_cellname_set, target_size_factor_set, target_Y_set \
            = dataset_spliting(X, count_X, cell_name, size_factor, class_set, stage_number, labeled_ratio=labeled_ratio, random_seed=args.random_seed)
        print("we have finished the dataset splitting process!!!")

        if args.structure == 0:
            model = AutoEncoder(X.shape[1], 32, encodeLayer=[256, 64], decodeLayer=[64, 256], activation="relu")
        else:
            model = AutoEncoder(X.shape[1], 128, encodeLayer=[512, 256], decodeLayer=[256, 512], activation="relu")
        model = model.to(device)

        class_number_set = [0]

        for current_stage in range(stage_number):
            current_result = [filename]
            current_result.append(current_stage + 1)
            source_x = source_X_set[current_stage]
            source_raw_x = source_count_X_set[current_stage]
            source_cellname = source_cellname_set[current_stage]
            source_sf = source_size_factor_set[current_stage]
            source_y = source_Y_set[current_stage]

            target_x = target_X_set[current_stage]
            target_raw_x = target_count_X_set[current_stage]
            target_cellname = target_cellname_set[current_stage]
            target_sf = target_size_factor_set[current_stage]
            target_y = target_Y_set[current_stage]

            if current_stage == 0:
                current_classes = len(np.unique(target_y))
                class_number_set.append(current_classes)
                print("the class set is {}".format(class_number_set))
            else:
                current_classes = len(np.unique(target_y)) + class_number_set[-1]
                class_number_set.append(current_classes)
                print("the class set is {}".format(class_number_set))

            if current_stage > 0:
                last_source_x = np.concatenate(source_X_set[:current_stage])
                last_source_raw_x = np.concatenate(source_count_X_set[:current_stage])
                last_source_cellname = np.concatenate(source_cellname_set[:current_stage])
                last_source_sf = np.concatenate(source_size_factor_set[:current_stage])
                last_source_y = np.concatenate(source_Y_set[:current_stage])

                last_target_x = np.concatenate(target_X_set[:current_stage])
                last_target_raw_x = np.concatenate(target_count_X_set[:current_stage])
                last_target_cellname = np.concatenate(target_cellname_set[:current_stage])
                last_target_sf = np.concatenate(target_size_factor_set[:current_stage])
                last_target_y = np.concatenate(target_Y_set[:current_stage])

            total_source_x = np.concatenate(source_X_set[:(current_stage + 1)])
            total_source_raw_x = np.concatenate(source_count_X_set[:(current_stage + 1)])
            total_source_cellname = np.concatenate(source_cellname_set[:(current_stage + 1)])
            total_source_sf = np.concatenate(source_size_factor_set[:(current_stage + 1)])
            total_source_y = np.concatenate(source_Y_set[:(current_stage + 1)])

            total_target_x = np.concatenate(target_X_set[:(current_stage + 1)])
            total_target_raw_x = np.concatenate(target_count_X_set[:(current_stage + 1)])
            total_target_cellname = np.concatenate(target_cellname_set[:(current_stage + 1)])
            total_target_sf = np.concatenate(target_size_factor_set[:(current_stage + 1)])
            total_target_y = np.concatenate(target_Y_set[:(current_stage + 1)])

            if source_x.shape[0] < args.batch_size:
                args.batch_size = source_x.shape[0]

            if args.structure == 0:
                proto_net = Prototype(current_classes, 32, tau=args.tau)
            else:
                proto_net = Prototype(current_classes, 128, tau=args.tau)
            proto_net = proto_net.to(device)
            if current_stage > 0:
                state_dict = proto_net.state_dict()
                state_dict['fc.weight'][:class_number_set[-2]] = F.normalize(prototype_weight_store).to(device)
                proto_net.load_state_dict(state_dict)
                print("In the {}-th state, we have loaded the prototype weight in the last stage successfully".format(current_stage))

            source_dataset = TensorDataset(torch.tensor(total_source_x), torch.tensor(total_source_raw_x), torch.tensor(total_source_sf),
                                           torch.tensor(total_source_y), torch.arange(total_source_x.shape[0]))
            source_dataloader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            target_dataset = TensorDataset(torch.tensor(total_target_x), torch.tensor(total_target_raw_x), torch.tensor(total_target_sf),
                                           torch.tensor(total_target_y), torch.arange(total_target_x.shape[0]))
            target_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            train_dataset = TensorDataset(torch.tensor(source_x), torch.tensor(source_raw_x), torch.tensor(source_sf),
                                           torch.tensor(source_y), torch.arange(source_x.shape[0]))
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
            test_dataset = TensorDataset(torch.tensor(target_x), torch.tensor(target_raw_x), torch.tensor(target_sf),
                                           torch.tensor(target_y), torch.arange(target_x.shape[0]))
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            if current_stage > 0:
                last_train_dataset = TensorDataset(torch.tensor(last_source_x), torch.tensor(last_source_raw_x), torch.tensor(last_source_sf),
                                              torch.tensor(last_source_y), torch.arange(last_source_x.shape[0]))
                last_train_dataloader = DataLoader(last_train_dataset, batch_size=args.batch_size, shuffle=False)
                last_test_dataset = TensorDataset(torch.tensor(last_target_x), torch.tensor(last_target_raw_x), torch.tensor(last_target_sf),
                                                   torch.tensor(last_target_y), torch.arange(last_target_x.shape[0]))
                last_test_dataloader = DataLoader(last_test_dataset, batch_size=args.batch_size, shuffle=False)

            optimizer = optim.Adam(itertools.chain(model.parameters(), proto_net.parameters()), lr=args.lr, amsgrad=True)

            bce = nn.BCELoss().to(device)
            ce = nn.CrossEntropyLoss().to(device)

            if current_stage == 0:
                for epoch in range(args.pretrain + args.finetune + 1):
                    if epoch % args.interval == 0:
                        model.eval()
                        proto_net.eval()
                        preds = np.array([])
                        targets = np.array([])
                        confs = np.array([])

                        test_embeddings = []
                        test_indexes = []

                        with torch.no_grad():
                            for _, data in enumerate(test_dataloader):
                                x_t, label_t, index_t = data[0].to(device), data[3].to(device), data[4].to(device)
                                z_t, _, _, _ = model(x_t)
                                output_t = proto_net(z_t)
                                conf, pred = output_t.max(1)
                                targets = np.append(targets, label_t.cpu().numpy())
                                preds = np.append(preds, pred.cpu().numpy())
                                confs = np.append(confs, conf.cpu().numpy())

                                test_embeddings.append(z_t.detach())
                                test_indexes.append(index_t)

                        targets = targets.astype(int)
                        preds = preds.astype(int)
                        overall_acc = accuracy(preds, targets)
                        print("In the {}-th stage and {}-th epoch, Test overall acc for this stage {:.4f}".format(
                            current_stage, epoch, overall_acc))
                        model.train()
                        proto_net.train()
                        if epoch == args.pretrain + args.finetune:
                            current_result.extend([0., round(overall_acc, 4), round(overall_acc, 4)])

                            test_embeddings = torch.cat(test_embeddings, dim=0)
                            test_indexes = torch.cat(test_indexes)
                            _, test_indexes = torch.sort(test_indexes, descending=False)
                            test_embeddings = test_embeddings[test_indexes].cpu().numpy()
                            test_indexes = test_indexes.cpu().numpy()
                            test_true_labels = targets[test_indexes]
                            test_pred_labels = preds[test_indexes]
                            test_true_celltypes = target_cellname
                            test_pred_celltypes = target_cellname.copy()
                            unique_test_preds = np.unique(test_pred_labels)
                            for j in range(len(unique_test_preds)):
                                test_pred_celltypes[test_pred_labels == unique_test_preds[j]] = class_set[
                                    unique_test_preds[j]]
                            test_data_infor = pd.DataFrame(
                                {"true label": test_true_labels, "true cell type": test_true_celltypes,
                                 "pred label": test_pred_labels, "pred cell type": test_pred_celltypes})
                            test_data_infor.to_csv(
                                "case/{}_stage_{}_test_data_joint_sankey_information.csv".format(
                                    dataname, current_stage))
                            pd.DataFrame(test_embeddings).to_csv(
                                "case/{}_stage_{}_test_data_joint_visualization_feature.csv".format(
                                    dataname, current_stage))

                    recon_losses = AverageMeter('recon_loss', ':.4e')
                    ce_losses = AverageMeter('ce_loss', ':.4e')
                    model.train()
                    proto_net.train()

                    for batch_idx, (x_s, raw_x_s, sf_s, y_s, index_s) in enumerate(source_dataloader):
                        x_s, raw_x_s, sf_s, y_s, index_s = x_s.to(device), raw_x_s.to(device), \
                                                           sf_s.to(device), y_s.to(device), \
                                                           index_s.to(device)
                        z_s, mean_s, disp_s, pi_s = model(x_s)
                        recon_loss = ZINBLoss().to(device)(x=raw_x_s, mean=mean_s, disp=disp_s, pi=pi_s,
                                                           scale_factor=sf_s)
                        output_s = proto_net(z_s)
                        ce_loss = ce(output_s, y_s)
                        if epoch < args.pretrain:
                            loss = recon_loss
                        else:
                            loss = recon_loss + ce_loss
                        recon_losses.update(recon_loss.item(), args.batch_size)
                        ce_losses.update(ce_loss.item(), args.batch_size)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    print("In {}-th stage, Training {}/{}, zinb loss: {:.4f}, ce loss: {:.4f}".format(current_stage, epoch, args.pretrain + args.finetune + 1,
                                                                                      recon_losses.avg, ce_losses.avg))

            else:
                for epoch in range(args.finetune + 1):
                    if epoch % args.interval == 0:
                        model.eval()
                        proto_net.eval()
                        preds = np.array([])
                        targets = np.array([])
                        confs = np.array([])

                        test_embeddings = []
                        test_indexes = []

                        with torch.no_grad():
                            for _, data in enumerate(test_dataloader):
                                x_t, label_t, index_t = data[0].to(device), data[3].to(device), data[4].to(device)
                                z_t, _, _, _ = model(x_t)
                                output_t = proto_net(z_t)
                                conf, pred = output_t.max(1)
                                targets = np.append(targets, label_t.cpu().numpy())
                                preds = np.append(preds, pred.cpu().numpy())
                                confs = np.append(confs, conf.cpu().numpy())

                                test_embeddings.append(z_t.detach())
                                test_indexes.append(index_t)

                        targets = targets.astype(int)
                        preds = preds.astype(int)
                        current_acc = accuracy(preds, targets)
                        print("In the {}-th stage and {}-th epoch, Test acc for this stage {:.4f}".format(current_stage,
                                                                                                          epoch,
                                                                                                          current_acc))

                        last_preds = np.array([])
                        last_targets = np.array([])

                        last_test_embeddings = []
                        last_test_indexes = []

                        with torch.no_grad():
                            for _, data in enumerate(last_test_dataloader):
                                x_t, label_t, index_t = data[0].to(device), data[3].to(device), data[4].to(device)
                                z_t, _, _, _ = model(x_t)
                                output_t = proto_net(z_t)
                                conf, pred = output_t.max(1)
                                last_targets = np.append(last_targets, label_t.cpu().numpy())
                                last_preds = np.append(last_preds, pred.cpu().numpy())

                                last_test_embeddings.append(z_t.detach())
                                last_test_indexes.append(index_t)

                        last_targets = last_targets.astype(int)
                        last_preds = last_preds.astype(int)
                        last_overall_acc = accuracy(last_preds, last_targets)
                        print("In the {}-th stage and {}-th epoch, Test acc for previous stage {:.4f}".format(
                            current_stage, epoch,
                            last_overall_acc))

                        overall_targets = np.concatenate((targets, last_targets))
                        overall_preds = np.concatenate((preds, last_preds))
                        overall_acc = accuracy(overall_preds, overall_targets)
                        print("In the {}-th stage and {}-th epoch, Test acc for overall stage {:.4f}".format(
                            current_stage, epoch,
                            overall_acc))
                        model.train()
                        proto_net.train()
                        if epoch == args.finetune:
                            current_result.extend(
                                [round(last_overall_acc, 4), round(current_acc, 4), round(overall_acc, 4)])

                            test_embeddings = torch.cat(test_embeddings, dim=0)
                            test_indexes = torch.cat(test_indexes)
                            _, test_indexes = torch.sort(test_indexes, descending=False)
                            test_embeddings = test_embeddings[test_indexes].cpu().numpy()
                            test_indexes = test_indexes.cpu().numpy()
                            test_true_labels = targets[test_indexes]
                            test_pred_labels = preds[test_indexes]
                            test_true_celltypes = target_cellname
                            test_pred_celltypes = target_cellname.copy()
                            unique_test_preds = np.unique(test_pred_labels)
                            for j in range(len(unique_test_preds)):
                                test_pred_celltypes[test_pred_labels == unique_test_preds[j]] = class_set[
                                    unique_test_preds[j]]

                            last_test_embeddings = torch.cat(last_test_embeddings, dim=0)
                            last_test_indexes = torch.cat(last_test_indexes)
                            _, last_test_indexes = torch.sort(last_test_indexes, descending=False)
                            last_test_embeddings = last_test_embeddings[last_test_indexes].cpu().numpy()
                            last_test_indexes = last_test_indexes.cpu().numpy()
                            last_test_true_labels = last_targets[last_test_indexes]
                            last_test_pred_labels = last_preds[last_test_indexes]
                            last_test_true_celltypes = last_target_cellname
                            last_test_pred_celltypes = last_target_cellname.copy()
                            unique_last_test_preds = np.unique(last_test_pred_labels)
                            for j in range(len(unique_last_test_preds)):
                                last_test_pred_celltypes[last_test_pred_labels == unique_last_test_preds[j]] = \
                                class_set[unique_last_test_preds[j]]

                            overall_test_embeddings = np.concatenate((last_test_embeddings, test_embeddings), axis=0)
                            overall_test_true_labels = np.concatenate((last_test_true_labels, test_true_labels))
                            overall_test_pred_labels = np.concatenate((last_test_pred_labels, test_pred_labels))
                            overall_test_true_celltypes = np.concatenate(
                                (last_test_true_celltypes, test_true_celltypes))
                            overall_test_pred_celltypes = np.concatenate(
                                (last_test_pred_celltypes, test_pred_celltypes))
                            overall_test_data_infor = pd.DataFrame(
                                {"true label": overall_test_true_labels, "true cell type": overall_test_true_celltypes,
                                 "pred label": overall_test_pred_labels, "pred cell type": overall_test_pred_celltypes})
                            overall_test_data_infor.to_csv(
                                "case/{}_stage_{}_test_data_joint_sankey_information.csv".format(
                                    dataname, current_stage))
                            pd.DataFrame(overall_test_embeddings).to_csv(
                                "case/{}_stage_{}_test_data_joint_visualization_feature.csv".format(
                                    dataname, current_stage))

                        model.train()
                        proto_net.train()

                    recon_losses = AverageMeter('recon_loss', ':.4e')
                    ce_losses = AverageMeter('ce_loss', ':.4e')
                    model.train()
                    proto_net.train()

                    for batch_idx, (x_s, raw_x_s, sf_s, y_s, index_s) in enumerate(source_dataloader):
                        x_s, raw_x_s, sf_s, y_s, index_s = x_s.to(device), raw_x_s.to(device), \
                                                                    sf_s.to(device), y_s.to(device), \
                                                                    index_s.to(device)
                        z_s, mean_s, disp_s, pi_s = model(x_s)
                        recon_loss = ZINBLoss().to(device)(x=raw_x_s, mean=mean_s, disp=disp_s, pi=pi_s, scale_factor=sf_s)
                        output_s = proto_net(z_s)
                        ce_loss = ce(output_s, y_s)
                        loss = recon_loss + ce_loss
                        recon_losses.update(recon_loss.item(), args.batch_size)
                        ce_losses.update(ce_loss.item(), args.batch_size)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    print("In {}-th stage, Training {}/{}, zinb loss: {:.4f}, ce loss: {:.4f}".format(current_stage, epoch, args.finetune + 1,
                                                                                                      recon_losses.avg, ce_losses.avg))

            prototype_weight_store = proto_net.fc.weight.data
            result_list.append(current_result)
            print("The result list is {}".format(result_list))

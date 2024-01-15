from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sc_utils as utils
import numpy as np
import h5py
import scipy as sp
import pandas as pd
import scanpy.api as sc
from sklearn.metrics.cluster import contingency_matrix
import anndata


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = utils.decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = utils.dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def read_data(filename, sparsify=False, skip_exprs=False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index=utils.decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index=utils.decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns


def read_real_with_genes(filename, batch=True):
    # data_path = "/data/public/scrna/data/" + filename + "/data.h5"
    data_path = "../scrna/data/" + filename + "/data.h5"
    # data_path = "../scrna/data/" + filename + "/data.h5"
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_ontology_class"])
    gene_name = np.array(list(var.index))
    if (cell_name == "").sum() > 0:
        cell_name[cell_name == ""] = "unknown_class"
    # cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    if batch == True:
        if "dataset_name" in obs.keys():
            batch_name = np.array(obs["dataset_name"])
        else:
            batch_name = np.array(obs["study"])
        # _, batch_label = np.unique(batch_name, return_inverse=True)
        return X, cell_name, batch_name, gene_name
    else:
        return X, cell_name, gene_name


def read_real_with_genes_new(filename, batch=False):
    data_path = "../scrna/data/" + filename + "/data.h5ad"
    adata = anndata.read_h5ad(data_path)
    mat = adata.X
    obs = adata.obs
    var = adata.var
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_ontology_class"])
    gene_name = np.array(list(var.index))
    if (cell_name == "").sum() > 0:
        cell_name[cell_name == ""] = "unknown_class"
    if batch == True:
        if "dataset_name" in obs.keys():
            batch_name = np.array(obs["dataset_name"])
        else:
            batch_name = np.array(obs["study"])
        return X, cell_name, batch_name, gene_name
    else:
        return X, cell_name, gene_name


def class_splitting_single(dataname):
    class_set = []
    if dataname == "Quake_10x": # 36
        class_set = ['B cell', 'T cell', 'alveolar macrophage', 'basal cell', 'basal cell of epidermis', 'bladder cell',
                     'bladder urothelial cell', 'blood cell', 'endothelial cell', 'epithelial cell', 'fibroblast',
                     'granulocyte', 'granulocytopoietic cell', 'hematopoietic precursor cell', 'hepatocyte',
                     'immature T cell', 'keratinocyte', 'kidney capillary endothelial cell', 'kidney collecting duct epithelial cell',
                     'kidney loop of Henle ascending limb epithelial cell', 'kidney proximal straight tubule epithelial cell',
                     'late pro-B cell', 'leukocyte', 'luminal epithelial cell of mammary gland', 'lung endothelial cell',
                     'macrophage', 'mesenchymal cell', 'mesenchymal stem cell', 'monocyte', 'natural killer cell',
                     'neuroendocrine cell', 'non-classical monocyte', 'proerythroblast', 'promonocyte', 'skeletal muscle satellite cell',
                     'stromal cell']
    if dataname == "Quake_Smart-seq2": # 45
        class_set = ['B cell', 'Slamf1-negative multipotent progenitor cell', 'T cell', 'astrocyte of the cerebral cortex',
                     'basal cell', 'basal cell of epidermis', 'bladder cell', 'bladder urothelial cell', 'blood cell',
                     'endothelial cell', 'enterocyte of epithelium of large intestine', 'epidermal cell', 'epithelial cell',
                     'epithelial cell of large intestine', 'epithelial cell of proximal tubule', 'fibroblast', 'granulocyte',
                     'hematopoietic precursor cell', 'hepatocyte', 'immature B cell', 'immature T cell', 'keratinocyte',
                     'keratinocyte stem cell', 'large intestine goblet cell', 'late pro-B cell', 'leukocyte',
                     'luminal epithelial cell of mammary gland', 'lung endothelial cell', 'macrophage', 'mesenchymal cell',
                     'mesenchymal stem cell', 'mesenchymal stem cell of adipose', 'microglial cell', 'monocyte', 'myeloid cell',
                     'naive B cell', 'neuron', 'oligodendrocyte', 'oligodendrocyte precursor cell', 'pancreatic A cell',
                     'pro-B cell', 'skeletal muscle satellite cell', 'skeletal muscle satellite stem cell', 'stromal cell',
                     'type B pancreatic cell']
    if dataname == "Cao": # 16
        class_set = ['GABAergic neuron', 'cholinergic neuron', 'ciliated olfactory receptor neuron', 'coelomocyte', 'epidermal cell',
                     'germ line cell', 'glial cell', 'interneuron', 'muscle cell', 'nasopharyngeal epithelial cell', 'neuron',
                     'seam cell', 'sensory neuron', 'sheath cell', 'socket cell (sensu Nematoda)', 'visceral muscle cell']
    if dataname == "Zeisel_2018": # 21
        class_set = ['CNS neuron (sensu Vertebrata)', 'astrocyte', 'cerebellum neuron', 'choroid plexus epithelial cell',
                     'dentate gyrus of hippocampal formation granule cell',
                     'endothelial cell of vascular tree', 'enteric neuron', 'ependymal cell', 'glial cell',
                     'inhibitory interneuron',
                     'microglial cell', 'neuroblast', 'oligodendrocyte', 'oligodendrocyte precursor cell', 'peptidergic neuron', 'pericyte cell',
                     'peripheral sensory neuron',
                     'perivascular macrophage', 'radial glial cell', 'sympathetic noradrenergic neuron',
                     'vascular associated smooth muscle cell']
    if dataname == "Cao_2020_Eye": # 51836, 11
        class_set = ['photoreceptor cell', 'retinal ganglion cell', 'amacrine cell', 'retina horizontal cell',
                      'retinal bipolar neuron', 'stromal cell', 'visual pigment cell', 'lens fiber cell',
                      'blood vessel endothelial cell', 'astrocyte', 'cell of skeletal muscle']
    if dataname == "Cao_2020_Intestine": # 51650, 12
        class_set = ['intestinal epithelial cell', 'stromal cell', 'myeloid cell', 'enteric neuron', 'leukocyte', 'glial cell',
                     'blood vessel endothelial cell', 'enteric smooth muscle cell', 'chromaffin cell', 'endothelial cell of lymphatic vessel',
                     'mesothelial cell', 'erythroblast']
    if dataname == "Cao_2020_Pancreas": # 45653, 13
        class_set = ['pancreatic acinar cell', 'leukocyte', 'stromal cell of pancreas', 'pancreatic ductal cell',
                     'blood vessel endothelial cell', 'pancreatic endocrine cell', 'smooth muscle cell', 'myeloid cell',
                     'erythroblast', 'glial cell', 'enteric neuron', 'endothelial cell of lymphatic vessel', 'mesothelial cell']
    if dataname == "Cao_2020_Stomach": # 12106, 10
        class_set = ['goblet cell', 'squamous epithelial cell', 'stromal cell', 'leukocyte', 'blood vessel endothelial cell',
                     'stomach neuroendocrine cell', 'ciliated epithelial cell', 'mesothelial cell', 'myeloid cell', 'erythroblast']
    if dataname == "Madissoon_Lung": # 17
        class_set = ['natural killer cell', 'CD4-positive helper T cell', 'monocyte', 'cytotoxic T cell', 'lung macrophage',
                     'type II pneumocyte', 'fibroblast', 'mast cell', 'blood vessel endothelial cell', 'dendritic cell',
                     'B cell', 'muscle cell', 'type I pneumocyte', 'regulatory T cell', 'ciliated cell',
                     'endothelial cell of lymphatic vessel', 'plasma cell']
    if dataname == "Stewart_Fetal": # 18
        class_set = ['mesenchymal stem cell', 'stromal cell', 'myofibroblast cell', 'kidney resident macrophage',
                     'endothelial cell', 'fibroblast', 'conventional dendritic cell', 'monocyte', 'natural killer cell',
                     'epithelial cell of proximal tubule', 'kidney pelvis urothelial cell', 'CD4-positive helper T cell',
                     'B cell', 'glomerular visceral epithelial cell', 'neutrophil', 'lymphocyte', 'megakaryocyte',
                     'kidney loop of Henle epithelial cell']
    if dataname == "He_Lone_Bone": # 11
        class_set = ['cell of skeletal muscle', 'stromal cell of bone marrow', 'mesenchymal cell', 'Chondroblast', 'Chondrocyte',
                     'obsolete osteoprogenitor cell', 'stromal cell', 'muscle cell', 'macrophage', 'endothelial cell', 'erythrocyte']
    if dataname == "Vento-Tormo_10x": # 17
        class_set = ['stromal cell', 'decidual natural killer cell', 'placental villous trophoblast', 'T cell', 'macrophage',
                     'trophoblast cell', 'fibroblast', 'natural killer cell', 'Hofbauer cell', 'endothelial cell',
                     'syncytiotrophoblast cell', 'monocyte', 'dendritic cell', 'glandular epithelial cell', 'plasma cell',
                     'granulocyte', 'lymphocyte']
    return class_set


def normalize(adata, highly_genes = None, size_factors=True, normalize_input=True, logtrans_input=True):
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, subset=True)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


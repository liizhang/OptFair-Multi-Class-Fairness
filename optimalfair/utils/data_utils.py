import copy
import torch
import random
import json
import numpy as np
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import datasets as hf_datasets
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from sklearn.preprocessing import MinMaxScaler
import pickle
import requests
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
import zipfile
import multiprocessing
import threading
import urllib.request
import sys
import re


class FairDataset(Dataset):
    def __init__(self, X, Y, A, n_class=None, n_group=None, weight=None):
        self.X = X
        self.Y = Y
        self.A = A
        self.weight = weight
        self.data_info = self.get_data_info( Y, A )
        # self.data_num = self.X.shape[0]

    def __getitem__(self, index):

        X = torch.tensor(self.X[index], dtype=torch.float)
        Y = torch.tensor(self.Y[index], dtype=torch.long)
        A = torch.tensor(self.A[index], dtype=torch.long)

        if self.weight is not None:
            assert len(self.weight) == self.X.shape[0]
            weight = self.weight[index]
            return (X, Y, A, weight)
        return (X, Y, A)

    def __len__(self):
        if hasattr(self.X, "shape"):
            return self.X.shape[0]
        return None
    
    def dim(self):
        if hasattr(self.X, "shape"):
            return self.X.shape[1:]
        return None
    
    def get_data_info(self, Y, A):
        unique_A = np.unique(A)
        unique_Y = np.unique(Y)
        self.n_group = len(unique_A)
        self.n_class = len(unique_Y)

        df_YA = pd.DataFrame({'Y': Y.ravel(), 'A': A.ravel()})

        # print(Y.shape, A.shape)

        info = pd.DataFrame(index=unique_A.astype(int), columns=unique_Y.astype(int))
        for a in unique_A:
            for y in unique_Y:
                info.at[a, y] = df_YA[(df_YA['A'] == a) & (df_YA['Y'] == y)].shape[0]

        return info

class FairTextDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, max_length=256, weight=None,
                 text_col="bio", label_col="title", group_col="gender"):
        self.ds = raw_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.weight = weight

        self.text_col = text_col
        self.label_col = label_col
        self.group_col = group_col

        self.Y = np.asarray(self.ds[self.label_col], dtype=np.int64)
        self.A = np.asarray(self.ds[self.group_col], dtype=np.int64)
        self.data_info = self.get_data_info(self.Y, self.A)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = self.ds[self.text_col][idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )

        x = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }
        if "token_type_ids" in enc:
            x["token_type_ids"] = torch.tensor(enc["token_type_ids"], dtype=torch.long)

        y = torch.tensor(self.Y[idx], dtype=torch.long)
        a = torch.tensor(self.A[idx], dtype=torch.long)

        if self.weight is not None:
            w = torch.tensor(self.weight[idx], dtype=torch.float)
            return (x, y, a, w)
        return (x, y, a)

    def get_data_info(self, Y, A):
        unique_A = np.unique(A)
        unique_Y = np.unique(Y)
        self.n_group = len(unique_A)
        self.n_class = len(unique_Y)

        df_YA = pd.DataFrame({'Y': Y.ravel(), 'A': A.ravel()})
        info = pd.DataFrame(index=unique_A.astype(int), columns=unique_Y.astype(int))
        for a in unique_A:
            for y in unique_Y:
                info.at[a, y] = df_YA[(df_YA['A'] == a) & (df_YA['Y'] == y)].shape[0]
        return info



def mkdir(*args: str) -> tuple:
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    return args

def get_data(options):
    """ 
    Returns train and test datasets:
    """

    data_name = options['data'].lower()
    data_settings = {"sensitive_attr": options['sensitive_attr'], "generate": options['generate_data']}
    options.update(data_settings)

    if data_name == 'adult':
        raw_path = "data/adult/raw_data/"
        train_path = raw_path + "train.csv"
        test_path = raw_path + "test.csv"
        processed_path = "data/adult/processed_data/"
        processed_data_path = processed_path + "processed_adult.npy"
        if os.path.exists(train_path) and os.path.exists(test_path) and not data_settings.get('generate',False):
            options['data_exist'] = 1
            data = np.load(processed_data_path, allow_pickle=True).item()
            X, Y, A = data['X'], data['Y'], data['A']
        else:
            mkdir(processed_path)
            if os.path.exists(train_path) and os.path.exists(test_path):
                pass
            else:
                adult_process()
            df = pd.concat([pd.read_csv(train_path),pd.read_csv(test_path)], axis=0)
            X, Y = df.drop('salary', axis=1).to_numpy().astype(np.float32),  df['salary'].to_numpy().astype(np.float32)
            colname = df.drop('salary', axis=1).columns.tolist()
            if data_settings['sensitive_attr'] == 'gender-race':
                sensitive_attr = 'sex-race'
                X, A, Y = adult_get_sensitive_feature(X, colname, sensitive_attr, Y)
            elif data_settings['sensitive_attr'] == 'gender':
                sensitive_attr = 'sex'
                X, A = adult_get_sensitive_feature(X, colname, sensitive_attr)
            elif data_settings['sensitive_attr'] == 'race':
                sensitive_attr = 'race'
                X, A = adult_get_sensitive_feature(X, colname, sensitive_attr)
            np.save(processed_data_path,{"X" : X, "Y" : Y, "A" : A})
        
        train_data, val_data, test_data, n_group, n_class = split(X, Y, A)


    elif data_name == 'celeba':
        processed_path = f"data/celeba/processed_data/sensitive_attr_{data_settings['sensitive_attr']}_multiclass/"
        processed_data_path = processed_path + "processed_celeba.npy"
        options['data_save_path'] = processed_path
        if os.path.exists(processed_data_path) and not data_settings.get('generate',False):
            data = np.load(processed_data_path, allow_pickle=True).item()
            X, Y, A = data['X'], data['Y'], data['A']
            options['data_exist'] = 1
        else:
            mkdir(processed_path)
            if data_settings['sensitive_attr'] == 'sex':
                sensitive_attr = 'Male'
            elif data_settings['sensitive_attr'] == 'age':
                sensitive_attr = 'Young'
            elif data_settings['sensitive_attr'] == 'sex-race':
                sensitive_attr = ['Male', 'Pale_Skin']
            elif data_settings['sensitive_attr'] == 'race':
                sensitive_attr = 'Pale_Skin'

            X, Y, A = celeba_data_processing(sensitive_attr, multiclass=True)
            np.save(processed_data_path,{"X" : X, "Y" : Y, "A" : A})

        train_data, val_data, test_data, n_group, n_class = split(X, Y, A)

        print('celeba data processed.')
        
    elif data_name == 'enem':
        print("use enem.")
        X, A, Y = enem_process(data_settings['sensitive_attr'], n_classes=5)
        train_data, val_data, test_data, n_group, n_class = split(X, Y, A)

    elif data_name == 'acs':

        X, Y, label_names, A, sensitive_attr_names = acsincome_process(2, data_settings['sensitive_attr'])
        train_data, val_data, test_data, n_group, n_class = split(X, Y, A)
        print(f'y class:{Y.shape}, a class:{A.shape}')
    
    elif data_name == 'compas':

        X, Y, A = compas_1_data_processing(data_settings['sensitive_attr'])
        train_data, val_data, test_data, n_group, n_class = split(X, Y, A)
    
    else:
        raise ValueError('Not support dataset {}!'.format(data_name))

    print_statistics_info(train_data, val_data, test_data)
    return train_data, val_data, test_data, n_group, n_class

def enem_process(sensitive_attr, n_classes=5):
    enem_path = 'data/enem/raw_data/microdados_enem_2020/DADOS/' #changed to 2020
    enem_file = 'MICRODADOS_ENEM_2020.csv' #changed for 2020
    label = ['NU_NOTA_CH'] ## Labels could be: NU_NOTA_CH=human science, NU_NOTA_LC=languages&codes, NU_NOTA_MT=math, NU_NOTA_CN=natural science
    group_attribute = ['TP_COR_RACA','TP_SEXO']
    question_vars = ['Q00'+str(x) if x<10 else 'Q0' + str(x) for x in range(1,25)] #changed for 2020
    domestic_vars = ['SG_UF_PROVA', 'TP_FAIXA_ETARIA'] #changed for 2020
    all_vars = label+group_attribute+question_vars+domestic_vars

    n_sample = 1400000

    if n_classes == 2:
        n_groups = 2
        multigroup = False
    elif n_classes == 5:
        n_groups = 5
        multigroup = True
    
    fname = 'data/enem/processed_data/enem-'+str(n_classes) + '-g' + str(n_groups) + '-' + str(n_sample) + '-20.pkl'

    if os.path.isfile(fname):
        df = pd.read_pickle(fname)
    else:
        df = load_enem(enem_path, enem_file, all_vars, label, n_sample, n_classes, multigroup=multigroup)
        if not os.path.exists(fname):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
        df.to_pickle(fname)

    df['gradebin'] = df['gradebin'].astype(int)

    label_name = 'gradebin'
    if sensitive_attr == 'race':
        protected_attrs = 'racebin'
    else:
        raise ValueError('Not support sensitive attribute {}!'.format(sensitive_attr))

    X = df.drop(columns=[label_name,protected_attrs]).to_numpy()
    A = df[protected_attrs].to_numpy()
    Y = df[label_name].to_numpy()

    return X,A,Y


def get_idx_wo_protected(feature_names, protected_attrs):
    idx_wo_protected = set(range(len(feature_names)))
    protected_attr_idx = [feature_names.index(x) for x in protected_attrs]
    idx_wo_protected = list(idx_wo_protected - set(protected_attr_idx))
    return idx_wo_protected

def get_idx_w_protected(feature_names):
    return list(set(range(len(feature_names))))

def get_idx_protected(feature_names, protected_attrs):
    protected_attr_idx = [feature_names.index(x) for x in protected_attrs]
    idx_protected = list(set(protected_attr_idx))
    return idx_protected


def load_enem(file_path, filename, features, grade_attribute, n_sample, n_classes, multigroup=False):
    ## load csv
    df = pd.read_csv(file_path+filename, encoding='cp860', sep=';')
    # print('Original Dataset Shape:', df.shape)

    ## Remove all entries that were absent or were eliminated in at least one exam
    ix = ~df[['TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT']].applymap(lambda x: False if x == 1.0 else True).any(axis=1)
    df = df.loc[ix, :]

    ## Remove "treineiros" -- these are individuals that marked that they are taking the exam "only to test their knowledge". It is not uncommon for students to take the ENEM in the middle of high school as a dry run
    df = df.loc[df['IN_TREINEIRO'] == 0, :]

    ## drop eliminated features
    df.drop(['TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT', 'IN_TREINEIRO'], axis=1, inplace=True)

    ## subsitute race by names
    # race_names = ['N/A', 'Branca', 'Preta', 'Parda', 'Amarela', 'Indigena']
    race_names = [np.nan, 'Branca', 'Preta', 'Parda', 'Amarela', 'Indigena']
    df['TP_COR_RACA'] = df.loc[:, ['TP_COR_RACA']].applymap(lambda x: race_names[x]).copy()

    ## remove repeated exam takers
    ## This pre-processing step significantly reduces the dataset.
    df = df.loc[df.TP_ST_CONCLUSAO.isin([1])]

    ## select features
    df = df[features]

    ## Dropping all rows or columns with missing values
    df = df.dropna()

    ## Creating racebin & gradebin & sexbin variable
    df['gradebin'] = construct_grade(df, grade_attribute, n_classes)
    if multigroup:
        df['racebin'] = construct_race(df, 'TP_COR_RACA')
    else:
        df['racebin'] =np.logical_or((df['TP_COR_RACA'] == 'Branca').values, (df['TP_COR_RACA'] == 'Amarela').values).astype(int)
    df['sexbin'] = (df['TP_SEXO'] == 'M').astype(int)

    df.drop([grade_attribute[0], 'TP_COR_RACA', 'TP_SEXO'], axis=1, inplace=True)

    ## encode answers to questionaires
    ## Q005 is 'Including yourself, how many people currently live in your household?'
    question_vars = ['Q00' + str(x) if x < 10 else 'Q0' + str(x) for x in range(1, 25)]
    for q in question_vars:
        if q != 'Q005':
            df_q = pd.get_dummies(df[q], prefix=q)
            df.drop([q], axis=1, inplace=True)
            df = pd.concat([df, df_q.iloc[:, :-1]], axis=1)
            
    ## check if age range ('TP_FAIXA_ETARIA') is within attributes
    if 'TP_FAIXA_ETARIA' in features:
        q = 'TP_FAIXA_ETARIA'
        df_q = pd.get_dummies(df[q], prefix=q)
        df.drop([q], axis=1, inplace=True)
        df = pd.concat([df, df_q.iloc[:, :-1]], axis=1)

    ## encode SG_UF_PROVA (state where exam was taken)
    df_res = pd.get_dummies(df['SG_UF_PROVA'], prefix='SG_UF_PROVA')
    df.drop(['SG_UF_PROVA'], axis=1, inplace=True)
    df = pd.concat([df, df_res], axis=1)

    df = df.dropna()
    ## Scaling ##
    scaler = MinMaxScaler()
    scale_columns = list(set(df.columns.values) - set(['gradebin', 'racebin']))
    df[scale_columns] = pd.DataFrame(scaler.fit_transform(df[scale_columns]), columns=scale_columns, index=df.index)
    # print('Preprocessed Dataset Shape:', df.shape)

    df = df.sample(n=min(n_sample, df.shape[0]), axis=0, replace=False)
    return df


def construct_race(df, protected_attribute):
    race_dict = {'Branca': 0, 'Preta': 1, 'Parda': 2, 'Amarela': 3, 'Indigena': 4} 
    # race_dict = {'Branca': 0, 'Preta': 1, 'Parda': 2, 'Amarela': 3}
    # changed to match ENEM 2020 numbering
    return df[protected_attribute].map(race_dict)

def construct_grade(df, grade_attribute, n):
    v = df[grade_attribute[0]].values
    quantiles = np.nanquantile(v, np.linspace(0.0, 1.0, n+1))
    return pd.cut(v, quantiles, labels=np.arange(n))

def _download_biasbios_if_needed(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.pickle")
    test_path  = os.path.join(data_dir, "test.pickle")
    dev_path   = os.path.join(data_dir, "dev.pickle")
    if any(not os.path.exists(p) for p in [train_path, test_path, dev_path]):
        urllib.request.urlretrieve("https://storage.googleapis.com/ai2i/nullspace/biasbios/train.pickle", train_path)
        urllib.request.urlretrieve("https://storage.googleapis.com/ai2i/nullspace/biasbios/test.pickle",  test_path)
        urllib.request.urlretrieve("https://storage.googleapis.com/ai2i/nullspace/biasbios/dev.pickle",   dev_path)
    return train_path, test_path, dev_path

def load_biasbios_full_dataset(
    data_dir: str,
    add_sensitive_attribute: bool = True,
):
    """
    return: train_ds, val_ds, test_ds, tokenizer, label_names, group_names

    - train_ds / val_ds / test_ds: FairTextDataset
    - tokenizer: AutoTokenizer
    - label_names / group_names: list
    """

    # 1. label & group
    label_names = [
        "accountant", "architect", "attorney", "chiropractor", "comedian",
        "composer", "dentist", "dietitian", "dj", "filmmaker",
        "interior_designer", "journalist", "model", "nurse", "painter",
        "paralegal", "pastor", "personal_trainer", "photographer", "physician",
        "poet", "professor", "psychologist", "rapper", "software_engineer",
        "surgeon", "teacher", "yoga_teacher"
    ]
    group_names = ["female", "male"]

    features = hf_datasets.Features({
        "bio": hf_datasets.Value("string"),
        "title": hf_datasets.ClassLabel(names=label_names),
        "gender": hf_datasets.ClassLabel(names=group_names),
    })

    # load dataset
    train_path, test_path, dev_path = _download_biasbios_if_needed(data_dir)
    def _load_split(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        rows = {"bio": [], "title": [], "gender": []}
        for row in data:
            gender = "female" if row["g"] == "f" else "male"
            title  = row["p"]
            bio_text = row["hard_text_untokenized"]
            if add_sensitive_attribute:
                bio_text = gender.capitalize() + ". " + bio_text
            rows["bio"].append(bio_text)
            rows["title"].append(title)
            rows["gender"].append(gender)
        return rows

    r1 = _load_split(train_path)
    r2 = _load_split(test_path)
    r3 = _load_split(dev_path)

    rows_full = {
        "bio":    r1["bio"]    + r2["bio"]    + r3["bio"],
        "title":  r1["title"]  + r2["title"]  + r3["title"],
        "gender": r1["gender"] + r2["gender"] + r3["gender"],
    }
    full_ds = hf_datasets.Dataset.from_dict(rows_full, features=features)
    return full_ds, label_names, group_names

def mean_pool(last_hidden, attention_mask):
    # last_hidden: [B,L,H], attention_mask: [B,L]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)     # [B,L,1]
    summed = (last_hidden * mask).sum(dim=1)                     # [B,H]
    denom = mask.sum(dim=1).clamp(min=1e-6)                      # [B,1]
    return summed / denom

def build_full_embeddings_meanpool(
    data_dir="data/biasbios/raw_data",
    out_dir="data/biasbios/emb_full_bert_meanpool",
    model_name="bert-base-uncased",
    max_length=256,
    batch_size=64,
    dtype="float16",          # "float16" or "float32"
    device="cuda",
    num_workers=4,
    add_sensitive_attribute=True,
):
    os.makedirs(out_dir, exist_ok=True)

    # (a) load full dataset
    full_ds, label_names, group_names = load_biasbios_full_dataset(
        data_dir=data_dir,
        add_sensitive_attribute=add_sensitive_attribute
    )

    # (b) tokenizer + tokenize (batched map faster than on-the-fly)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tok_fn(examples):
        return tokenizer(
            examples["bio"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = full_ds.map(tok_fn, batched=True, remove_columns=["bio"], desc="Tokenizing(full)")
    # labels / groups
    Y = np.asarray(tokenized["title"], dtype=np.int64)   # N
    A = np.asarray(tokenized["gender"], dtype=np.int64)  # N

    # (c) model frozen
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dl = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    # (d) get hidden size H
    H = model.config.hidden_size
    N = len(tokenized)

    # (e) prepare memmap N x H
    emb_dtype = np.float16 if dtype == "float16" else np.float32
    emb_path = os.path.join(out_dir, "X_emb.dat")
    X_emb = np.memmap(emb_path, mode="w+", dtype=emb_dtype, shape=(N, H))

    # (f) run forward & mean pool
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    write_ptr = 0

    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ["input_ids", "attention_mask", "token_type_ids"]}

            out = model(**batch)
            last = out.last_hidden_state.to(torch_dtype)                 # [B,L,H]
            emb = mean_pool(last, batch["attention_mask"])               # [B,H]

            bsz = emb.size(0)
            X_emb[write_ptr:write_ptr+bsz] = emb.detach().cpu().numpy().astype(emb_dtype, copy=False)
            write_ptr += bsz

    X_emb.flush()

    # (g) save meta + Y/A
    np.save(os.path.join(out_dir, "Y.npy"), Y)
    np.save(os.path.join(out_dir, "A.npy"), A)
    meta = {
        "model_name": model_name,
        "max_length": max_length,
        "pooling": "mean",
        "dtype": dtype,
        "N": int(N),
        "H": int(H),
        "n_class": int(len(label_names)),  # 28
        "n_group": int(len(group_names)),  # 2
        "add_sensitive_attribute": bool(add_sensitive_attribute),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return X_emb, Y, A, meta


def adult_process():
    # Adult
    sensitive_attributes = ['sex']
    categorical_attributes = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    continuous_attributes = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    features_to_keep = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 
                'native-country', 'salary']
    label_name = 'salary'

    path_adult = 'data/adult/raw_data/adult.data'
    path_test = 'data/adult/raw_data/adult.test'

    adult = process_adult_csv(path_adult, label_name, ' >50K', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep)
    test = process_adult_csv(path_test, label_name, ' >50K.', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep) # the distribution is very different from training distribution
    test['native-country_ Holand-Netherlands'] = 0
    test = test[adult.columns]

    adult_num_features = len(adult.columns)-1

    adult.to_csv('data/adult/raw_data/train.csv', index=None)
    test.to_csv('data/adult/raw_data/test.csv', index=None)
    
def process_adult_csv(filename, label_name, favorable_class, sensitive_attributes, privileged_classes, categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = 'infer', columns = None):
    """
    from https://github.com/yzeng58/Improving-Fairness-via-Federated-Learning/blob/main/FedFB/DP_load_dataset.py
    process the adult file: scale, one-hot encode
    only support binary sensitive attributes -> [gender, race] -> 4 sensitive groups 
    """
    skiprows = 1 if filename.endswith('test') else 0
    df = pd.read_csv(os.path.join(filename), delimiter = ',', header = header, na_values = na_values, skiprows=skiprows)
    if header == None: df.columns = columns
    df = df[features_to_keep]

    # apply one-hot encoding to convert the categorical attributes into vectors
    df = pd.get_dummies(df, columns = categorical_attributes)

    # normalize numerical attributes to the range within [0, 1]
    def scale(vec):
        minimum = min(vec)
        maximum = max(vec)
        return (vec-minimum)/(maximum-minimum)
    
    df[continuous_attributes] = df[continuous_attributes].apply(scale, axis = 0)
    df.loc[df[label_name] != favorable_class, label_name] = 0
    df.loc[df[label_name] == favorable_class, label_name] = 1
    df[label_name] = df[label_name].astype('category').cat.codes
    df['sex'] = df['sex'].map({' Male':0, ' Female':1}).astype('category')
    return df


def compas_1_data_processing(sensitive='sex-race'):
    #@title Load COMPAS dataset

    LABEL_COLUMN = 'two_year_recid'
    if sensitive == 'sex-race':
        sensitive_attributes = ['sex_Female', 'race_African-American']
    elif sensitive == 'race':
        sensitive_attributes = ['race_African-American']


    def get_data():
        data_path = "data/compas/raw_data/compas-scores-two-years.csv"
        df = pd.read_csv(data_path)
        FEATURES = [
            'age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex',
            'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid',
            'two_year_recid'
        ]
        df = df[FEATURES]
        df = df[df.days_b_screening_arrest <= 30]
        df = df[df.days_b_screening_arrest >= -30]
        df = df[df.is_recid != -1]
        df = df[df.c_charge_degree != 'O']
        df = df[df.score_text != 'N/A']
        continuous_features = [
            'priors_count', 'days_b_screening_arrest', 'is_recid', 'two_year_recid'
        ]
        continuous_to_categorical_features = ['age', 'decile_score', 'priors_count']
        categorical_features = ['c_charge_degree', 'race', 'score_text', 'sex']
        # continuous_to_categorical_features = [ 'priors_count']
        # categorical_features = ['c_charge_degree', 'race', 'sex']

        # Functions for preprocessing categorical and continuous columns.
        def binarize_categorical_columns(input_df, categorical_columns=[]):
            # Binarize categorical columns.
            binarized_df = pd.get_dummies(input_df, columns=categorical_columns)
            return binarized_df

        def bucketize_continuous_column(input_df, continuous_column_name, bins=None):
            input_df[continuous_column_name] = pd.cut(
                input_df[continuous_column_name], bins, labels=False)

        for c in continuous_to_categorical_features:
            b = [0] + list(np.percentile(df[c], [20, 40, 60, 80, 90, 100]))
            if c == 'priors_count':
                b = list(np.percentile(df[c], [0, 50, 70, 80, 90, 100]))
            bucketize_continuous_column(df, c, bins=b)

        # df = binarize_categorical_columns(
        #     df,
        #     categorical_columns=categorical_features)

        df = binarize_categorical_columns(
            df,
            categorical_columns=categorical_features +
            continuous_to_categorical_features)

        to_fill = [
            u'decile_score_0', u'decile_score_1', u'decile_score_2',
            u'decile_score_3', u'decile_score_4', u'decile_score_5'
        ]
        for i in range(len(to_fill) - 1):
            df[to_fill[i]] = df[to_fill[i:]].max(axis=1)
            
        to_fill = [
            u'priors_count_0.0', u'priors_count_1.0', u'priors_count_2.0',
            u'priors_count_3.0', u'priors_count_4.0'
        ]
        for i in range(len(to_fill) - 1):
            df[to_fill[i]] = df[to_fill[i:]].max(axis=1)

        print(df.columns)
        features = [
            u'days_b_screening_arrest', u'c_charge_degree_F', u'c_charge_degree_M',
            u'race_African-American', u'race_Asian', u'race_Caucasian',
            u'race_Hispanic', u'race_Native American', u'race_Other',
            u'score_text_High', u'score_text_Low', u'score_text_Medium',
            u'sex_Female', u'sex_Male', u'age_0', u'age_1', u'age_2', u'age_3',
            u'age_4', u'age_5', u'decile_score_0', u'decile_score_1',
            u'decile_score_2', u'decile_score_3', u'decile_score_4',
            u'decile_score_5', u'priors_count_0.0', u'priors_count_1.0',
            u'priors_count_2.0', u'priors_count_3.0', u'priors_count_4.0'
        ]

        # # new
        # features = [
        #     u'days_b_screening_arrest', u'c_charge_degree_F', u'c_charge_degree_M',
        #     u'race_African-American', u'race_Asian', u'race_Caucasian',
        #     u'race_Hispanic', u'race_Native American', u'race_Other',
        #     u'sex_Female', u'sex_Male', u'age', u'priors_count_0.0', u'priors_count_1.0',
        #     u'priors_count_2.0', u'priors_count_3.0', u'priors_count_4.0'
        # ]
        # print(len(features))

        label = ['two_year_recid']

        df = df[features + label]
        return df, features, label

    df, feature_names, label_column = get_data()

    from sklearn.utils import shuffle
    df = shuffle(df)
    N = len(df)
    # train_df = df[:int(N * 0.66)]
    # test_df = df[int(N * 0.66):]

    X_compas = np.array(df[feature_names],dtype=np.float32)
    y_compas = np.array(df[label_column], dtype=np.float32).flatten()

    if sensitive == 'sex-race':

        # 0: male non-black, 1: female non-black, 2: male black, 3: female black
        A_compas = np.array(df[sensitive_attributes[0]] + df[sensitive_attributes[1]] * 2).flatten()

        sex_race_idx = [i for i, value in enumerate(feature_names) if (value.startswith('race') or value.startswith('sex')) ==True]
        X_compas = np.delete(X_compas, sex_race_idx, axis=1)

        print(X_compas.shape)
    
    elif sensitive == 'race':
        # 0: non-black, 1: black
        A_compas = df[sensitive_attributes[0]].to_numpy(dtype=np.float32).flatten()

        sen_idx = [i for i, value in enumerate(feature_names) if value.startswith('race')==True]
        X_compas = np.delete(X_compas, sen_idx, axis=1)

    print("compas process end.")

    return X_compas, y_compas,  A_compas

class CelebAMMapDataset(Dataset):
    def __init__(self, image_paths, image_dict, transform, multiclass=False):
        self.image_paths = image_paths
        self.image_dict = image_dict
        self.transform = transform
        self.multiclass = multiclass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img_id = os.path.basename(p)  # '000001.jpg'
        y, a, other = self.image_dict[img_id]

        img = Image.open(p).convert("RGB")
        x = self.transform(img)  # torch.float32, (3,128,128)

        if self.multiclass:
            y = y * 2 + other

        # Return tensor plus integer labels; DataLoader batches/stacks them automatically.
        return x, int(y), int(a)


def celeba_data_processing(sensitive_attr, batch_size=256, mmap_file="processed_data.mmap", multiclass=False, num_workers=4):
    path = os.path.join('data', 'celeba', 'processed_data')
    os.makedirs(path, exist_ok=True)

    # ---------- read attributes ----------
    attr_file = os.path.join('data', 'celeba', 'raw_data', 'list_attr_celeba.txt')
    with open(attr_file, 'r', encoding='utf-8') as f:
        attributes = f.read().splitlines()

    tar = 'Smiling'
    other_tar = 'Big_Nose'
    header = attributes[1].split()
    target_idx = header.index(tar)
    other_idx = header.index(other_tar)

    if isinstance(sensitive_attr, list):
        assert len(sensitive_attr) == 2
        sen_idx = [header.index(sen) for sen in sensitive_attr]
    else:
        sen_idx = header.index(sensitive_attr)

    image = {}
    for line in attributes[2:]:
        info = line.split()
        if not info:
            continue
        image_id = info[0]
        vals = info[1:]

        tar_img = (int(vals[target_idx]) + 1) // 2
        other_img_val = (int(vals[other_idx]) + 1) // 2

        if isinstance(sensitive_attr, list):
            sen_img1 = (int(vals[sen_idx[0]]) + 1) // 2
            sen_img2 = (int(vals[sen_idx[1]]) + 1) // 2
            sen_img = sen_img1 + 2 * sen_img2
        else:
            sen_img = (int(vals[sen_idx]) + 1) // 2

        image[image_id] = (tar_img, sen_img, other_img_val)

    # ---------- list images ----------
    images_path = Path(os.path.join('data', 'celeba', 'raw_data', 'img_align_celeba'))
    images_list = sorted(images_path.glob('*.jpg'))
    assert len(images_list) > 0, f"[ERROR] No jpg found in: {images_path.resolve()}"

    images_ids = [str(x) for x in images_list]
    N = len(images_ids)

    # Extra sanity check: sample one image id and verify it can match a label.
    test_id = os.path.basename(images_ids[0])
    assert test_id in image, f"[ERROR] image id {test_id} not found in attribute dict. Check list_attr_celeba.txt"

    # ---------- transform ----------
    transform = transforms.Compose([
        transforms.CenterCrop((178, 178)),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ---------- DataLoader for robust multiprocessing ----------
    ds = CelebAMMapDataset(images_ids, image, transform, multiclass=multiclass)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    # ---------- mmap ----------
    shape = (N, 3, 128, 128)
    X = np.memmap(mmap_file, dtype=np.float16, mode='w+', shape=shape)
    Y = np.empty((N,), dtype=np.int64)
    A = np.empty((N,), dtype=np.int64)

    print(f"start. N={N}, batch_size={batch_size}, num_workers={num_workers}")

    idx = 0
    for xb, yb, ab in dl:
        bs = xb.size(0)
        # xb: torch float32 -> numpy float16
        X[idx:idx+bs] = xb.numpy().astype(np.float16)
        Y[idx:idx+bs] = np.asarray(yb, dtype=np.int64)
        A[idx:idx+bs] = np.asarray(ab, dtype=np.int64)
        idx += bs

    assert idx == N, f"[ERROR] filled {idx} != {N}"

    # flush
    X.flush()

    Y = Y.reshape(-1, 1)
    A = A.reshape(-1, 1)

    print("end.")
    print("X shape:", X.shape, "dtype:", X.dtype)
    print("Y shape:", Y.shape, "A shape:", A.shape)

    unique, counts = np.unique(Y, return_counts=True)
    print("Value counts:", dict(zip(unique.tolist(), counts.tolist())))

    return X, Y, A


class multiprocess_img_load(object):
    def __init__(self, img_paths:list, transform, img_size=(3,128,128), n_thread=None) -> None:
        self.image_paths = img_paths
        self.img_size = img_size
        self.num_img = len(img_paths)
        self._mutex_put = threading.Lock()
        self.n_thread = n_thread if (n_thread is not None) else max(1, multiprocessing.cpu_count() - 2)
        self.transform = transform
    
    def get_imgs(self):
        self._buffer = np.zeros([self.num_img]+list(self.img_size))
        batch_size = round(self.num_img / self.n_thread)
        batch_idx = []
        for i in range(self.n_thread):
            idx = list(range(i * batch_size, (i+1) * batch_size)) if (i+1) * batch_size <= self.num_img else list(range(i * batch_size, self.num_img))
            batch_idx.append(idx)
        t_list = []
        for tid in range(self.n_thread):
            img_ids = list(range(tid * batch_size, (tid+1) * batch_size)) if (tid+1) * batch_size <= self.num_img else range(tid * batch_size, self.num_img)
            img_target = [self.image_paths[i] for i in img_ids]
            t = threading.Thread(target=self.load_image, args=(img_target, img_ids))
            t_list.append(t)
            t.start()

        for t in t_list:
            t.join()

        del t_list

        return self._buffer

    def load_image(self, img_names, img_ids):
        batch_images = np.vstack([np.expand_dims(self.transform(Image.open(img)).numpy(), axis=0) for img in img_names])
        self._mutex_put.acquire()
        self._buffer[img_ids] = batch_images
        self._mutex_put.release()

def celeba_split(data_indices, X ,Y ,A):
    split_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    for i in range(len(data_indices)):
        split_data['users'].append(str(i))
        split_data['user_data'][str(i)] = {'x':X[data_indices[i],:],
                                      'y':Y[data_indices[i]],
                                      'A':A[data_indices[i]]}
        split_data['num_samples'].append(len(data_indices[i]))
    return split_data

def get_unsaved_data(data_split):
    for client in data_split['user_data']:
        X = np.array(data_split['user_data'][client]["x"]).astype(np.float32)
        Y = np.array(data_split['user_data'][client]["y"]).astype(np.float32).reshape(-1,1)
        A = np.array(data_split['user_data'][client]["A"]).astype(np.float32).reshape(-1,1)
        dataset = FairDataset(X, Y, A)
        data_split['user_data'][client] = dataset
    return data_split

def bank_get_sensitive_feature(X, colname, sensitive_attr):
    if sensitive_attr == 'age':
        attr_idx = colname.index(sensitive_attr)
        A = X[:,attr_idx]
        X = np.delete(X, attr_idx, axis = 1)
    return X,A

def compas_get_sensitive_feature(X, colname, sensitive_attr):
    sex_attr = []
    race_attr = []
    for col in colname:
        if col.startswith('race'):
            race_attr.append(col)
        elif col.startswith('sex'):
            sex_attr.append(col)
    
    if sensitive_attr == 'sex':
        attr_idx = [colname.index(attr) for attr in sex_attr]
        A = np.argmax(X[:,attr_idx], axis =1 )  # [1: Male, 0: Female]
        X = np.delete(X, attr_idx, axis = 1)
    elif sensitive_attr == 'race':
        attr_idx = [colname.index(attr) for attr in race_attr]
        A = np.argmax(X[:,attr_idx], axis = 1) # ['African-American': 0,'Caucasian': 1,'Asian':2,'Hispanic':3]
        A[A>=1] = 1
        X = np.delete(X, attr_idx, axis = 1)
    elif sensitive_attr == 'non-sex':
        attr_idx = [colname.index(attr) for attr in sex_attr]
        A = np.argmax(X[:,attr_idx], axi = 1) 
    elif sensitive_attr == 'non-race':
        attr_idx = [colname.index(attr) for attr in race_attr] 
        A = np.argmax(X[:,attr_idx], axis = 1)
    return X, A

def split_celeba_data(ids: list):
    path = 'data/celeba/raw_data/img_align_celeba/'
    imgs = np.concatenate([np.expand_dims(np.array(Image.open(path + id)).transpose(2,0,1), axis=0) for id in ids], axis=0)
    
    return imgs

def partition_test_data(separation, targets):
    label_num = len(set(targets))
    targets_numpy = np.array(targets, dtype=np.int32)
    data_indices = [[] for _ in range(len(separation[0]))]
    data_idx_for_each_label = [
        np.where(targets_numpy == i)[0] for i in range(label_num)
    ]
    for k in range(label_num):
        distrib_cumsum = (np.cumsum(separation[k]) * len(data_idx_for_each_label[k])).astype(int)[:-1]
        data_indices = [
            np.concatenate((idx_j, idx.tolist())).astype(np.int64)
            for idx_j, idx in zip(
                data_indices, np.split(data_idx_for_each_label[k], distrib_cumsum)
            )
        ]
    
    return data_indices

def split(X ,Y ,A, prop=None):
    n = X.shape[0]
    Y = Y.reshape(-1,1)
    A = A.reshape(-1,1)
    n_group = len(np.unique(A))
    n_class = len(np.unique(Y))
    is_A_valid = np.array_equal(np.unique(A), np.arange(n_group))
    is_Y_valid = np.array_equal(np.unique(Y), np.arange(n_class))
    assert is_A_valid == True and is_Y_valid ==True

    indices = np.random.permutation(n)
    train_index, val_index, test_index = indices[:int(n*0.6)], indices[int(n*0.4):int(n*0.6)], indices[int(n*0.6):int(n*1)]
    train_data = FairDataset(X[train_index,:], Y[train_index,:], A[train_index,:])
    val_data = FairDataset(X[val_index,:], Y[val_index,:], A[val_index,:])
    test_data = FairDataset(X[test_index,:], Y[test_index,:], A[test_index,:])
    return train_data, val_data, test_data, n_group, n_class


def adult_get_sensitive_feature(X, colname, sensitive, Y=None):
    sex_attr = 'sex'
    race_attr = []
    for col in colname:
        if col.startswith('race'):
            race_attr.append(col)
    if sensitive == "race":
        attr = 'race_ White'
        attr_idx = colname.index(attr)
        A = np.array(X[:,attr_idx])
        # print(np.unique(A))
        del_idx = [colname.index(attr) for attr in race_attr]
        X = np.delete(X, del_idx, axis = 1)
    elif sensitive == "sex":
        attr_idx = colname.index(sex_attr)
        A = X[:, attr_idx] # [1: female, 0: male]
        X = np.delete(X, attr_idx, axis = 1)
    elif sensitive == "none-race":
        attr_idx = [colname.index(attr) for attr in race_attr]
        A = np.argmax(X[:,attr_idx], axis =1 ) 
    elif sensitive == "none-sex":
        attr_idx = colname.index(sex_attr)
        A = X[:, attr_idx] # [1: female, 0: male]
    elif sensitive == "sex-race":
        race_idx = [colname.index(attr) for attr in race_attr] 
        race_unused = [colname.index(attr) for attr in ['race_ Amer-Indian-Eskimo', 'race_ Asian-Pac-Islander', 'race_ Other']] 
        Y = Y[np.sum(X[:,race_unused],axis=1) == 0]
        X = X[np.sum(X[:,race_unused],axis=1) == 0,:]
        sex_idx = colname.index(sex_attr)
        A = (np.argmax(X[:,race_idx], axis =1) + X[:,sex_idx]) - 2
        X = np.delete(X, race_idx + [sex_idx], axis = 1)
        return X,A,Y


    else:
        print("error sensitive attr")
        exit()
    
    return X, A

def read_data(path, name=None, sensitive_process=None):
    split_train = {'users': [], 'user_data':{}, 'num_samples':{}}
    split_val = copy.deepcopy(split_train)
    split_test = copy.deepcopy(split_train)

    if name == 'celeba':
        data_split = np.load(path, allow_pickle=True).item()
    elif name == 'enem':
        with open(path, 'rb') as f:
            data_split = pickle.load(f)
    else:
        with open(path, 'rb') as file:
            data_split = json.load(file)

    for client in data_split['users']:
        split_train['users'].append(client)
        split_val['users'].append(client)
        split_test['users'].append(client)

        X = np.array(data_split['user_data'][client]["x"]).astype(np.float32)

        Y = np.array(data_split['user_data'][client]["y"]).astype(np.float32).reshape(-1,1)

        A = np.array(data_split['user_data'][client]["A"]).astype(np.float32).reshape(-1,1)

        n = np.arange(X.shape[0])
        indices = np.random.permutation(n)
        train_index, val_index, test_index = indices[:int(len(n)*0.6)], indices[:int(len(n)*0.6)], indices[int(len(n)*0.6):int(len(n)*1)]
        split_train['user_data'][client] = FairDataset(X[train_index,:], Y[train_index,:], A[train_index,:])
        split_val['user_data'][client] = FairDataset(X[val_index,:], Y[val_index,:], A[val_index,:])
        split_test['user_data'][client] = FairDataset(X[test_index,:], Y[test_index,:], A[test_index,:])

        split_train['num_samples'][client] = len(train_index)
        split_val['num_samples'][client] = len(val_index)
        split_test['num_samples'][client] = len(test_index)
        
    return split_train,split_val,split_test
    
def celeba_read_data(data_split, name=None, sensitive_process=None):
    split_train = {'users': [], 'user_data':{}, 'num_samples':{}}
    split_val = copy.deepcopy(split_train)
    split_test = copy.deepcopy(split_train)

    for client in data_split['users']:
        split_train['users'].append(client)
        split_val['users'].append(client)
        split_test['users'].append(client)

        X = np.array(data_split['user_data'][client]["x"]).astype(np.float32)

        Y = np.array(data_split['user_data'][client]["y"]).astype(np.float32).reshape(-1,1)

        A = np.array(data_split['user_data'][client]["A"]).astype(np.float32).reshape(-1,1)

        n = np.arange(X.shape[0])
        indices = np.random.permutation(n)
        train_index, val_index, test_index = indices[:int(len(n)*0.6)], indices[:int(len(n)*0.6)], indices[int(len(n)*0.6):]
        split_train['user_data'][client] = FairDataset(X[train_index,:], Y[train_index,:], A[train_index,:])
        split_val['user_data'][client] = FairDataset(X[val_index,:], Y[val_index,:], A[val_index,:])
        split_test['user_data'][client] = FairDataset(X[test_index,:], Y[test_index,:], A[test_index,:])

        split_train['num_samples'][client] = len(train_index)
        split_val['num_samples'][client] = len(val_index)
        split_test['num_samples'][client] = len(test_index)
    
    return split_train,split_val,split_test


def acsincome_process(n_classes=2, sensitive_attr='sex', remove_sensitive_attr=True):

    if sensitive_attr == 'sex':
        sensitive_attr = 'SEX' 
    elif sensitive_attr == 'race':
        sensitive_attr = 'RAC1P' 

    from fairlearn.datasets import fetch_acs_income
    target = 'PINCP'
    features = [
        'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX',
        'RAC1P'
    ]
    categories = {
        "COW": {
            1.0: ("Employee of a private for-profit company or"
                "business, or of an individual, for wages,"
                "salary, or commissions"),
            2.0: ("Employee of a private not-for-profit, tax-exempt,"
                "or charitable organization"),
            3.0:
                "Local government employee (city, county, etc.)",
            4.0:
                "State government employee",
            5.0:
                "Federal government employee",
            6.0: ("Self-employed in own not incorporated business,"
                "professional practice, or farm"),
            7.0: ("Self-employed in own incorporated business,"
                "professional practice or farm"),
            8.0:
                "Working without pay in family business or farm",
            9.0:
                "Unemployed and last worked 5 years ago or earlier or never worked",
        },
        "SCHL": {
            1.0: "No schooling completed",
            2.0: "Nursery school, preschool",
            3.0: "Kindergarten",
            4.0: "Grade 1",
            5.0: "Grade 2",
            6.0: "Grade 3",
            7.0: "Grade 4",
            8.0: "Grade 5",
            9.0: "Grade 6",
            10.0: "Grade 7",
            11.0: "Grade 8",
            12.0: "Grade 9",
            13.0: "Grade 10",
            14.0: "Grade 11",
            15.0: "12th grade - no diploma",
            16.0: "Regular high school diploma",
            17.0: "GED or alternative credential",
            18.0: "Some college, but less than 1 year",
            19.0: "1 or more years of college credit, no degree",
            20.0: "Associate's degree",
            21.0: "Bachelor's degree",
            22.0: "Master's degree",
            23.0: "Professional degree beyond a bachelor's degree",
            24.0: "Doctorate degree",
        },
        "MAR": {
            1.0: "Married",
            2.0: "Widowed",
            3.0: "Divorced",
            4.0: "Separated",
            5.0: "Never married or under 15 years old",
        },
        "SEX": {
            1.0: "Male",
            2.0: "Female"
        },
        "RAC1P": {
            1.0: "White alone",
            2.0: "Black or African American alone",
            3.0: "American Indian alone",
            4.0: "Alaska Native alone",
            5.0: ("American Indian and Alaska Native tribes specified;"
                "or American Indian or Alaska Native,"
                "not specified and no other"),
            6.0: "Asian alone",
            7.0: "Native Hawaiian and Other Pacific Islander alone",
            8.0: "Some Other Race alone",
            9.0: "Two or More Races",
        },
    }

    cache_dir = 'data/acs'
    os.makedirs(cache_dir, exist_ok=True)
    pkl_path = os.path.join(cache_dir, 'acsincome5.pkl')
    if os.path.exists(pkl_path):
        print(f"Found existing file: {pkl_path}. Loading from disk...")
        with open(pkl_path, "rb") as f:
            data, labels, label_names, groups, group_names = pickle.load(f)
            data = data.to_numpy(dtype='float32')
        return data, labels, label_names, groups, group_names
    print(f"processing continues.")

    # Download or load the dataset
    csv_path = os.path.join(cache_dir, 'acs_income.csv')
    if os.path.exists(csv_path):
        print(f"Found existing file: {csv_path}. Loading from disk...")
        df = pd.read_csv(csv_path)
    else:
        print(f"{csv_path} not found. Downloading ACSIncome dataset...")
        # return pandas DataFrame
        X, y = fetch_acs_income(as_frame=True, return_X_y=True)
        df = X.copy()
        df["PINCP"] = y
        df.to_csv(csv_path, index=False)
        print(f"Downloaded and saved to {csv_path}.")
    print(f"Dataset shape: {df.shape} (rows, columns)")

    if n_classes == 2:
        label_names = ["<=50K", ">50K"]
        target_transform = lambda x: (x > 50000).astype(int)

    else:
        # Compute empirical CDF of PINCP
        x = np.sort(df[target])
        y = np.arange(len(x)) / float(len(x))

        # Partition into bins containing roughly the same number of samples
        partitions = np.array([
            x[np.argmax(y >= q)] for q in np.arange(1 / n_classes, 1, 1 / n_classes)
        ] + [np.inf])

        label_names = [f'[0, {partitions[0]})'] + [
            f'[{partitions[i]}, {partitions[i+1]})'
            for i in range(len(partitions) - 1)
        ]
        target_transform = lambda x: np.argmax(
            np.array(x)[:, None] < partitions[None, :], axis=1)

    if sensitive_attr == 'RAC1P':
        # Combine RAC1P categories 3, 4, 5, and 6, 7, and 8, 9 into new categories
        # 10, 11, and 12 respectively, due to small sample size in some groups.
        # This is also consistent with the UCI Adult dataset.
        categories['RAC1P'][10.0] = "American Indian or Alaska Native alone"
        categories['RAC1P'][
            11.0] = "Asian, Native Hawaiian or Other Pacific Islander alone"
        categories['RAC1P'][12.0] = "Other"
        df['RAC1P'] = df['RAC1P'].replace([3.0, 4.0, 5.0], 10.0)
        df['RAC1P'] = df['RAC1P'].replace([6.0, 7.0], 11.0)
        df['RAC1P'] = df['RAC1P'].replace([8.0, 9.0], 12.0)


    data, labels, groups = folktables.BasicProblem(
      features=features,
      target=target,
      target_transform=target_transform,
      group=sensitive_attr,
      postprocess=lambda x: np.nan_to_num(x, -1),
    ).df_to_pandas(df, categories=categories, dummies=True)

    labels = labels.values.squeeze()
    groups = groups.values.squeeze()

    group_names, groups = np.unique(groups, return_inverse=True)
    group_names = [categories[sensitive_attr][n] for n in group_names]

    if remove_sensitive_attr:
        data.drop(columns=list(data.filter(regex=f'^{sensitive_attr}')),
                inplace=True)
        
    data = df.values

    return data, labels, label_names, groups, group_names

def print_statistics_info(train_data, val_data, test_data):
    # Print statistics info
    print("=== Train Data ===")
    print("Number of samples:", len(train_data))
    print("Info table:")
    print(train_data.data_info)

    print("\n=== Validation Data ===")
    print("Number of samples:", len(val_data))
    print("Info table:")
    print(val_data.data_info)

    print("\n=== Test Data ===")
    print("Number of samples:", len(test_data))
    print("Info table:")
    print(test_data.data_info)

    # return {'train_info'}

def add_symmetric_noise_to_A(A, noise_rate, num_classes=None, seed=None):
    """
    Add symmetric noise to a multiclass variable A with shape (n, 1).

    Parameters:
        A: numpy.ndarray, shape (n, 1)
           Elements should take values 0, 1, ..., m.
        noise_rate: float
           Noise rate in the range [0, 1].
        num_classes: int or None
           Total number of classes. If A takes values 0 through m,
           then num_classes = m + 1. If None, infer it from A.max() + 1.
        seed: int or None
           Random seed.

    Returns:
        A_noisy: numpy.ndarray, shape (n, 1)
            The array after noise is added.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("A must be a numpy.ndarray")

    if A.ndim != 2 or A.shape[1] != 1:
        raise ValueError("A must have shape (n, 1)")

    if not (0.0 <= noise_rate <= 1.0):
        raise ValueError("noise_rate must be in [0, 1]")

    if num_classes is None:
        num_classes = int(A.max()) + 1

    if np.any(A < 0) or np.any(A >= num_classes):
        raise ValueError("Elements in A must be in {0, 1, ..., num_classes-1}")

    rng = np.random.default_rng(seed)
    A_noisy = A.copy()

    n = A.shape[0]

    # Decide which samples need to be flipped.
    flip_mask = rng.random(n) < noise_rate
    flip_indices = np.where(flip_mask)[0]

    for idx in flip_indices:
        original_label = A[idx, 0]

        # Exclude the original class from candidate labels.
        candidate_labels = list(range(num_classes))
        candidate_labels.remove(original_label)

        # Uniformly sample one label from the remaining classes.
        A_noisy[idx, 0] = rng.choice(candidate_labels)

    return A_noisy

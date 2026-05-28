import numpy as np
import torch
import copy
import time
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from optimalfair.utils.models import *
from optimalfair.utils.model_utils import *

class basicprocess(object):
    def __init__(self, dataset, options, name = ''):
        # dataset
        self.train_data, self.val_data, self.test_data, self.n_group, self.n_class = dataset  

        # Basic parameters
        self.gpu = options['gpu']
        self.device = options['device']
        self.num_round = options['num_round']
        self.eval_round = options['eval_round']
        self.lr = options['lr']
        self.batch_size = options['batch_size']
        self.load_model = options['load_model']
        self.verbose = options['print_result']
        # self.data_info = options['data_info']
        self.fair_metric = options['fairness_metric']
        self.fair_bound = options['fairness_bound']
        self.options = options
        self.algo = options['algorithm']  

    # convert numpy/list to torch.Tensor
    def _to_tensor(self, X, dtype=torch.float32):
        if isinstance(X, torch.Tensor):
            return X.to(dtype)
        return torch.as_tensor(X, dtype=dtype)

    def fit_A_give_X(self):
        input_dim = self.train_data.X.shape[1]
        num_classes = self.n_group

        clf = Classifier(
            model_fn=choose_model(self.options),
            model_kwargs={"input_shape": input_dim, "output_dim": num_classes},
            lr=self.lr,
            epochs=self.num_round,
            batch_size=self.batch_size,
            options=self.options,
            load_trained=self.load_model,
            task_name='predict_P_A_X',
        )

        clf.fit(self.train_data.X, self.train_data.A)
        return clf

    def fit_Y_give_XA(self):
        X = torch.tensor(self.train_data.X)
        A = torch.tensor(self.train_data.A)
        XA = torch.cat([X,A], dim=1)
        Y = torch.tensor(self.train_data.Y)
        input_dim = XA.shape[1]
        num_classes = self.n_class

        clf = Classifier(
            model_fn=choose_model(self.options),
            model_kwargs={"input_shape": input_dim, "output_dim": num_classes},
            lr=self.lr,
            epochs=self.num_round,
            batch_size=self.batch_size,
            options=self.options,
            load_trained=self.load_model,
            task_name='predict_P_Y_XA',
        )
        clf.fit(torch.cat(torch.tensor(self.train_data.X), torch.tensor(self.train_data.A)), torch.tensor(self.train_data.Y))
        return clf
    
    def fit_A_give_XY(self):
        XY = concat_xy_onehot_numpy_to_torch(self.train_data.X, self.train_data.Y, self.n_class)
        input_dim = XY.shape[1]
        num_classes = self.n_group

        clf = Classifier(
            model_fn=choose_model(self.options),
            model_kwargs={"input_shape": input_dim, "output_dim": num_classes},
            lr=self.lr,
            epochs=self.num_round,
            batch_size=self.batch_size,
            options=self.options,
            load_trained=self.load_model,
            task_name='predict_P_A_XY',
        )

        clf.fit(XY, torch.tensor(self.train_data.A))

        return clf

    def fit_Y_give_X(self):
        input_dim = 1
        for dim in self.train_data.X.shape[1:]:
            input_dim *= dim
        if self.options['model'] == 'resnet':
            input_dim = 3
        num_classes = self.n_class

        clf = Classifier(
            model_fn=choose_model(self.options),
            model_kwargs={"input_shape": input_dim, "output_dim": num_classes},
            lr=self.lr,
            epochs=self.num_round,
            batch_size=self.batch_size,
            options=self.options,
            load_trained=self.load_model,
            task_name='predict_P_Y_X',
        )

        clf.fit(torch.tensor(self.train_data.X), self.train_data.Y)
        return clf

    def fit_AY_give_X(self):
        AY = self.train_data.A * self.n_class + self.train_data.Y
        assert AY.shape == self.train_data.Y.shape
        input_dim = self.train_data.X.shape[1]
        num_classes = self.n_class * self.n_group

        clf = Classifier(
            model_fn=choose_model(self.options),
            model_kwargs={"input_shape": input_dim, "output_dim": num_classes},
            lr=self.lr,
            epochs=self.num_round,
            batch_size=self.batch_size,
            options=self.options,
            load_trained=self.load_model,
            task_name='predict_P_AY_X',
        )

        clf.fit(self.train_data.X, AY)
        return clf

    def fair_evaluate(self, Y, pred_Y, A):

        # statistics
        group_confusion_metrix = confusion_matrix(Y, pred_Y, A, n_classes=self.n_class, n_groups=self.n_group, normalize='all')
        a_counts = np.bincount(A.astype(np.int64))  # count occurrences of each group
        y_counts = np.bincount(Y.astype(np.int64))  # count occurrences of each class
        total_samples = len(Y)
        a_priors = a_counts / total_samples  # class prior probabilities
        y_priors = y_counts / total_samples
        joint_index = Y * self.n_group + A
        joint_counts = np.bincount(joint_index.astype(np.int64)).reshape(self.n_class, self.n_group)
        joint_priors = (joint_counts / total_samples)

        if self.fair_metric == 'dp':
            DP_test = delta_sp(pred_Y, A, n_classes=self.n_class, n_groups=self.n_group)
            
            # group-wise constraint
            matrix = np.zeros((self.n_group, self.n_class))
            for a in range (self.n_group):
                for y in range (self.n_class):
                    matrix[a,y] = np.sum( [ ( a_priors[a_upper] - (a_upper == a) )*group_confusion_metrix[a_upper,:,y] for a_upper in range (self.n_group) ])
            diff = np.max(np.abs(matrix))
        
        elif self.fair_metric == 'eop':
            eop_test = delta_eopp(Y,pred_Y, A, n_classes=self.n_class, n_groups=self.n_group)
            
            matrix = np.zeros((self.n_group, self.n_class))
            for a in range (self.n_group):
                for y in range (self.n_class):
                    # matrix[y,a] = np.abs(np.sum( (pred_Y==y).ravel() * (data.Y==y).ravel() ) / np.sum(data.Y==y) - group_confusion_metrix[a,y,y]/np.sum(group_confusion_metrix[a,y,:]))
                    matrix[a,y] = np.sum( [ a_priors[a_upper] * ( 1/y_priors[y] - (a_upper == a) / joint_priors[y,a] ) * group_confusion_metrix[a_upper,y,y] for a_upper in range (self.n_group) ])
            diff = np.max(np.abs(matrix))
        elif self.fair_metric == 'eo':
            eo_test = delta_eopp(Y,pred_Y.ravel(), A, n_classes=self.n_class, n_groups=self.n_group)
            
            matrix = np.zeros((self.n_group,self.n_class,self.n_class))
            for a in range(self.n_group):
                for y in range(self.n_class):
                    for yy in range(self.n_class):
                        matrix[a, y, yy] = np.sum( [np.sum( [ a_priors[a_upper] * ( 1/y_priors[yy] - (a_upper == a) / joint_priors[yy,a] ) * group_confusion_metrix[a_upper,yy,y] for a_upper in range (self.n_group) ])] )
            diff = np.max(np.abs(matrix))
        return diff, matrix
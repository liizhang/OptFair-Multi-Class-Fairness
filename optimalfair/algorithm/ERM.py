import numpy as np
import torch
import copy
import time
import torch.nn as nn
from tqdm import tqdm
from optimalfair.algorithm.classifierbase import basicprocess
from optimalfair.utils.models import *
from optimalfair.utils.model_utils import *

class classifier(basicprocess):
    def __init__(self, dataset, options, name=''):
        super().__init__(dataset, options, name)

    def train(self):
        # init logger
        run_dir = make_run_dir(self.options)
        logger = JSONLStepLogger(run_dir, config={"lr": self.lr, "bs": self.batch_size})

        self.model_Y_give_X = self.fit_Y_give_X()
        self.train_pred = self.model_Y_give_X.predict(self.train_data.X)
        self.test_pred = self.model_Y_give_X.predict(self.test_data.X)

        test_acc = self.model_Y_give_X.evaluate(X=self.test_data.X, y=self.test_data.Y)
        test_diff, test_matrix = self.fair_evaluate(Y=self.test_data.Y.ravel(), pred_Y= self.test_pred.ravel(),A=self.test_data.A.ravel())
        logger.log_step(round='final', metrics={"acc": float(test_acc) ,"fairness_level": float(test_diff)},)
        if self.verbose:
                print(f"[Eval] Task: fair in-processing, test accuracy = {test_acc:.4f}, disparity = {test_diff:.4f}")
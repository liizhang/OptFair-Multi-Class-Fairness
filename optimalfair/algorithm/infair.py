import numpy as np
import torch
import torch.nn.functional as F
import copy
import time
import torch.nn as nn
from tqdm import tqdm
from optimalfair.algorithm.classifierbase import basicprocess
from optimalfair.utils.models import *
from optimalfair.utils.model_utils import *
from transformers import DataCollatorWithPadding, BatchEncoding

class classifier(basicprocess):
    def __init__(self, dataset, options, name=''):
        super().__init__(dataset, options, name)
        self.dual_params_lr = self.options['dual_params_lr']
        self.dual_params_bound = self.options['dual_params_bound']
        self.inner_round = self.options['inner_round']
        self.tau = self.options['tau']

        # init dual parameter
        if self.fair_metric == 'dp':
            self.lamb = torch.ones(self.n_group, self.n_class, requires_grad=False, device=self.device) * 0.0001
        elif self.fair_metric == 'eop':
            self.lamb = torch.ones(self.n_group, self.n_class, requires_grad=False, device=self.device) * 0.0001
        elif self.fair_metric == 'eo':
            self.lamb = torch.ones(self.n_group, self.n_class, self.n_class, requires_grad=False, device=self.device) * 0.0001
    
    # convert numpy/list to torch.Tensor
    def _to_tensor(self, X, dtype=torch.float32):
        if isinstance(X, torch.Tensor):
            return X.to(dtype)
        return torch.as_tensor(X, dtype=dtype)

    def train(self):
        # build model
        input_dim = self.train_data.X.shape[1]
        
        self.model_fn = choose_model(self.options)
        self.model_kwargs={"input_shape": input_dim, "output_dim": self.n_class}
        self.model_ = self.model_fn(**self.model_kwargs).to(self.device)

        # collected prediction
        self.pred_collected = torch.zeros((self.num_round, len(self.test_data)))

        # build DataLoader
        dl = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        # build optimizer
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)

        # init logger
        run_dir = make_run_dir(self.options)
        logger = JSONLStepLogger(run_dir, config={"lr": self.lr, "bs": self.batch_size,
                                                   "dual_params_lr":self.dual_params_lr, "tau":self.tau,
                                                   "dual_params_bound":self.dual_params_bound,"inner_round":self.inner_round})


        # training
        for self.round in range(self.num_round):
            
            # cal matrix
            M_a_lamb = self.get_cal_matrix(self.lamb).detach().to(self.device)
            
            # every epoch lr *= gamma
            for pg in opt.param_groups:
                pg["lr"] = self.lr
            dual_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.8)
            
            self.model_.train()
            for epoch in range(self.inner_round):
                total_loss = 0.0
                for (x, y, a) in dl:
                    if self.gpu:
                        x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)

                    outputs = F.log_softmax(self.model_(x), dim=1)
                    
                    n_batch = outputs.shape[0]
                    M_sel = M_a_lamb[a.view(-1).long()] # (n, m, m)
                    # print(M_a_lamb.device, a.device)
                    rows = M_sel[torch.arange(n_batch), y.view(-1).long()] # (n, m)
                    
                    loss = -(outputs * rows).sum(dim=1).mean()
                    
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total_loss += loss.item() * n_batch

                # decay once per epoch
                dual_scheduler.step()
                
                epoch_loss = total_loss / len(self.train_data)
                if self.verbose:
                    print(f"[Train] Task: fair in-processing, [Round {self.round+1}/{self.num_round}], [Epoch {epoch+1}/{self.inner_round}] loss={epoch_loss:.4f}")
            
            # if self.round % self.options['eval_round'] == 0:
            self.model_.eval()
            test_acc, test_diff, _ = self.model_eval(self.test_data, round=self.round)
            if self.verbose:
                print(f"[Eval] Task: fair in-processing, test accuracy = {test_acc:.4f}, disparity = {test_diff:.4f}")
            logger.log_step(round=self.round, metrics={"acc": float(test_acc) ,"fairness_level": float(test_diff)},)

            if self.round % 10 == 0 and self.round > 0:
                test_acc, test_diff, _ = self.model_eval(self.test_data, round='final')
                if self.verbose:
                    print(f"[Eval] Task: fair in-processing, test accuracy = {test_acc:.4f}, disparity = {test_diff:.4f}")
                logger.log_step(round=str(self.round)+'test', metrics={"acc": float(test_acc) ,"fairness_level": float(test_diff)},)


            train_acc, train_diff, train_constraint_matrix = self.model_eval(self.train_data)
            if self.verbose:
                print(f"[Train] Task: fair in-processing, train accuracy = {train_acc:.4f}, disparity = {train_diff:.4f}")

            # update lambda: proximal gradient algorithm
            with torch.no_grad():
                lam_t = self.lamb.detach().clone() + self.dual_params_lr * torch.tensor(train_constraint_matrix).to(self.lamb.device)
                projected = prox_l1_plus_l1ball(v=lam_t.contiguous().view(-1), eta=self.dual_params_lr, xi=self.fair_bound, B=self.dual_params_bound)
                self.lamb.copy_(projected.view(self.lamb.shape))

        self.model_.eval()
        test_acc, test_diff, _ = self.model_eval(self.test_data, round='final')
        if self.verbose:
            print(f"[Eval] Task: fair in-processing, test accuracy = {test_acc:.4f}, disparity = {test_diff:.4f}")
        logger.log_step(round='final', metrics={"acc": float(test_acc) ,"fairness_level": float(test_diff)},)


    def get_cal_matrix(self, lamb):

        device = lamb.device
        dtype  = lamb.dtype

        a_used = self.train_data.A.ravel().astype(np.int64)
        y_used = self.train_data.Y.ravel().astype(np.int64)

        # empirical statistics
        a_priors = torch.tensor(np.bincount(a_used) / len(a_used), device=device, dtype=dtype)
        y_priors = torch.tensor(np.bincount(y_used) / len(y_used), device=device, dtype=dtype)

        joint_index = torch.tensor(y_used * self.n_group + a_used)
        joint_counts = torch.bincount(joint_index, minlength=self.n_class * self.n_group).reshape(self.n_class, self.n_group)

        joint_priors = (joint_counts.float() / len(y_used)).to(device=device, dtype=dtype)

        # calibration matrix
        if self.fair_metric == 'dp':
            assert lamb.numel() == self.n_class * self.n_group
            G_a_k = torch.zeros(self.n_group, self.n_class, self.n_group, self.n_class, self.n_class, device=device, dtype=dtype)
            for a in range(self.n_group):
                for y in range(self.n_class):
                    for a_upper in range(self.n_group):
                        G_a_k[a,y,a_upper,:,y] = a_priors[a_upper] - torch.tensor(a_upper == a, dtype=torch.float32)
            
            S = torch.einsum('ay,aygij->gij', lamb, G_a_k)

            I = torch.eye(self.n_class, device=lamb.device, dtype=lamb.dtype).unsqueeze(0)  # (1, m, m)
            M_a_lamb = I - S / a_priors.view(-1, 1, 1)  

        elif self.fair_metric == 'eop':
            assert lamb.numel() == self.n_class * self.n_group
            G_a_k = torch.zeros(self.n_group, self.n_class, self.n_group, self.n_class, self.n_class, device=device, dtype=dtype)
            for a in range(self.n_group):
                for y in range(self.n_class):
                    for a_upper in range(self.n_group):
                        G_a_k[a,y,a_upper,y,y] = a_priors[a_upper] * ( 1 / y_priors[y] - torch.tensor(a_upper == a, dtype=torch.float32)/joint_priors[y,a] ) 

            S = torch.einsum('ay,aygij->gij', lamb, G_a_k)
            I = torch.eye(self.n_class, device=lamb.device, dtype=lamb.dtype).unsqueeze(0)  # (1, m, m)
            M_a_lamb = I - S / a_priors.view(-1, 1, 1)  
        
        elif self.fair_metric == 'eo':
            assert lamb.numel() == self.n_class * self.n_class * self.n_group
            G_a_k = torch.zeros(self.n_group, self.n_class, self.n_class, self.n_group, self.n_class, self.n_class, device=device, dtype=dtype)
            for a in range(self.n_group):
                for y in range(self.n_class):
                    for yy in range(self.n_class):
                        for a_upper in range(self.n_group):
                            G_a_k[a,y,yy,a_upper,yy,y] = a_priors[a_upper] * ( 1 / y_priors[yy] - torch.tensor(a_upper == a, dtype=torch.float32)/joint_priors[yy,a] ) 

            S = torch.einsum('ayk,aykgij->gij', lamb, G_a_k)
            I = torch.eye(self.n_class, device=lamb.device, dtype=lamb.dtype).unsqueeze(0)  # (1, m, m)
            M_a_lamb = I - S / a_priors.view(-1, 1, 1) 

        # positive calibration
        global_min = M_a_lamb.min().item()
        if global_min < 0:
            shift = -global_min + 0.00001
            M_a_lamb += shift
        
        print(f'calibrated matrix, max: {M_a_lamb.max().item(), M_a_lamb.min().item()}')
        print(f'lambda, max: {self.lamb.max().item(), self.lamb.min().item()}')
        return M_a_lamb
    

    @torch.no_grad()
    def model_eval(self, data, ensemble=False,round=None):
        assert self.model_ is not None
        dataLoader = DataLoader(data, batch_size = self.batch_size, shuffle = False)
        self.model_.eval()
        test_correct = test_num = 0.0
        preds = []

        for (x, y, a) in dataLoader:
            if self.gpu:
                x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)
            
            if ensemble == False:
                ## Deterministic Classifier
                pred = self.model_(x)
                _, predicted = torch.max(pred, 1)
                preds.append(predicted.detach().cpu())
                correct = predicted.eq(y.view(-1).long()).sum().item()
                batch_size = y.size(0)

                test_correct += correct # total correct, not average
                test_num += batch_size 

            else:
                assert self.model_pool is not None

        acc = (test_correct / test_num)
        pred_class = torch.cat(preds, dim=0).numpy()

        if round and round != 'final':
            assert pred_class.shape[0] == len(self.test_data)
            self.pred_collected[round] = torch.tensor(pred_class)
        elif round == 'final':
            n = len(data)
            random_round = torch.randint(max(0,self.round - 10), self.round, (n,))
            random_preds = self.pred_collected[random_round, torch.arange(n)]
            pred_class = random_preds.numpy()
            acc = np.equal(pred_class, data.Y.ravel()).sum() / len(data)

        diff, matrix = self.fair_evaluate(Y=data.Y.ravel(), pred_Y=pred_class.ravel(),A=data.A.ravel())
        
        return acc, diff, matrix
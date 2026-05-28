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
        self.dual_params_lr = self.options['dual_params_lr']
        self.dual_params_bound = self.options['dual_params_bound']
        self.tau = self.options['tau']
        self.post_num_round = self.options['post_num_round']
        self.post_batch_size = self.options['post_batch_size']
        # self.inner_round = self.options['inner_round']

        # init dual parameter
        if self.fair_metric == 'dp':
            self.lamb = torch.ones(self.n_group,self.n_class, requires_grad=False) * 0.001
        elif self.fair_metric == 'eop':
            self.lamb = torch.ones(self.n_group,self.n_class, requires_grad=False) * 0.001
        elif self.fair_metric == 'eo':
            self.lamb = torch.ones(self.n_group,self.n_class, self.n_class, requires_grad=False) * 0.001

    def train(self):
        # train model for AY_given_X
        self.model_AY_given_X = self.fit_AY_give_X()
        self.post_AY_given_X = self.model_AY_given_X.predict_proba(self.train_data.X).view(-1,self.n_group,self.n_class).to(self.device) # (N,a,m)

        self.post_Y_given_X = self.post_AY_given_X.sum(dim=1)
        den = self.post_Y_given_X.unsqueeze(1)  
        self.post_A_given_XY = torch.where(
                            den > 0,
                            self.post_AY_given_X / den,            # (N, a, m)
                            torch.zeros_like(self.post_AY_given_X) # den==0 
                        )
        
        self.test_AY_given_X = self.model_AY_given_X.predict_proba(self.test_data.X).view(-1,self.n_group,self.n_class).to(self.device) # (N,a,m)
        self.test_Y_given_X = self.test_AY_given_X.sum(dim=1)
        den = self.test_Y_given_X.unsqueeze(1)  
        self.test_A_given_XY = torch.where(
                            den > 0,
                            self.test_AY_given_X / den,            # (N, a, m)
                            torch.zeros_like(self.test_AY_given_X) # den==0 
                        )

        assert self.post_A_given_XY.shape[0] == self.post_Y_given_X.shape[0]

        print('Finished fitting model for post-processing.')

        post_dataset = TensorDataset(self.post_A_given_XY, self.post_Y_given_X)
        if self.post_batch_size == 0:
            self.post_batch_size = len(post_dataset)
        dl = DataLoader(dataset=post_dataset, batch_size=self.post_batch_size, shuffle=False)

        # init dual parameter
        self.lamb = torch.nn.Parameter( self.lamb.to(device=self.device))
        self.opt_dual = torch.optim.SGD([self.lamb], lr=self.dual_params_lr)
        # every epoch lr *= gamma
        self.dual_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt_dual, gamma=0.95)

        # init logger
        run_dir = make_run_dir(self.options)
        logger = JSONLStepLogger(run_dir, config={"lr": self.lr, "bs": self.batch_size, 
                                                  "dual_params_lr":self.dual_params_lr, "tau":self.tau,
                                                   "dual_params_bound":self.dual_params_bound,})

        test_acc, test_diff, _ = self.model_eval('test')
        if self.verbose:
            print(f"[Eval] Task: fair Post-processing, test accuracy = {test_acc:.4f}, disparity = {test_diff:.4f}")


        # training
        for self.round in range(self.post_num_round):

            for p_A_given_XY, p_Y_given_X in dl:
                if self.gpu:
                    p_A_given_XY, p_Y_given_X = p_A_given_XY.to(self.device), p_Y_given_X.to(self.device)
                
                batch_size = p_Y_given_X.shape[0]
                M_a_lamb = self.get_cal_matrix(self.lamb) # (a,m,m)
                score = torch.zeros(batch_size, self.n_class, self.n_class, device=self.device) # (N,m,m)
                for a in range(self.n_group):
                    rho_a_x = torch.diag_embed(p_A_given_XY[:,a,:]) # (N,m,m)
                    M_a = M_a_lamb[a] # (m,m)
                    score += rho_a_x @ M_a
                beta = (p_Y_given_X.unsqueeze(1) @ score).squeeze(1)
                log_sum_exp = torch.logsumexp(beta/self.tau, dim=1, keepdim=True) # (N,1)

                # proximal
                loss_H = self.tau * log_sum_exp.mean()

                self.opt_dual.zero_grad()
                loss_H.backward()
                self.opt_dual.step()

                with torch.no_grad():
                    # proximal
                    lam_t = self.lamb.detach().clone().contiguous().view(-1)
                    projected = prox_l1_plus_l1ball(v=lam_t, eta=self.dual_params_lr, xi=self.fair_bound, B=self.dual_params_bound)

                    projected = projected.view_as(self.lamb).to(self.device)
                    self.lamb.copy_(projected)

            if self.verbose:
                print(f"[Train] Task: Post-processing, [Round {self.round+1}/{self.post_num_round}], lambda norm:{self.lamb.abs().sum():.4f}")
            
            if self.round % self.options['eval_round'] == 0:
                train_acc, train_diff, _ = self.model_eval('train')
                if self.verbose:
                    print(f"[Eval] Task: fair Post-processing, train accuracy = {train_acc:.4f}, disparity = {train_diff:.4f}")
                test_acc, test_diff, _ = self.model_eval('test')
                if self.verbose:
                    print(f"[Eval] Task: fair Post-processing, test accuracy = {test_acc:.4f}, disparity = {test_diff:.4f}")

                logger.log_step(round=self.round, metrics={"acc": float(test_acc) ,"fairness_level": float(test_diff)},)
        
        
        test_acc, test_diff, _ = self.model_eval('test')
        if self.verbose:
            print(f"[Final Eval] Task: fair Post-processing, test accuracy = {test_acc:.4f}, disparity = {test_diff:.4f}")

        logger.log_step(round='final', metrics={"acc": float(test_acc) ,"fairness_level": float(test_diff)},)


    def get_cal_matrix(self, lamb):

        device = lamb.device
        dtype  = lamb.dtype

        a_used = self.train_data.A.ravel().astype(np.int64)
        y_used = self.train_data.Y.ravel().astype(np.int64)

        # sensitive statistics probabilities
        a_priors = torch.tensor(np.bincount(a_used) / len(a_used), device=device, dtype=dtype)
        
        # label statistics probabilities
        y_priors = torch.tensor(np.bincount(y_used) / len(y_used), device=device, dtype=dtype)

        joint_index = torch.tensor(y_used * self.n_group + a_used)
        joint_counts = torch.bincount(joint_index, minlength=self.n_class * self.n_group).reshape(self.n_class, self.n_group)

        joint_priors = (joint_counts.float() / len(y_used)).to(device=device, dtype=dtype)

            
        if self.fair_metric == 'dp':
            assert lamb.numel() == self.n_class * self.n_group
            M_a_lamb = torch.zeros(self.n_group, self.n_class, self.n_class, device=device, dtype=dtype)
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
            M_a_lamb = torch.zeros(self.n_group, self.n_class, self.n_class, device=device, dtype=dtype)
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
            M_a_lamb = torch.zeros(self.n_group, self.n_class, self.n_class, device=device, dtype=dtype)
            G_a_k = torch.zeros(self.n_group, self.n_class, self.n_class, self.n_group, self.n_class, self.n_class, device=device, dtype=dtype)
            for a in range(self.n_group):
                for y in range(self.n_class):
                    for yy in range(self.n_class):
                        for a_upper in range(self.n_group):
                            G_a_k[a,y,yy,a_upper,yy,y] = a_priors[a_upper] * ( 1 / y_priors[yy] - torch.tensor(a_upper == a, dtype=torch.float32)/joint_priors[yy,a] ) 

            S = torch.einsum('ayk,aykgij->gij', lamb, G_a_k)
            I = torch.eye(self.n_class, device=lamb.device, dtype=lamb.dtype).unsqueeze(0)  # (1, m, m)
            M_a_lamb = I - S / a_priors.view(-1, 1, 1) 
        
        return M_a_lamb
    

    @torch.no_grad()
    def model_eval(self, split='test'):
        assert self.lamb is not None

        if split == 'train':
            # assert self.post_AY_given_X is not None

            data = self.train_data
            dataset = TensorDataset(self.post_A_given_XY, self.post_Y_given_X, torch.tensor(data.Y))
            dl = DataLoader(dataset=dataset, batch_size=self.post_batch_size, shuffle=False)
        elif split == 'test':
            # assert self.test_AY_given_X is not None

            data = self.test_data
            dataset = TensorDataset(self.test_A_given_XY, self.test_Y_given_X, torch.tensor(data.Y))
            dl = DataLoader(dataset=dataset, batch_size=self.post_batch_size, shuffle=False)
        else:
            raise ValueError(f'Not support data split {split}!')

        test_correct = test_num = 0.0
        preds = []

        M_a_lamb = self.get_cal_matrix(self.lamb).to(self.device) # (a,m,m)

        for p_A_given_XY, p_Y_given_X, y in dl:
            if self.gpu:
                p_A_given_XY, p_Y_given_X, y = p_A_given_XY.to(self.device), p_Y_given_X.to(self.device), y.to(self.device)
                
            batch_size = p_Y_given_X.shape[0]
            score = torch.zeros(batch_size, self.n_class, self.n_class, device=self.device) # (N,m,m)
            for a in range(self.n_group):
                rho_a_x = torch.diag_embed(p_A_given_XY[:,a,:]) # (N,m,m)
                M_a = M_a_lamb[a] # (m,m)
                score += rho_a_x @ M_a
            beta = (p_Y_given_X.unsqueeze(1) @ score).squeeze(1)
            softmax = torch.softmax(beta/self.tau, dim=1)

            predicted = torch.multinomial(softmax, 1).view(-1)

            preds.append(predicted.detach().cpu())
            correct = predicted.eq(y.view(-1).long()).sum().item()
            batch_size = y.size(0)

            test_correct += correct # total correct, not average
            test_num += batch_size 

        acc = test_correct / test_num
        pred_class = torch.cat(preds, dim=0).numpy()

        diff, matrix = self.fair_evaluate(Y=data.Y.ravel(), pred_Y=pred_class.ravel(),A=data.A.ravel())
        
        return acc, diff, matrix
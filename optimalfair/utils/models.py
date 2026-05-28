import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
import os 
from optimalfair.utils.model_utils import mkdir

class Classifier(BaseEstimator, ClassifierMixin):
    """
    Description
    ------------------
    A sklearn-style classifier wrapper trained with PyTorch; accepts any injected nn.Module.
    """

    def __init__(
        self,
        model_fn,                      # your nn.Module constructor (callable)
        model_kwargs=None,             # kwargs for model_fn
        task_name=' ',                 # task name
        optimizer_fn=torch.optim.AdamW, # optimizer constructor
        optimizer_kwargs=None,         # kwargs for optimizer (except lr)
        lr=1e-3,                       # learning rate
        epochs=100,                     # number of epochs
        batch_size=64,                 # batch size
        device=None,                   # None -> auto-select
        loss_type="softCE",            # type of loss (CE, softCE, LA)
        random_state=None,             # random seed
        verbose=1,                     # verbosity
        options=None,                  # options
        label_smoothing=0.0,           # label smoothing parameter
        patience=0,                    # early stopping patience
        min_delta=1e-4,                # early stopping min delta
        load_trained=True,             # load training (for post-processing), if False, re-train model and save 
    ):
        
        self.model_fn = model_fn
        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self.task_name = task_name
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.loss_type = loss_type
        self.random_state = random_state
        self.verbose = verbose
        self.options = {} if options is None else dict(options)
        self.load_trained = load_trained
        self.label_smoothing = float(label_smoothing)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        # attributes set after fitting
        self.model_ = None
        self.classes_ = None
        self.class_to_index_ = None
        self.n_features_in_ = None
        self.class_priors = None  # class priors


    # convert numpy/list to torch.Tensor
    def _to_tensor(self, X, dtype=torch.float32):
        if isinstance(X, torch.Tensor):
            return X.to(dtype)
        return torch.as_tensor(X, dtype=dtype)

    # pick device
    def _pick_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # derive classes_ and mapping from y
    def _set_classes(self, y):
        try:
            if isinstance(y, torch.Tensor):
                y_np = y.detach().cpu().numpy()
            else:
                y_np = np.asarray(y)
        except Exception:
            y_np = np.asarray(y)

        if y_np.ndim == 2:
            if y_np.shape[1] == 1:
                y_np = y_np.ravel()
            else:
                y_np = np.argmax(y_np, axis=1)
        self.classes_ = np.unique(y_np)
        self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}
        # calculate class priors
        class_counts = np.bincount(y_np.astype(np.int64))  # count occurrences of each class
        total_samples = len(y_np)
        self.class_priors = class_counts / total_samples  # class prior probabilities

    # map arbitrary labels y to [0..C-1]
    def _encode_y(self, y):
        y_np = np.asarray(y)
        return np.vectorize(self.class_to_index_.get)(y_np)

    # choose loss criterion (CE)
    def _build_criterion(self, n_classes, sample_weight):
        # multiclass: use CrossEntropyLoss
        if sample_weight is None:
            return nn.CrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss(reduction='none')

    # convert model "outputs" to probabilities --
    def _to_proba(self, logits, n_classes):
        """
        logits of shape (N, C)
        """
        # use softmax by default to convert to probabilities
        prob = torch.softmax(logits, dim=1)
        return prob

    def fit(self, X, y):
        """
        Train the model
        X: (n_samples, ...) numpy or torch.Tensor
        y: (n_samples,)  labels (hashable)
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        device = self._pick_device()

        # set classes and feature info
        self._set_classes(y)
        y_encoded = self._encode_y(y)
        n_classes = len(self.classes_)
        X_t = self._to_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y_encoded, dtype=torch.long)

        # record n_features_in_ if X is 2D
        if hasattr(X_t, "shape") and X_t.ndim >= 2:
            self.n_features_in_ = X_t.shape[1]
        else:
            self.n_features_in_ = None

        # build model
        self.model_ = self.model_fn(**self.model_kwargs).to(device)
        model_path = f'saved_model/{self.options["data"]}/{self.task_name + '_' + self.options['model']}.pth'

        loaded_ok = False
        if self.load_trained and os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=device)
                self.model_.load_state_dict(state)
                loaded_ok = True
                if self.verbose:
                    print(f"[Info] Loaded {self.task_name} model from: {model_path}")
            except Exception as e:
                if self.verbose:
                    print(f"[Warn] Failed to load model from {model_path}: {e}")

        # if loaded, return now
        if loaded_ok:
            self.model_.eval()
            pred_acc = self.evaluate(X=X,y=y)
            return self

        # build optimizer
        opt = self.optimizer_fn(self.model_.parameters(), lr=self.lr, weight_decay=1e-4,**self.optimizer_kwargs)
        dual_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

        # choose loss
        # if self.loss_type == 'CE':
        #     criterion = nn.CrossEntropyLoss()  # standard cross-entropy

        # elif self.loss_type == 'softCE':
        #     # Explicit softmax cross-entropy: logsumexp - target_logit
        #     def criterion(logits, targets):
        #         """
        #         logits: (N, C) unnormalized scores
        #         targets: (N,)  class indices (LongTensor)
        #         """
        #         # log \sum_j exp(z_j)   (numerically stable)
        #         logsumexp = torch.logsumexp(logits, dim=1)                    # (N,)
        #         # gather z_{y} for each sample
        #         z_y = logits.gather(1, targets.view(-1, 1)).squeeze(1)        # (N,)
        #         loss_vec = logsumexp - z_y                                    # (N,)
        #         return loss_vec.mean()                                        # mean reduction
        if self.loss_type in ["CE", "softCE"]:
            def criterion(logits, targets):
                return F.cross_entropy(logits, targets.view(-1).long(), label_smoothing=self.label_smoothing)

        elif self.loss_type == 'LA':
            # Logit-Adjusted CE:  z'_c = z_c + tau * log(pi_c)
            # Requires self.class_priors (length C), optional self.tau (default 1.0)
            tau = float(getattr(self, "tau", 1.0))
            priors = torch.as_tensor(self.class_priors, dtype=torch.float32)
            eps = 1e-12                       # avoid log(0)
            log_priors = torch.log(priors.clamp_min(eps))

            def criterion(logits, targets):
                """
                logits: (N, C), targets: (N,)
                """
                # align device & dtype
                lpi = log_priors.to(device=logits.device, dtype=logits.dtype)
                adjusted_logits = logits + tau * lpi                          # (N, C)
                return F.cross_entropy(adjusted_logits, targets.view(-1).long())              # CE on adjusted logits

        else:
            raise ValueError(f" Unknown loss_type: {self.loss_type}")


        # build DataLoader
        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)

                opt.zero_grad()
                out = self.model_(xb)  #  expected logits or probabilities
                loss = criterion(out, yb.view(-1))
                loss.backward()
                opt.step()
                total_loss += loss.item()
                
            # decay once per epoch
            dual_scheduler.step()

            if self.verbose and epoch % self.options['eval_round'] == 0:
                print(f"[Train] Task: {self.task_name}, [Epoch {epoch+1}/{self.epochs}] loss={total_loss:.4f}")
                self.evaluate(X=X,y=y)

        self.model_.eval()

        pred_acc = self.evaluate(X=X,y=y)

        # with torch.no_grad():
        #     # normalize y shape & type
        #     y_true = np.asarray(y)
        #     if y_true.ndim == 2 and y_true.shape[1] == 1:
        #         y_true = y_true.ravel()
        #     # handle one-hot or prob labels
        #     if y_true.ndim == 2 and y_true.shape[1] > 1:
        #         y_true = np.argmax(y_true, axis=1)

        #     # Use existing predict()
        #     y_pred = self.predict(X)

        #     y_true_enc = np.vectorize(self.class_to_index_.get)(y_true)
        #     y_pred_enc = np.vectorize(self.class_to_index_.get)(y_pred)

        #     acc = float(np.mean(y_true_enc == y_pred_enc))
        #     self.train_acc_ = acc  # store for later access
        #     if self.verbose:
        #         print(f"[Eval] Task: {self.task_name}, train accuracy = {acc:.4f}")

        # save state_dict
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.model_.state_dict(), model_path)
            if self.verbose:
                print(f"[Info] Saved {self.task_name} model to: {model_path}")

        return self  #  sklearn convention

    @torch.no_grad()
    def evaluate(self, X, y, batch_size: int = None, use_fp16: bool = False):
        """
        Evaluate and return accuracy. No full-tensor GPU transfer.
        """
        assert self.model_ is not None, "Call fit() or load a pre-trained model before evaluate()"

        y_true = np.asarray(y)
        if y_true.ndim == 2 and y_true.shape[1] == 1:
            y_true = y_true.ravel()
        if y_true.ndim == 2 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)

        # get predictions (uses batch-wise GPU inference)
        P = self.predict_proba(X, batch_size=batch_size, use_fp16=use_fp16)  # CPU torch.Tensor (N,C)
        idx = torch.argmax(P, dim=1).cpu().numpy()
        y_pred = self.classes_[idx]

        # encode labels to indices and compute acc
        y_true_enc = np.array([self.class_to_index_.get(label, -1) for label in y_true])
        y_pred_enc = np.array([self.class_to_index_.get(label, -1) for label in y_pred])

        acc = float(np.mean(y_true_enc == y_pred_enc))

        if self.verbose:
            print(f"[Eval] Task: {self.task_name}, accuracy = {acc:.4f}")

        return acc


    # @torch.no_grad()
    # def predict_proba(self, X):
    #     """
    #     Return class probabilities per sample: (n_samples, n_classes).
    #     """
    #     assert self.model_ is not None, "call fit() before predict_proba"
    #     device = self._pick_device()
    #     n_classes = len(self.classes_)

    #     X_t = self._to_tensor(X, dtype=torch.float32).to(device)

    #     # batch inference to avoid OOM
    #     self.model_.eval()
    #     probs = []
    #     bs = self.batch_size if self.batch_size is not None else 1024
    #     for i in range(0, X_t.shape[0], bs):
    #         xb = X_t[i:i+bs]
    #         out = self.model_(xb)
    #         prob = self._to_proba(out, n_classes)
    #         probs.append(prob.detach().cpu())

    #     P = torch.cat(probs, dim=0)  # (N, C)
    #     return P

    @torch.no_grad()
    def predict_proba(self, X, batch_size: int = None, use_fp16: bool = False):
        """
        Return class probabilities per sample: torch.Tensor (N, C) on CPU.
        - X stays on CPU; only mini-batches are moved to GPU to avoid OOM.
        """
        assert self.model_ is not None, "call fit() before predict_proba"
        device = self._pick_device()
        n_classes = len(self.classes_)

        # keep X on CPU
        X_cpu = self._to_tensor(X, dtype=torch.float32)
        if X_cpu.ndim == 1:
            X_cpu = X_cpu.view(1, -1)

        self.model_.eval()

        bs = int(batch_size) if batch_size is not None else (int(self.batch_size) if self.batch_size is not None else 1024)

        probs = []
        for i in range(0, X_cpu.shape[0], bs):
            xb = X_cpu[i:i + bs].to(device, non_blocking=True)

            if use_fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model_(xb)
            else:
                out = self.model_(xb)

            prob = self._to_proba(out, n_classes)
            probs.append(prob.detach().cpu())

            # free intermediates early
            del xb, out, prob

        return torch.cat(probs, dim=0)  # CPU tensor

    @torch.no_grad()
    def predict(self, X, batch_size: int = None, use_fp16: bool = False):
        """
        Return predicted labels (mapped back to original classes_).
        """
        P = self.predict_proba(X, batch_size=batch_size, use_fp16=use_fp16)  # CPU torch.Tensor
        idx = torch.argmax(P, dim=1).cpu().numpy()
        return self.classes_[idx]


# # ====== Usage example ======
# class MLP(nn.Module):
#     """A simple 2-layer MLP"""
#     def __init__(self, input_dim, num_classes, hidden=64):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, num_classes)  #  output logits
#         )

#     def forward(self, x):
#         return self.net(x)


# if __name__ == "__main__":
#     # Fake data
#     N, D, C = 200, 10, 3
#     X = np.random.randn(N, D).astype(np.float32)
#     y = np.random.randint(0, C, size=N)

#     clf = TorchSklearnClassifier(
#         model_fn=MLP,                       # pass nn.Module constructor
#         model_kwargs={"input_dim": D, "num_classes": C, "hidden": 64},
#         lr=1e-3,
#         epochs=20,
#         batch_size=32,
#         output_activation="softmax",        # treat as multiclass, use softmax
#         verbose=1
#     )

#     clf.fit(X, y)
#     proba = clf.predict_proba(X[:5])
#     pred  = clf.predict(X[:5])
#     print("proba:\n", proba)
#     print("pred:\n", pred)

class Logistic(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(Logistic, self).__init__()
        print(f"input_shape:{input_shape}, output_dim:{output_dim}")
        self.layer = nn.Linear(input_shape, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.output_dim = output_dim

    def forward(self, x):
        # print(x.shape)
        if len(x.shape) == 1:
            x = x.view([1, x.shape[0]])
        # print(np.prod(x.shape[1:]))
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.layer(x)
        if self.output_dim == 1:
            x = self.sigmoid(x)
        else:
            x = x
        return x
    
class OneHiddenLayerMLP(nn.Module):
    def __init__(self, input_shape, output_dim, hidden_dim=200):
        super(OneHiddenLayerMLP, self).__init__()
        self.layer_input = nn.Linear(input_shape, hidden_dim)
        self.layer_hidden = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.output_dim = output_dim

    def forward(self, x):
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.layer_input(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer_hidden(x)
        if self.output_dim == 1:
            x = self.sigmoid(x)
        else:
            x = x
        return x

class TwoHiddenLayerMLP(nn.Module):
    def __init__(self, input_shape, output_dim, hidden_dim=100):
        super(TwoHiddenLayerMLP, self).__init__()
        self.layer_input = nn.Linear(input_shape, int(hidden_dim*2))
        self.layer_hidden_0 = nn.Linear(int(hidden_dim*2), hidden_dim)
        self.layer_hidden_1 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.layer_hidden_2 = nn.Linear(int(hidden_dim/2), output_dim)
        # self.drop1 = nn.Dropout(p=0.3)
        self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.output_dim = output_dim

    def forward(self, x):
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.layer_input(x)
        x = F.relu(x)
        x = self.layer_hidden_0(x)
        x = F.relu(x)
        x = self.layer_hidden_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer_hidden_2(x)
        if self.output_dim == 1:
            x = self.sigmoid(x)
        else:
            x = x
        return x

# class TwoHiddenLayerMLP(nn.Module):
#     """
#     2-hidden-layer MLP for classification on frozen embeddings.
#     Default: 768 -> 256 -> 128 -> 28

#     Args:
#         in_dim: embedding dim (e.g., 768 for BERT)
#         n_classes: number of classes (e.g., 28 for Bias in Bios)
#         hidden1: first hidden layer width (default 256)
#         hidden2: second hidden layer width (default 128)
#         dropout: dropout probability (default 0.2)
#         use_layernorm: whether to use LayerNorm (default True)
#     """
#     def __init__(
#         self,
#         input_shape: int = 768,
#         output_dim: int = 28,
#         hidden1: int = 256,
#         hidden2: int = 128,
#         dropout: float = 0.4,
#         use_layernorm: bool = True,
#     ):
#         super().__init__()
#         self.use_layernorm = use_layernorm

#         self.fc1 = nn.Linear(input_shape, hidden1)
#         self.ln1 = nn.LayerNorm(hidden1) if use_layernorm else nn.Identity()
#         self.drop1 = nn.Dropout(dropout)

#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.ln2 = nn.LayerNorm(hidden2) if use_layernorm else nn.Identity()
#         self.drop2 = nn.Dropout(dropout)

#         self.out = nn.Linear(hidden2, output_dim)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         # Xavier init for stability
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.zeros_(self.fc1.bias)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.zeros_(self.fc2.bias)
#         nn.init.xavier_uniform_(self.out.weight)
#         nn.init.zeros_(self.out.bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, in_dim] float tensor (your embedding vectors)
#         return: logits [B, n_classes]
#         """
#         x = self.fc1(x)
#         x = self.ln1(x)
#         x = F.gelu(x)
#         x = self.drop1(x)

#         x = self.fc2(x)
#         x = self.ln2(x)
#         x = F.gelu(x)
#         x = self.drop2(x)

#         logits = self.out(x)
#         return logits
    

class Residual(nn.Module):  #@save
    def __init__(self, input_shape, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_shape, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    
def resnet_block(input_shape, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_shape, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class ResNet10(nn.Module):  #@save
    def __init__(self, input_shape, output_dim):
        super().__init__()
        
        self.b1 = nn.Sequential(nn.Conv2d(input_shape, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.b2 = nn.Sequential(*resnet_block(64, 64, 1, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 1))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 1))

        self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),
                    nn.Dropout(p=0.3),
                    nn.Linear(256, output_dim))
        self.sigmoid = nn.Sigmoid()
        self.output_dim = output_dim

    def forward(self, x):
        x =  self.net(x)
        return x

# Neural Network for F-divergence
class TNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=5, depth=3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

class DummyEstimator(BaseEstimator, ClassifierMixin):
    """
    - This "classifier" treats each row of X as a per-class probability distribution.
    """
    def __init__(self, n_classes=2):
        self.n_classes = int(n_classes)
        self.classes_ = np.arange(self.n_classes)  # preset class labels
        self.is_fitted_ = True  # mark fitted for cv='prefit'

    def fit(self, X, y=None):
        # If y is provided, update classes_ from it (safer)
        if y is not None:
            self.classes_ = np.unique(y)
            self.n_classes = len(self.classes_)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        # convert to 2D array
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # number of columns must equal n_classes
        if X.shape[1] != len(self.classes_):
            raise ValueError(
                f"Expected X with {len(self.classes_)} columns (n_classes), got {X.shape[1]}"
            )

        # Ensure valid probabilities: non-negative + row-normalization
        X = np.clip(X, 0.0, None)
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return X / row_sums

    def predict(self, X):
        # argmax over probabilities
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


def ModelMapping(options):
    dataset = str(options['data']).lower()
    model = str(options['model']).lower()
    criterion = str(options['criterion']).lower()
    fairness_type = options['fairness_type']
    if dataset == 'celeba':
        if model in ['logistic', '2nn', '1nn']:
            if criterion =='binary':
                return {'input_shape': np.prod(options['num_shape']), 'num_class': 1}
            elif criterion =='multiclass' and fairness_type == 'groupwise':
                return {'input_shape': np.prod(options['num_shape']), 'num_class': 2}
            elif criterion =='multiclass' and fairness_type == 'subgroup':
                return {'input_shape': np.prod(options['num_shape']), 'num_class': 4}
        if model in ['resnet']:
            if criterion =='binary':
                return {'input_shape': 3, 'num_class': 1}
            elif criterion =='multiclass' and fairness_type == 'groupwise':
                return {'input_shape': 3, 'num_class': 2}
            elif criterion =='multiclass' and fairness_type == 'subgroup':
                return {'input_shape': 3, 'num_class': 4}
            # return {'input_shape': (3,128,128), 'num_class': 1} input_shape=3, num_channels=1
    elif dataset == 'adult':
        if model in ['logistic', '2nn', '1nn']:
            if criterion =='binary':
                return {'input_shape': np.prod(options['num_shape']), 'num_class': 1}
            elif criterion =='multiclass' and fairness_type == 'groupwise':
                return {'input_shape': np.prod(options['num_shape']), 'num_class': 2}
            elif criterion =='multiclass' and fairness_type == 'subgroup':
                raise ValueError(f'Partition of {dataset} doesnot support multi-class classification!')
        else:
            raise ValueError('{} doesnot support model {}!'.format(dataset, model))
    elif dataset == 'acs':
        if model in ['logistic', '2nn', '1nn']:
            if criterion =='binary':
                return {'input_shape': np.prod(options['num_shape']), 'num_class': 1}
            elif criterion =='multiclass' and fairness_type == 'groupwise':
                return {'input_shape': np.prod(options['num_shape']), 'num_class': 2}
        else:
            raise ValueError('{} doesnot support model {}!'.format(dataset, model))
    elif dataset == 'compas':
        if model in ['logistic', '2nn', '1nn']:
            if criterion =='binary':
                return {'input_shape': np.prod(options['num_shape']), 'num_class': 1}
            elif criterion =='multiclass'and fairness_type == 'groupwise':
                return {'input_shape': np.prod(options['num_shape']), 'num_class': 2}
            elif criterion =='multiclass' and fairness_type == 'subgroup':
                raise ValueError(f'Partition of {dataset} doesnot support multi-class classification!')
        else:
            raise ValueError('{} doesnot support model {}!'.format(dataset, model))
    elif dataset == 'compas_1':
        if model in ['logistic', '2nn', '1nn']:
            if criterion =='binary':
                return {'input_shape': np.prod(options['num_shape']), 'num_class': 1}
            elif criterion =='multiclass':
                return {'input_shape': np.prod(options['num_shape']), 'num_class': 2}
        else:
            raise ValueError('{} doesnot support model {}!'.format(dataset, model))
    elif dataset == 'enem':
        if model in ['logistic', '2nn', '1nn']:
            if criterion =='binary':
                return {'input_shape': np.prod(options['num_shape']), 'num_class': 1}
            elif criterion =='multiclass' and fairness_type == 'groupwise':
                return {'input_shape': np.prod(options['num_shape']), 'num_class': 2}
            elif criterion =='multiclass' and fairness_type == 'subgroup':
                return {'input_shape': np.prod(options['num_shape']), 'num_class': 5}
        else:
            raise ValueError('{} doesnot support model {}!'.format(dataset, model))
    elif dataset == 'synth':
        if model in ['logistic']:
            return {'input_shape': np.prod(options['num_shape']), 'num_class': 1}
        else:
            raise ValueError('{} doesnot support model {}!'.format(dataset, model))
    elif dataset == 'bank':
        if model in ['logistic', '2nn', '1nn']:
            return {'input_shape':np.prod(options['num_shape']), 'num_class': 1}
    else:
        raise ValueError('Not support dataset {}!'.format(dataset))

# def choose_model(options):
#     model_name = str(options['model']).lower()
#     modelconfig = ModelMapping(options)
#     options.update(modelconfig)
#     modelconfig['output_dim'] = modelconfig.pop('num_class')
#     if model_name == 'logistic':
#         return Logistic(**modelconfig)
#     elif model_name == '2nn':
#         return TwoHiddenLayerMLP(**modelconfig)
#     elif model_name == '1nn':
#         return OneHiddenLayerMLP(**modelconfig)
#     elif model_name == 'resnet':
#         return ResNet8(**modelconfig)
#     else:
#         raise ValueError("Not support model: {}!".format(model_name))

def choose_model(options):
    if options['model'] == 'logistic':
        return Logistic
    elif options['model'] == '1nn':
        return OneHiddenLayerMLP
    elif options['model'] == '2nn':
        return TwoHiddenLayerMLP
    elif options['model'] == 'resnet':
        return ResNet10
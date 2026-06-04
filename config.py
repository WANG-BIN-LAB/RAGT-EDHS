class Config:
    def __init__(self):
        # ---------- Dataset ----------
        self.dataset = type('', (), {})()
        self.dataset.name = 'abide'
        self.dataset.batch_size = 16
        self.dataset.test_batch_size = 16
        self.dataset.train_set = 0.9
        self.dataset.path = r"S:\small_paper\models\a2026.4.26\dataset\abide_age+sex.npy"
        self.dataset.stratified = True
        self.dataset.drop_last = True

        self.datasz = type('', (), {})()
        self.datasz.percentage = 1.0

        self.preprocess = type('', (), {})()
        self.preprocess.name = 'default'
        self.preprocess.continus = True

        # ---------- Model ----------
        self.model = type('', (), {})()
        self.model.name = 'BrainNetworkTransformer'
        self.model.pos_encoding = True
        self.model.pos_embed_dim = 200
        self.model.sizes = [200, 100, 50]
        self.model.orthogonal = True
        self.model.freeze_center = False
        self.model.project_assignment = True

        # ---------- Training ----------
        self.training = type('', (), {})()
        self.training.epochs = 100
        self.total_steps = 10000
        self.save_learnable_graph = False
        self.n_folds = 10  # 10-fold cross-validation
        self.seed = 42     # Global random seed for reproducibility

        # ===================== Optimizer =====================
        self.optimizer = type('', (), {})()
        self.optimizer.name = 'Adam'
        self.optimizer.lr = 5.0e-5
        self.optimizer.weight_decay = 1.0e-4

        # ========== Loss Weight ==========
        self.divergence_weight = 1.5

        # ===================== Grid Search Parameters =====================
        self.model.num_layers = 1
        self.model.nhead = 4
        self.model.global_topk_ratio = 0.1
        self.model.local_neighbor_num = 17

        # ---------- Others ----------
        self.unique_id = None
        self.project = 'brain_network'
        self.wandb_entity = 'user'
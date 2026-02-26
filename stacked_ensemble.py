# stacked_ensemble.py
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from foundation_adapter import FoundationAdapter

class T1DStackedFoundationModel:
    """
    Main stacked ensemble for MAI-T1D.
    Supports CPU, local GPU, or H200 automatically.
    """
    def __init__(
        self,
        wgs_path=None,
        rnaseq_path=None,
        clinical_path=None,
        device: str = "auto",      # <-- NEW: pass this from anywhere
        low_memory: bool = False
    ):
        self.device = device
        self.low_memory = low_memory

        self.adapters = {}
        if wgs_path:
            self.adapters['wgs'] = FoundationAdapter(wgs_path, 'wgs', device=device, low_memory=low_memory)
        if rnaseq_path:
            self.adapters['rnaseq'] = FoundationAdapter(rnaseq_path, 'rnaseq', device=device, low_memory=low_memory)
        if clinical_path:
            self.adapters['clinical'] = FoundationAdapter(clinical_path, 'clinical', device=device, low_memory=low_memory)

        self.meta_learner = None
        self.calibrators = {}

    def get_meta_features(self, data_dict):
        meta_feats = []
        for name, adapter in self.adapters.items():
            if name in data_dict and data_dict[name] is not None:
                probs, logits, emb = adapter.predict_and_extract(data_dict[name])
                feats = [probs.flatten()]
                if logits is not None:
                    feats.append(logits.flatten())
                feats.append(emb.mean(axis=0))
                meta_feats.extend(feats)
        return np.concatenate(meta_feats, axis=0) if meta_feats else np.array([])

    def fit_meta_learner(self, X_meta, y, n_folds=5):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        self.meta_learner = LogisticRegression(max_iter=1000)
        self.meta_learner = CalibratedClassifierCV(
            estimator=self.meta_learner,
            method='isotonic',
            cv=skf
        )
        self.meta_learner.fit(X_meta, y)
        joblib.dump(self.meta_learner, 'models/meta_learner.joblib')

    def predict(self, data_dict):
        meta_feat = self.get_meta_features(data_dict)
        meta_feat = meta_feat.reshape(1, -1)
        return self.meta_learner.predict_proba(meta_feat)[0][1]  # T1D positive prob

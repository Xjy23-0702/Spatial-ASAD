import numpy as np
import config as cfg
class CSP:
    def __init__(self, n_components=4):
        self.n_components = cfg.csp_n_components
        self.filters_ = None

    def fit(self, X, y):
        """
        X: (windows_num, windows_length, channel)
        y: (windows_num,)
        """
        class_0 = X[y == 0]# shape: (N0, windows_length, channel)
        class_1 = X[y == 1]# shape: (N1, windows_length, channel)

        cov_0 = self._covariance(class_0)
        cov_1 = self._covariance(class_1)

        cov_total = cov_0 + cov_1

        eigvals, eigvecs = np.linalg.eigh(cov_total)#对总协方差矩阵做特征分解
        eigvals = np.maximum(eigvals, 1e-8)
        P = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T#白化矩阵

        S0 = P @ cov_0 @ P.T
        eigvals_s0, eigvecs_s0 = np.linalg.eigh(S0)

        idx = np.argsort(eigvals_s0)[::-1]
        eigvecs_s0 = eigvecs_s0[:, idx]

        W = eigvecs_s0.T @ P#得到 CSP 滤波器矩阵,(C,C)

        # 选前后 components（最区分的）
        self.filters_ = np.vstack([
            W[:self.n_components],
            W[-self.n_components:]
        ])
        #filters_.shape = (2*n_components, channel)

        return self
    def transform(self, X, log=True):
        """
        X: (windows_num, windows_length, channel)
        return: (windows_num, windows_length, n_components*2)
        """
        X_csp = []
        for trial in X:
            projected = trial @ self.filters_.T
            X_csp.append(projected)
        return np.array(X_csp)


    def _covariance(self, X):
        covs = []
        for trial in X:
            cov = trial.T @ trial
            cov = cov / trial.shape[0]
            covs.append(cov)
        mean_cov = np.mean(covs, axis=0)
        return mean_cov / np.trace(mean_cov)
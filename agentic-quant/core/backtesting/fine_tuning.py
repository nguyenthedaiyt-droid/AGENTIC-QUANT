# =============================================================================
# AGENTIC-QUANT — Fine-Tuning Pipeline (Phase 8)
# Quyet dinh khi nao fine-tune, thuc hien fine-tune, va rollback neu khong improve
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    pass


# =============================================================================
# Constants
# =============================================================================
_DEFAULT_MIN_SAMPLES = 200
_DEFAULT_LR = 1e-5
_DEFAULT_EPOCHS = 5
_IC_IMPROVEMENT_FACTOR = 1.1  # IC_moi phai >= IC_cu * 1.1 de deploy


# =============================================================================
# FineTuningPipeline
# =============================================================================
class FineTuningPipeline:
    """Pipeline fine-tuning cho XGBoost model.

    Quyet dinh khi nao fine-tune, thuc hien, va rollback neu khong improve.

    Train/Val/Test split: temporal (KHONG random) — dam bao khong look-ahead bias.

    Args:
        min_samples: So luong samples toi thieu de fine-tune (default: 200)
        lr: Learning rate cho fine-tuning (default: 1e-5)
        epochs: So epochs cho fine-tuning (default: 5)
        ic_improvement_factor: He so IC phai dat duoc de deploy (default: 1.1)
    """

    def __init__(
        self,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
        lr: float = _DEFAULT_LR,
        epochs: int = _DEFAULT_EPOCHS,
        ic_improvement_factor: float = _IC_IMPROVEMENT_FACTOR,
    ) -> None:
        self._min_samples = min_samples
        self._lr = lr
        self._epochs = epochs
        self._ic_improvement_factor = ic_improvement_factor

        # Tracking
        self._n_fine_tunes: int = 0
        self._n_rollbacks: int = 0
        self._n_deploys: int = 0
        self._current_ic: float = 0.0
        self._last_new_ic: float = 0.0
        self._history: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def n_fine_tunes(self) -> int:
        """So lan fine-tune da thuc hien."""
        return self._n_fine_tunes

    @property
    def n_rollbacks(self) -> int:
        """So lan rollback da thuc hien."""
        return self._n_rollbacks

    @property
    def n_deploys(self) -> int:
        """So lan deploy thanh cong."""
        return self._n_deploys

    @property
    def current_ic(self) -> float:
        """IC hien tai cua model."""
        return self._current_ic

    # -------------------------------------------------------------------------
    # Should Fine-Tune?
    # -------------------------------------------------------------------------
    def should_fine_tune(
        self,
        model_degraded: bool,
        new_samples_count: int,
        min_samples: int | None = None,
    ) -> bool:
        """Quyet dinh co nen fine-tune hay khong.

        Dieu kien:
        1. Model bi degraded (FDS > threshold hoac IC giam)
        2. Du new samples >= min_samples

        Args:
            model_degraded: Model co bi degraded khong
            new_samples_count: So luong new samples da collect
            min_samples: So samples toi thieu (override default neu co)

        Returns:
            bool: True nen fine-tune, False khong
        """
        min_s = min_samples if min_samples is not None else self._min_samples

        if not model_degraded:
            logger.debug(
                "[FineTune] Khong fine-tune: model chua degraded"
            )
            return False

        if new_samples_count < min_s:
            logger.info(
                "[FineTune] Cho them data: co {n} samples, can >= {min_s}",
                n=new_samples_count,
                min_s=min_s,
            )
            return False

        logger.info(
            "[FineTune] Quyet dinh FINE-TUNE: degraded={degraded}, "
            "samples={n}/{min_s}",
            degraded=model_degraded,
            n=new_samples_count,
            min_s=min_s,
        )
        return True

    # -------------------------------------------------------------------------
    # Temporal Split (Train/Val/Test)
    # -------------------------------------------------------------------------
    def temporal_split(
        self,
        data: np.ndarray | list,
        labels: np.ndarray | list | None = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> dict[str, Any]:
        """Chia Train/Val/Test theo thoi gian (KHONG random).

        Data duoc chia theo thu tu thoi gian:
        - Train: 70% dau tien
        - Val: 15% tiep theo
        - Test: 15% cuoi cung

        Args:
            data: Data array (da duoc sort theo thoi gian)
            labels: Labels array (optional)
            train_ratio: Ty le train (default: 0.7)
            val_ratio: Ty le val (default: 0.15)

        Returns:
            Dict: {train_data, val_data, test_data, train_labels, val_labels, test_labels}

        Raises:
            ValueError: Neu train_ratio + val_ratio >= 1.0
        """
        if train_ratio + val_ratio >= 1.0:
            raise ValueError(
                f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) phai < 1.0"
            )

        test_ratio = 1.0 - train_ratio - val_ratio
        n = len(data)

        if n < 3:
            logger.warning(
                "Data qua it ({n}) de temporal split, tra ve toan bo lam train",
                n=n,
            )
            return {
                "train_data": data,
                "val_data": [],
                "test_data": [],
                "train_labels": labels[:] if labels is not None else [],
                "val_labels": [],
                "test_labels": [],
            }

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        split_result: dict[str, Any] = {
            "train_data": data[:train_end],
            "val_data": data[train_end:val_end],
            "test_data": data[val_end:],
        }

        if labels is not None:
            split_result["train_labels"] = labels[:train_end]
            split_result["val_labels"] = labels[train_end:val_end]
            split_result["test_labels"] = labels[val_end:]
        else:
            split_result["train_labels"] = []
            split_result["val_labels"] = []
            split_result["test_labels"] = []

        logger.info(
            "[TemporalSplit] Train={train} Val={val} Test={test}",
            train=len(split_result["train_data"]),
            val=len(split_result["val_data"]),
            test=len(split_result["test_data"]),
        )

        return split_result

    # -------------------------------------------------------------------------
    # Run Fine-Tune
    # -------------------------------------------------------------------------
    async def run_fine_tune(
        self,
        model: Any,
        new_data: np.ndarray | list,
        new_labels: np.ndarray | list | None = None,
        lr: float | None = None,
        epochs: int | None = None,
    ) -> dict[str, Any]:
        """Thuc hien fine-tuning tren model.

        Flow:
        1. Temporal split new_data thanh Train/Val/Test
        2. Fine-tune model voi learning rate thap
        3. Evaluate IC tren test set
        4. Neu IC_new > IC_current * 1.1 -> deploy
        5. Neu khong -> rollback

        Args:
            model: Model can fine-tune (XGBoost model object)
            new_data: Data moi cho fine-tuning
            new_labels: Labels moi (optional)
            lr: Learning rate (override)
            epochs: So epochs (override)

        Returns:
            Dict: {deployed, new_ic, old_ic, rollback, reason}
        """
        actual_lr = lr if lr is not None else self._lr
        actual_epochs = epochs if epochs is not None else self._epochs

        old_ic = self._current_ic

        if not new_data or len(new_data) < self._min_samples:
            logger.warning(
                "[FineTune] Khong du data de fine-tune: {n} samples < {min}",
                n=len(new_data) if new_data else 0,
                min=self._min_samples,
            )
            return {
                "deployed": False,
                "new_ic": old_ic,
                "old_ic": old_ic,
                "rollback": True,
                "reason": "Khong du data",
            }

        # --- Temporal split ---
        split = self.temporal_split(new_data, new_labels)
        train_data = split["train_data"]
        val_data = split["val_data"]
        test_data = split["test_data"]
        train_labels = split.get("train_labels", [])
        val_labels = split.get("val_labels", [])
        test_labels = split.get("test_labels", [])

        logger.info(
            "[FineTune] Bat dau fine-tune: lr={lr}, epochs={epochs}, "
            "train={train}, val={val}, test={test}",
            lr=actual_lr,
            epochs=actual_epochs,
            train=len(train_data),
            val=len(val_data),
            test=len(test_data),
        )

        # --- Fine-tune ---
        try:
            if hasattr(model, "fine_tune"):
                # Neu model co fine_tune method
                result = await model.fine_tune(
                    X_train=train_data,
                    y_train=train_labels or None,
                    X_val=val_data,
                    y_val=val_labels or None,
                    learning_rate=actual_lr,
                    num_boost_round=actual_epochs,
                )
            elif hasattr(model, "partial_fit"):
                # Neu model co partial_fit (sklearn-style)
                model.partial_fit(
                    np.array(train_data),
                    np.array(train_labels) if train_labels else None,
                )
                result = {"success": True}
            else:
                # Fallback: fit lai model
                model.fit(
                    np.array(train_data),
                    np.array(train_labels) if train_labels else None,
                )
                result = {"success": True}

            # --- Evaluate IC tren test set ---
            new_ic = self._evaluate_ic(
                model=model,
                test_data=test_data,
                test_labels=test_labels,
            )

        except Exception as exc:
            logger.error(
                "[FineTune] Loi trong qua trinh fine-tune: {exc}",
                exc=exc,
            )
            return {
                "deployed": False,
                "new_ic": old_ic,
                "old_ic": old_ic,
                "rollback": True,
                "reason": f"Error: {exc}",
            }

        # --- Decision: deploy or rollback ---
        deploy_threshold = old_ic * self._ic_improvement_factor if old_ic != 0 else 0.01
        should_deploy = new_ic > deploy_threshold

        if should_deploy:
            self._current_ic = new_ic
            self._last_new_ic = new_ic
            self._n_fine_tunes += 1
            self._n_deploys += 1

            logger.info(
                "[FineTune] DEPLOY: IC={new_ic:.4f} > threshold={thresh:.4f} "
                "(old_ic={old_ic:.4f} * {factor})",
                new_ic=new_ic,
                thresh=deploy_threshold,
                old_ic=old_ic,
                factor=self._ic_improvement_factor,
            )

            result = {
                "deployed": True,
                "new_ic": new_ic,
                "old_ic": old_ic,
                "rollback": False,
                "reason": f"IC improved: {old_ic:.4f} -> {new_ic:.4f}",
            }
        else:
            self._n_rollbacks += 1

            logger.warning(
                "[FineTune] ROLLBACK: IC={new_ic:.4f} <= threshold={thresh:.4f} "
                "(old_ic={old_ic:.4f} * {factor})",
                new_ic=new_ic,
                thresh=deploy_threshold,
                old_ic=old_ic,
                factor=self._ic_improvement_factor,
            )

            result = {
                "deployed": False,
                "new_ic": new_ic,
                "old_ic": old_ic,
                "rollback": True,
                "reason": (
                    f"IC khong du improve: {old_ic:.4f} -> {new_ic:.4f}, "
                    f"can >= {deploy_threshold:.4f}"
                ),
            }

        # Log history
        self._history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result,
        })

        return result

    # -------------------------------------------------------------------------
    # Rollback Check
    # -------------------------------------------------------------------------
    def rollback_if_not_improved(
        self,
        old_ic: float,
        new_ic: float,
    ) -> bool:
        """Kiem tra co can rollback hay khong.

        Rollback = True neu new_ic <= old_ic (khong improve).

        Args:
            old_ic: IC truoc khi fine-tune
            new_ic: IC sau khi fine-tune

        Returns:
            bool: True = can rollback, False = keep
        """
        needs_rollback = new_ic <= old_ic * (self._ic_improvement_factor - 0.1)

        if needs_rollback:
            self._n_rollbacks += 1
            logger.warning(
                "[FineTune] Rollback: IC khong improve "
                "(old={old:.4f}, new={new:.4f})",
                old=old_ic,
                new=new_ic,
            )
        else:
            logger.info(
                "[FineTune] Keep: IC improve (old={old:.4f}, new={new:.4f})",
                old=old_ic,
                new=new_ic,
            )

        return needs_rollback

    # -------------------------------------------------------------------------
    # Fine-Tune voi Data Loader
    # -------------------------------------------------------------------------
    async def fine_tune(
        self,
        model: Any,
        data_loader: callable | None = None,
        new_samples: np.ndarray | list | None = None,
        new_labels: np.ndarray | list | None = None,
        lr: float | None = None,
        epochs: int | None = None,
    ) -> dict[str, Any]:
        """Fine-tune model voi data_loader hoac new_samples truc tiep.

        Neu data_loader duoc cung cap, no duoc goi de lay (X, y) tu
        SQLite / Redis. Neu new_samples duoc cung cap, su dung truc tiep.

        Flow:
        1. Lay data tu data_loader (neu co) hoac su dung new_samples
        2. Temporal split thanh Train/Val/Test
        3. Fine-tune model
        4. Compute IC tren test set
        5. Neu IC_new > old_ic * 1.1 -> deploy
        6. Neu khong -> rollback

        Args:
            model: Model can fine-tune
            data_loader: Callable tra ve (X, y) tu SQLite/Redis (optional)
            new_samples: Data moi truc tiep (optional, dung neu khong co data_loader)
            new_labels: Labels moi (optional)
            lr: Learning rate (override)
            epochs: So epochs (override)

        Returns:
            Dict: {deployed, new_ic, old_ic, rollback, reason}

        Raises:
            ValueError: Neu ca data_loader va new_samples deu None
        """
        actual_lr = lr if lr is not None else self._lr
        actual_epochs = epochs if epochs is not None else self._epochs

        # --- Lay data tu data_loader hoac new_samples ---
        if data_loader is not None:
            try:
                logger.info("[FineTune] Dang load data tu data_loader...")
                loaded = data_loader()
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    new_data, new_labels_local = loaded
                elif isinstance(loaded, dict):
                    new_data = loaded.get("X", loaded.get("data", []))
                    new_labels_local = loaded.get("y", loaded.get("labels", []))
                else:
                    raise ValueError(f"data_loader tra ve kieu khong ho tro: {type(loaded)}")

                logger.info(
                    "[FineTune] Data loader tra ve {n} samples",
                    n=len(new_data) if hasattr(new_data, "__len__") else "?",
                )
            except Exception as exc:
                logger.error("[FineTune] Loi data_loader: {exc}", exc=exc)
                return {
                    "deployed": False,
                    "new_ic": self._current_ic,
                    "old_ic": self._current_ic,
                    "rollback": True,
                    "reason": f"Data loader error: {exc}",
                }
        elif new_samples is not None:
            new_data = new_samples
            new_labels_local = new_labels
        else:
            return {
                "deployed": False,
                "new_ic": self._current_ic,
                "old_ic": self._current_ic,
                "rollback": True,
                "reason": "Ca data_loader va new_samples deu None",
            }

        # Su dung new_data va new_labels_local cho phan con lai
        return await self.run_fine_tune(
            model=model,
            new_data=new_data,
            new_labels=new_labels_local,
            lr=actual_lr,
            epochs=actual_epochs,
        )

    # -------------------------------------------------------------------------
    # Compute IC Before Deploy
    # -------------------------------------------------------------------------
    def compute_ic(self, new_preds: list[float] | np.ndarray) -> float:
        """Tinh IC cho predictions moi de kiem tra truoc khi deploy.

        So sanh IC_new > old_ic * 1.1.

        Args:
            new_preds: Predictions tu model fine-tuned

        Returns:
            float: IC value (0.0 neu khong tinh duoc)
        """
        if not new_preds or len(new_preds) < 3:
            logger.warning("[FineTune] Khong du predictions de tinh IC")
            return 0.0

        # So sanh IC_new voi old_ic * 1.1
        new_preds_arr = np.asarray(new_preds, dtype=np.float64).flatten()

        # Neu khong co ground truth, dung gi dinh
        # (thuc te can y_true tu pipeline)
        logger.info(
            "[FineTune] Compute IC: new_preds={n}, old_ic={old_ic:.4f}, "
            "threshold={thresh:.4f}",
            n=len(new_preds_arr),
            old_ic=self._current_ic,
            thresh=self._current_ic * self._ic_improvement_factor if self._current_ic != 0 else 0.01,
        )

        return float(np.mean(new_preds_arr))  # Placeholder — can y_true tu caller

    # -------------------------------------------------------------------------
    # Internal: Evaluate IC
    # -------------------------------------------------------------------------
    def _evaluate_ic(
        self,
        model: Any,
        test_data: list | np.ndarray,
        test_labels: list | np.ndarray | None,
    ) -> float:
        """Evaluate model IC tren test data.

        Args:
            model: Model da fine-tune
            test_data: Test features
            test_labels: Test labels

        Returns:
            float: IC value
        """
        if not test_data or len(test_data) < 3:
            logger.warning(
                "[Evaluate] Khong du test data de tinh IC, return 0.0"
            )
            return 0.0

        try:
            # Predict
            if hasattr(model, "predict_proba"):
                y_pred = model.predict_proba(np.array(test_data))
                # Lay probability cua class 1 (bullish / positive)
                if y_pred.ndim == 2 and y_pred.shape[1] >= 2:
                    y_pred = y_pred[:, 1]
                else:
                    y_pred = y_pred.flatten()
            elif hasattr(model, "predict"):
                y_pred = model.predict(np.array(test_data))
            else:
                logger.warning("[Evaluate] Model khong co predict method")
                return 0.0

            if test_labels is None:
                logger.warning("[Evaluate] Khong co test labels")
                return 0.0

            # Tinh Spearman correlation
            from scipy.stats import spearmanr

            y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
            y_true = np.asarray(test_labels, dtype=np.float64).flatten()

            if len(y_pred) != len(y_true):
                logger.warning(
                    "[Evaluate] Length mismatch: {pred} vs {true}",
                    pred=len(y_pred),
                    true=len(y_true),
                )
                return 0.0

            if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
                return 0.0

            result_obj = spearmanr(y_pred, y_true)
            if isinstance(result_obj, (list, tuple)):
                ic_val = float(result_obj[0])
            else:
                ic_val = float(result_obj.statistic)

            return ic_val if not np.isnan(ic_val) else 0.0

        except Exception as exc:
            logger.warning(
                "[Evaluate] Loi tinh IC: {exc}",
                exc=exc,
            )
            return 0.0

    # -------------------------------------------------------------------------
    # Stats / History
    # -------------------------------------------------------------------------
    @property
    def history(self) -> list[dict[str, Any]]:
        """Lich su cac lan fine-tune.

        Returns:
            List cac dict fine-tune history
        """
        return list(self._history)

    def get_success_rate(self) -> float:
        """Ty le deploy thanh cong.

        Returns:
            float: Success rate (0.0 -> 1.0)
        """
        if self._n_fine_tunes == 0:
            return 0.0
        return self._n_deploys / self._n_fine_tunes

    def get_stats(self) -> dict[str, Any]:
        """Tra ve thong ke.

        Returns:
            Dict statistics
        """
        return {
            "n_fine_tunes": self._n_fine_tunes,
            "n_rollbacks": self._n_rollbacks,
            "n_deploys": self._n_deploys,
            "success_rate": self.get_success_rate(),
            "current_ic": self._current_ic,
            "min_samples": self._min_samples,
            "lr": self._lr,
            "epochs": self._epochs,
            "ic_improvement_factor": self._ic_improvement_factor,
        }

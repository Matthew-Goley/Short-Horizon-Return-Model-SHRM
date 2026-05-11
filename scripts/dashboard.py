import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # explicit GUI backend; reliable on Windows/Linux/Mac
import matplotlib.pyplot as plt


class Dashboard:
    """
    Live training dashboard. Functional, not pretty.

    Six panels in a 2x3 grid:
      (0,0) Predicted Gaussian PDF — updates live during training, overlays the
            unconditional baseline so you can see the model's prediction shifting/squeezing
            relative to what a constant-predictor would do.
      (0,1) |μ| per batch — running history of absolute predicted mean
      (0,2) σ per batch — running history of predicted std, with baseline reference line
      (1,0) NLL per epoch — train and val curves with the unconditional NLL baseline
      (1,1) Dir Acc per epoch — with 0.5 (random) reference line, zoomed
      (1,2) pred+ rate per epoch — fraction of μ predictions that are positive

    Update granularity:
      - update_batch() is called per training batch. Throttled by `batch_redraw_interval`
        to avoid trying to redraw GUI on every single batch (which would be ~25 Hz with
        small batches and totally lock the event loop).
      - update_epoch() is called once after each epoch's validation pass.
    """

    def __init__(self, return_std, batch_redraw_interval=5):
        self.return_std = float(return_std)
        # NLL of an N(0, return_std) predictor — our "did the model learn anything" floor
        self.baseline_nll = 0.5 * (1 + np.log(self.return_std ** 2))
        self.batch_redraw_interval = batch_redraw_interval
        self._batch_counter = 0

        # history buffers — keep simple lists so appending is O(1) and matplotlib can plot directly
        self.batch_mu = []
        self.batch_sigma = []
        self.epoch_train_nll = []
        self.epoch_val_nll = []
        self.epoch_dir_acc = []
        self.epoch_pred_pos_rate = []

        # build figure
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        self.fig.suptitle("Training Dashboard", fontsize=12, fontweight="bold")

        # set titles up front so even an empty dashboard looks sensible
        self.axes[0, 0].set_title("Predicted Gaussian (live)")
        self.axes[0, 1].set_title("|μ| per batch")
        self.axes[0, 2].set_title("σ per batch")
        self.axes[1, 0].set_title("NLL per epoch")
        self.axes[1, 1].set_title("Dir Acc per epoch")
        self.axes[1, 2].set_title("pred+ rate per epoch")

        plt.show(block=False)
        plt.pause(0.1)  # let the GUI thread spin up before we start hammering it

    def _is_alive(self):
        # if the user closes the window mid-training, fignum_exists returns False
        # and subsequent update calls become no-ops instead of crashing
        return plt.fignum_exists(self.fig.number)

    def update_batch(self, mu_abs_mean, sigma_mean):
        """Called per training batch. mu_abs_mean and sigma_mean are scalar floats."""
        if not self._is_alive():
            return
        self.batch_mu.append(float(mu_abs_mean))
        self.batch_sigma.append(float(sigma_mean))
        self._batch_counter += 1

        # throttle redraws — we still record every batch's data, just don't redraw every time
        if self._batch_counter % self.batch_redraw_interval != 0:
            return

        self._draw_pdf(mu_abs_mean, sigma_mean)
        self._draw_batch_panels()
        self._refresh()

    def update_epoch(self, train_nll, val_nll, dir_acc, pred_pos_rate):
        """Called once after each epoch's validation pass."""
        if not self._is_alive():
            return
        self.epoch_train_nll.append(float(train_nll))
        self.epoch_val_nll.append(float(val_nll))
        self.epoch_dir_acc.append(float(dir_acc))
        self.epoch_pred_pos_rate.append(float(pred_pos_rate))

        self._draw_epoch_panels()
        self._refresh()

    def _draw_pdf(self, mu, sigma):
        ax = self.axes[0, 0]
        ax.cla()
        ax.set_title("Predicted Gaussian (live)")

        # x-axis range: use the larger of (model σ, baseline std) so the curve is always
        # visible regardless of whether the model is in early-epoch wide-σ phase or late
        # narrow-σ phase. 4σ covers ~99.99% of the distribution either way.
        sigma_safe = max(sigma, 1e-8)
        x_range = max(4 * sigma_safe, 4 * self.return_std)
        x = np.linspace(-x_range, x_range, 300)

        # model's predicted Gaussian
        y_model = np.exp(-0.5 * ((x - mu) / sigma_safe) ** 2) / (sigma_safe * np.sqrt(2 * np.pi))
        ax.plot(x, y_model, color="blue", linewidth=1.5, label="model")

        # unconditional baseline N(0, return_std) — for visual comparison
        y_base = np.exp(-0.5 * (x / self.return_std) ** 2) / (self.return_std * np.sqrt(2 * np.pi))
        ax.plot(x, y_base, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="unconditional")

        ax.axvline(0, color="black", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.axvline(mu, color="red", linestyle="-", linewidth=1.0, alpha=0.7)

        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=8)
        # corner readout for current values
        ax.text(
            0.02, 0.98,
            f"μ = {mu:.5f}\nσ = {sigma:.5f}",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            family="monospace",
        )

    def _draw_batch_panels(self):
        # |μ| per batch
        ax = self.axes[0, 1]
        ax.cla()
        ax.set_title("|μ| per batch")
        ax.plot(self.batch_mu, color="blue", alpha=0.7, linewidth=0.8)
        ax.set_xlabel("Batch")
        ax.set_ylabel("|μ|")

        # σ per batch with baseline ref
        ax = self.axes[0, 2]
        ax.cla()
        ax.set_title("σ per batch")
        ax.plot(self.batch_sigma, color="purple", alpha=0.7, linewidth=0.8)
        ax.axhline(
            self.return_std, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
            label=f"unconditional = {self.return_std:.4f}",
        )
        ax.set_xlabel("Batch")
        ax.set_ylabel("σ")
        ax.legend(loc="upper right", fontsize=8)

    def _draw_epoch_panels(self):
        epochs = list(range(1, len(self.epoch_train_nll) + 1))

        # NLL train + val + baseline
        ax = self.axes[1, 0]
        ax.cla()
        ax.set_title("NLL per epoch")
        ax.plot(epochs, self.epoch_train_nll, color="blue", marker="o", markersize=3, label="train")
        ax.plot(epochs, self.epoch_val_nll, color="orange", marker="o", markersize=3, label="val")
        ax.axhline(
            self.baseline_nll, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
            label=f"baseline = {self.baseline_nll:.3f}",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NLL")
        ax.legend(loc="upper right", fontsize=8)

        # Dir Acc, zoomed to the only range that matters
        ax = self.axes[1, 1]
        ax.cla()
        ax.set_title("Dir Acc per epoch")
        ax.plot(epochs, self.epoch_dir_acc, color="green", marker="o", markersize=3)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="random = 0.5")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dir Acc")
        ax.set_ylim([0.4, 0.6])  # anything outside this range is either bug or magic
        ax.legend(loc="upper right", fontsize=8)

        # pred+ rate — looking for stability near 0.5 (balanced) vs flipping 0/1 (degenerate)
        ax = self.axes[1, 2]
        ax.cla()
        ax.set_title("pred+ rate per epoch")
        ax.plot(epochs, self.epoch_pred_pos_rate, color="orange", marker="o", markersize=3)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="balanced = 0.5")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fraction μ > 0")
        ax.set_ylim([-0.05, 1.05])
        ax.legend(loc="upper right", fontsize=8)

    def _refresh(self):
        # draw_idle queues a redraw; flush_events processes any pending GUI events.
        # together they make the window actually update without a full plt.pause() that would
        # block training.
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        # at end of training: switch off interactive mode and block on the window so the user
        # can inspect the final state. closing the window will then return control to train.py.
        if self._is_alive():
            plt.ioff()
            plt.show()

import numpy as np

from image_processing_pipeline.framework.process_step import process_steps
from image_processing_pipeline.processes.apply_mask import ApplyMask

class AnalyseStatistics(ApplyMask):
  deliverables = {
    "mean": list, "std": list, r"q\d+": list, "weight": list,
    "mode": list,
  }

  # Inherit inputs, and validations from ApplyMask

  def _on_verify_deliverables(self):
    self.quantiles = {}
    for key in self.deliverables_actual:
      if key.startswith("q"):
        quantile_str = key[1:]
        if not quantile_str.isdigit():
          raise ValueError(f"Invalid quantile deliverable '{key}'. Must be in format 'qX' where X is an integer.")
        quantile = int(quantile_str)
        if not (0 <= quantile <= 100):
          raise ValueError(f"Quantile in deliverable '{key}' must be between 0 and 100.")
        if quantile in self.quantiles:
          raise ValueError(f"Duplicate quantile deliverable 'q{quantile}'.")
        self.quantiles[quantile] = []
  
  @staticmethod
  def half_sample_mode(samples: np.ndarray) -> float:
    """Compute the half-sample mode (HSM) for 1D data.

    Parameters
    ----------
    samples : array-like, shape (n,)
      1D data

    Returns
    -------
    mode : float
      Half-sample mode estimate
    """
    n = len(samples)
    if n == 0:
      raise ValueError("samples must be non-empty")

    x = np.sort(samples)
    while n > 2:
      h = (n + 1) // 2  # half-sample size (ceil)
      widths = x[h - 1:] - x[:n - h + 1]
      i = np.argmin(widths)
      x = x[i:i + h]
      n = len(x)

    # Base case
    if n == 1:
      return float(x[0])
    return 0.5 * (float(x[0]) + float(x[1]))

  def _execute(self):
    """Computes statistics of the masked input stack.

    Deliverables:
      - mean: Mean intensity per frame.
      - std: Standard deviation of intensity per frame.
      - qX: X-th percentile of intensity per frame (e.g., q25 for 25th percentile).
      - mode: Value which maximizes the probability density function of each frame.
    """
    if self.mode == "common_footprint":
      combined_mask = np.any(self.mask_stack > 1, axis=0)
      self._get_mask_at_frame = lambda _frame_idx: combined_mask # Override to always return the combined mask

    self.mean, self.std, self.weight, self.mode = [], [], [], []
    for i in range(self.input_stack.shape[0]):
      mask = self._get_mask_at_frame(i)
      norm = np.sum(mask)
      self.weight.append(float(norm))

      samples = np.asarray(self.input_stack[i][mask == 1]).flatten()
      self.mean.append(float(np.sum(samples) / norm))
      self.std.append(float(np.sqrt(np.sum((samples - self.mean[-1])**2) / norm)))

      for quantile in self.quantiles:
        self.quantiles[quantile].append(float(np.percentile(
          self.input_stack[i], quantile, weights=mask, method="inverted_cdf"
        )))

      self.mode.append(float(self.half_sample_mode(samples)))

    for quantile, values in self.quantiles.items():
      setattr(self, f"q{quantile}", values)


process_steps["AnalyseStatistics"] = AnalyseStatistics
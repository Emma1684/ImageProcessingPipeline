import numpy as np

from image_process_pipeline.framework.process_step import process_steps
from image_process_pipeline.processes.apply_mask import ApplyMask

class AnalyseStatistics(ApplyMask):
  deliverables = {"mean": list, "std": list, r"q\d+": list}

  # Inherit inputs, options, and validations from ApplyMask

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

  def _execute(self):
    """
    Computes statistics of the masked input stack.
    Deliverables:
      - mean: Mean intensity per frame.
      - std: Standard deviation of intensity per frame.
      - qX: X-th percentile of intensity per frame (e.g., q25 for 25th percentile).
    """
    if self.mode == "common_footprint":
      combined_mask = np.any(self.mask_stack > 1, axis=0)
      self._get_mask_at_frame = lambda frame_idx: combined_mask # Override to always return the combined mask

    self.mean, self.std = [], []
    for i in range(self.input_stack.shape[0]):
      mask = self._get_mask_at_frame(i)
      norm = np.sum(mask)
      self.mean.append(float(np.sum(self.input_stack[i] * mask) / norm))
      self.std.append(float(np.sqrt(np.sum((self.input_stack[i] - self.mean[-1])**2 * mask) / norm)))
      for quantile in self.quantiles:
        self.quantiles[quantile].append(float(np.percentile(
          self.input_stack[i], quantile, weights=mask, method="inverted_cdf"
        )))

    for quantile, values in self.quantiles.items():
      setattr(self, f"q{quantile}", values)


process_steps["AnalyseStatistics"] = AnalyseStatistics
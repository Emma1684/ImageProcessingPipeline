import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class RemoveOutliers(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray,}
  deliverables = {"filtered_stack": np.ndarray,}

  options = {"lower_quantile": (float, 0.0), "upper_quantile": (float, 1.0)}

  def _execute(self):
    """
    Removes outliers from the image stack.

    For this every pixel value below or above the quantile specified in the options parameter
    is set to the respective quantile values.
    """
    # Get quantiles for each slice
    ql = self.lower_quantile
    qh = self.upper_quantile
    qs = np.quantile(self.input_stack, [ql, qh], axis=(1, 2))

    # Extract low/high, reshape to broadcast over the input stack height
    low  = qs[0, :, None, None]
    high = qs[1, :, None, None]
    self.filtered_stack = np.clip(self.input_stack, low, high)

process_steps["RemoveOutliers"] = RemoveOutliers
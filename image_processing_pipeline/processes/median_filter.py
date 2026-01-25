import numpy as np
from scipy.ndimage import median_filter

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class MedianFilter(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray,}
  deliverables = {"filtered_stack": np.ndarray,}

  options = {"iterations": (int, 1), "size": (int, 5),}

  def _execute(self):
    """
    Applies a median filter to the image stack.

    For this the scipy.ndimage.median_filter function is used. The filter is applied
    `iterations` times with a filter size of `size`.
    """
    self.filtered_stack = np.copy(self.input_stack)
    for _ in range(self.iterations):
      self.filtered_stack = median_filter(self.filtered_stack, size=self.size, axes=(1,2))

process_steps["MedianFilter"] = MedianFilter
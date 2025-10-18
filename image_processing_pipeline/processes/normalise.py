import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class Normalise(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray,}
  deliverables = {"normalised_stack": np.ndarray,}

  def _execute(self):
    """
    Normalises the image stack to [0, 1] range.
    """
    min_vals = self.input_stack.min(axis=(1, 2), keepdims=True)
    max_vals = self.input_stack.max(axis=(1, 2), keepdims=True)
    self.normalised_stack = (self.input_stack - min_vals) / (max_vals - min_vals + 1e-8)

process_steps["Normalise"] = Normalise
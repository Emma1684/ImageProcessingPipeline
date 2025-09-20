import numpy as np

from image_process_pipeline.framework.process_step import AbstractProcessStep, process_steps

class ExtractDimensions(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray}
  deliverables = {
    "depth": int, "width": int, "height": int
  }

  def _on_set_inputs(self):
    assert self.input_stack.ndim in [2, 3], \
      f"Input stack must be 2D or 3D, got {self.input_stack.ndim}D."

  def _execute(self):
    """
    Extracts the shape of a image stack. 2d images will have a depth of 0
    """
    if self.input_stack.ndim == 3:
      self.depth, self.height, self.width = self.input_stack.shape
    else:
      self.depth = 0
      self.height, self.width = self.input_stack.shape

process_steps["ExtractDimensions"] = ExtractDimensions
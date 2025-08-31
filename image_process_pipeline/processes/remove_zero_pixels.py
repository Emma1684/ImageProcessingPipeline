import numpy as np

from image_process_pipeline.pipeline_framework.process_step import AbstractProcessStep, process_steps

class RemoveZeroPixels(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray,}
  deliverables = {"corrected_stack": np.ndarray,}

  options = {"replace_by": (str, "min")}

  def _on_set_options(self):
    assert self.replace_by in ["min", "max"], "Option 'replace_by' must be either 'min' or 'max'."

  def _execute(self):
    """
    Removes dead pixels from the image stack.

    Pixels with value 0 are considered dead pixels and replaced by either the minimum
    (bar 0 value pixels) or the maximum of the respective slice, depending on the options parameter.
    """
    replace_by = 0.
    if self.replace_by == "min":
      replace_by = np.min(self.input_stack[self.input_stack > 0])
    elif self.replace_by == "max":
      replace_by = np.max(self.input_stack)
    
    self.corrected_stack = np.where(self.input_stack == 0, replace_by, self.input_stack)

process_steps["RemoveZeroPixels"] = RemoveZeroPixels
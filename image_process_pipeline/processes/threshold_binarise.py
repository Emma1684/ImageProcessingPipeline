import numpy as np

from image_process_pipeline.framework.process_step import AbstractProcessStep, process_steps

class ThresholdBinarise(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray,}
  deliverables = {"binary_stack": np.ndarray,}

  options = {"threshold": (float, 0.5),}

  def _on_set_inputs(self):
    assert np.all((self.input_stack >= 0) & (self.input_stack <= 1)), "Input stack must be in [0, 1] range."

  def _on_set_options(self):
    assert 0 <= self.threshold <= 1, "Threshold must be in [0, 1] range."

  def _execute(self):
    """
    Binerises the image stack based on a threshold.

    Assume input is normalised to [0,1]. For this every pixel value below the threshold
    is set to 0, every pixel value above or equal to the threshold is set to 1.
    """
    self.binary_stack = self.input_stack > self.threshold

process_steps["ThresholdBinarise"] = ThresholdBinarise
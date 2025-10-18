import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class Invert(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray,}
  deliverables = {"inverted_stack": np.ndarray,}

  def _on_set_inputs(self):
    assert np.all((self.input_stack >= 0) & (self.input_stack <= 1)), "Input stack must be in [0, 1] range."

  def _execute(self):
    """
    Inverts the image stack.

    Assumes input is normalised to [0, 1]. Then the image is inverted by calculating 1 - image.
    """
    self.inverted_stack = 1 - self.input_stack

process_steps["Invert"] = Invert
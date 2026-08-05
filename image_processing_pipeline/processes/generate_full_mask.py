import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class GenerateFullMask(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray,}
  deliverables = {"mask_stack": np.ndarray}
  
  def _execute(self):
    """
    Generates a mask that is '1' everywhere, and has the shape of the input image.
    """
    if self.input_stack is None or self.input_stack.size == 0:
      raise ValueError("Input stack is empty, cannot generate mask")



process_steps["GenerateFullMask"] = GenerateFullMask

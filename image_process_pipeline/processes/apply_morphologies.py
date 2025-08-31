import numpy as np
import scipy.ndimage as nd

from image_process_pipeline.pipeline_framework.process_step import AbstractProcessStep, process_steps

class ApplyMorphologies(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray}
  deliverables = {"morphed_stack": np.ndarray,}

  options = {
    "strategy": (dict, {"binary_erosion": {"iterations": 1}})
  }

  def _execute(self):
    """
    Apply morphological operations to the input stack according to the specified strategy.
    """
    for name, params in self.strategy.items():
      if name == "binary_erosion":
        iterations = params.get("iterations", 1)
        self.input_stack = nd.binary_erosion(self.input_stack, iterations=iterations, axes=(1,2)).astype(self.input_stack.dtype)
      elif name == "binary_dilation":
        iterations = params.get("iterations", 1)
        self.input_stack = nd.binary_dilation(self.input_stack, iterations=iterations, axes=(1,2)).astype(self.input_stack.dtype)
      elif name == "binary_opening":
        iterations = params.get("iterations", 1)
        self.input_stack = nd.binary_opening(self.input_stack, iterations=iterations, axes=(1,2)).astype(self.input_stack.dtype)
      elif name == "binary_closing":
        iterations = params.get("iterations", 1)
        self.input_stack = nd.binary_closing(self.input_stack, iterations=iterations, axes=(1,2)).astype(self.input_stack.dtype)
      else:
        raise ValueError(f"Unknown morphology operation '{name}'")
    self.morphed_stack = self.input_stack

process_steps["ApplyMorphologies"] = ApplyMorphologies
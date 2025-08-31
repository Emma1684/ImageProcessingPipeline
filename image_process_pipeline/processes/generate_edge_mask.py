import numpy as np
import scipy.ndimage as nd

from image_process_pipeline.pipeline_framework.process_step import AbstractProcessStep, process_steps

class GenerateEdgeMask(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray,}
  deliverables = {"edge_mask": np.ndarray,}

  options = {"sigma": (float, 25.)}

  def _on_set_options(self):
    assert self.sigma > 0, "Sigma must be positive."

  def _execute(self):
    """
    Generates an edge mask.

    The masks contains True where an edge is detected, False otherwise.
    Parameter sigma defines the kernel width used for the Gaussian Laplace filter.
    """
    res = nd.gaussian_laplace(self.input_stack, self.sigma, axes=(1,2))
    self.edge_mask = res < 0

process_steps["GenerateEdgeMask"] = GenerateEdgeMask
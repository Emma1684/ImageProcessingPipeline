import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class StarFill(AbstractProcessStep):
  inputs = {"input_mask": np.ndarray}
  deliverables = {"output_mask": np.ndarray}

  def _execute(self):
    """
    Fills the interior of a mask like stack. Pixels are considered to be in the
    interior, if they are surrounded by mask pixels on the same x and y axis.

    Assumptions: input_mask is either binary or contains only 0 and 1s. This is
    not checked however. Using this process for other inputs is considered
    undefined behaviour. 
    """
    cs1 = np.cumsum(self.input_stack, axis=1)
    inner1 = cs1[:,-1,:][:,None,:] * cs1 - cs1**2
    cs2 = np.cumsum(self.input_stack, axis=2)
    inner2 = cs2[:,:,-1][:,:,None] * cs2 - cs2**2
    self.output_mask = inner1 * inner2 >= 1


process_steps["StarFill"] = StarFill
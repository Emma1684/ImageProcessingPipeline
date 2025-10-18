import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class ShrinkToContent(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray}
  deliverables = {"output_stack": np.ndarray, "offset": tuple}

  def _execute(self):
    """
    Iterates through all images in the input stack and identifies a crop, which shrinks the
    stack to the smallest footprint which contains all non-zero/ non-false values.

    Returns the cropped stack and the applied offset (crop width & height implicitly
    communicated through the dimensions of the cropped stack).
    """
    projection = np.any(self.input_stack, axis=0)

    a0_projection = np.any(projection, axis=0)
    a0_nonzero = a0_projection.nonzero()[0]
    a0_offset, a0_max = a0_nonzero[0], a0_nonzero[-1]
    a1_projection = np.any(projection, axis=1)
    a1_nonzero = a1_projection.nonzero()[0]
    a1_offset, a1_max = a1_nonzero[0], a1_nonzero[-1]

    self.offset = (a1_offset, a0_offset)
    self.output_stack = self.input_stack[:, a1_offset:a1_max+1, a0_offset:a0_max+1]


process_steps["ShrinkToContent"] = ShrinkToContent
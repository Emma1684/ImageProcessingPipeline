import numpy as np
import scipy.ndimage as nd

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class GeometryFilterMasks(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray,}
  deliverables = {"filtered_mask_stack": np.ndarray,}

  options = {
    "min_aspect_dx_dy": (float, 0.),
    "min_aspect_dy_dx": (float, 0.),
    "min_area": (float, 0.),
    "max_area": (float, np.inf),
    "min_size_dx": (float, 0.),
    "max_size_dx": (float, np.inf),
    "min_size_dy": (float, 0.),
    "max_size_dy": (float, np.inf),
  }

  def _execute(self):
    """
    Applies geometric filtering to connected components in the input binary mask stack.

    For each connected component in each slice of the input stack, the following criteria are checked:
    - Aspect Ratio: The ratio of width to height (dx/dy) and height to width (dy/dx) must be above specified minimums.
    - Area: The area (width * height) must be within specified minimum and maximum bounds.
    - Size: The width (dx) and height (dy) must be within specified minimum and maximum bounds.
    """
    for n in range(self.input_stack.shape[0]):
      labelled, _ = nd.label(self.input_stack[n,:,:])
      for indices in nd.value_indices(labelled).values():
        dX = np.max(indices[0]) - np.min(indices[0])
        dY = np.max(indices[1]) - np.min(indices[1])
        area = dX*dY
        if area < self.min_area or area > self.max_area:
          self.input_stack[n,*indices] = 0
          continue
        if dX < self.min_size_dx or dX > self.max_size_dx or dY < self.min_size_dy or dY > self.max_size_dy:
          self.input_stack[n,*indices] = 0
          continue
        if dX / (dY + 1e-6) < self.min_aspect_dx_dy or dY / (dX + 1e-6) < self.min_aspect_dy_dx:
          self.input_stack[n,*indices] = 0
    self.filtered_mask_stack = self.input_stack

process_steps["GeometryFilterMasks"] = GeometryFilterMasks
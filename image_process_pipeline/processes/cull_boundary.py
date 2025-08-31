import numpy as np

from image_process_pipeline.pipeline_framework.process_step import AbstractProcessStep, process_steps

class CullBoundary(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray,}
  deliverables = {"culled_stack": np.ndarray,"former_image_shape": tuple, "culled_image_offset": tuple}

  options = {
    "top": ((int, float), 0),
    "bottom": ((int, float), 0),
    "left": ((int, float), 0),
    "right": ((int, float), 0),
    "width": ((int, float), None),
    "height": ((int, float), None),
    # TODO: implement width and height options
  }

  def _on_set_options(self):
    self.former_image_shape = self.input_stack.shape[1:]
    for option in self.options:
      value = getattr(self, option)
      max_pixel_size = self.former_image_shape[1 if option in ['left', 'right'] else 0]

      if isinstance(value, float):
        assert 0 <= value < 1., f"Float option {option} must be in [0, 1.0) range."
        pixel_value = int(value * max_pixel_size)
        setattr(self, option, pixel_value)
      else:
        assert isinstance(value, int) and 0 <= value, f"Integer option {option} must be a non-negative integer."
    return super()._on_set_options()

  def _execute(self):
    """
    Binerises the image stack based on a threshold.

    Assume input is normalised to [0,1]. For this every pixel value below the threshold
    is set to 0, every pixel value above or equal to the threshold is set to 1.
    """
    top, bottom = self.top, self.bottom
    left, right = self.left, self.right

    if top + bottom >= self.former_image_shape[0] or left + right >= self.former_image_shape[1]:
      raise ValueError(f"Culling options too large for image size {self.former_image_shape}: "
                       f"top {top}, bottom {bottom}, left {left}, right {right}.")

    self.culled_stack = self.input_stack[:, top:self.former_image_shape[0]-bottom,
                                        left:self.former_image_shape[1]-right]
    self.culled_image_offset = (top, left)

process_steps["CullBoundary"] = CullBoundary
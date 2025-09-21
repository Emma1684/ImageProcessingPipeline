import numpy as np

from image_process_pipeline.framework.process_step import AbstractProcessStep, process_steps

class ApplyMask(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray, "mask_stack": np.ndarray,}
  deliverables = {"masked_stack": np.ndarray,}

  options = {"mode": (str, "interpolate")}

  def _on_set_inputs(self):
    assert self.input_stack.shape[0] >= self.mask_stack.shape[0], (
      "Input stack must have equal or greater depth than mask stack."
    )

  def _on_set_options(self):
    if self.input_stack.shape[0] == self.mask_stack.shape[0]:
      self.mode = "previous" # No interpolation needed
      # TODO: Write logging
    if self.mode not in {"interpolate", "common_footprint", "previous", "next"}:
      raise ValueError(f"Unknown mode '{self.mode}'. Supported: interpolate, common_footprint, previous, next")
  
  def _get_mask_at_frame(self, frame_idx: int):
    mask_idx = frame_idx * (self.mask_stack.shape[0] - 1) / (self.input_stack.shape[0] - 1)
    lower_idx = int(np.floor(mask_idx))
    upper_idx = int(np.ceil(mask_idx))
    if self.mode == "previous":
      weight_upper = 0
    elif self.mode == "next":
      weight_upper = 1
    else:
      weight_upper = mask_idx - lower_idx
    weight_lower = 1 - weight_upper

    return weight_lower * self.mask_stack[lower_idx] + weight_upper * self.mask_stack[upper_idx]

  def _execute(self):
    """
    Masks the input stack with the mask stack.

    Assumes that the mask stack has a smaller depth than the input stack and that the mask frames
    are evenly distributed throughout the input stack.
    Modes:
      - interpolate: Linearly interpolate between mask frames.
      - common_footprint: Crop to the common footprint of the entire mask stack.
      - previous: Use the previous mask frame for each input frame.
      - next: Use the next mask frame for each input frame.
    """
    if self.mode == "common_footprint":
      # Find common footprint
      combined_mask = np.any(self.mask_stack > 1, axis=0)
      self.masked_stack = self.input_stack[:, combined_mask]
    else:
      self.masked_stack = np.empty_like(self.input_stack)
      for i in range(self.input_stack.shape[0]):
        mask = self._get_mask_at_frame(i)
        self.masked_stack[i,:,:] = self.input_stack[i,:,:] * mask


process_steps["ApplyMask"] = ApplyMask
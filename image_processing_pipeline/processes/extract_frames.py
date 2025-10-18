import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class ExtractFrames(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray}
  deliverables = {"extracted_frames": np.ndarray,}

  options = {"frames": (list, [0])}

  def _on_set_options(self):
    n_frames = self.input_stack.shape[0]
    assert np.min(self.frames) >= -n_frames, \
      f"Frame range exceeded, tried to extract down to frame index {np.min(self.frames)}, " + \
      f"but only {n_frames} frames available."
    assert np.max(self.frames) < n_frames, \
      f"Frame range exceeded, tried to extract up to frame index {np.min(self.frames)}, " + \
      f"but only {n_frames} frames available."

  def _execute(self):
    """
    Extracts a set of frames from the input stack.
    """
    self.extracted_frames = self.input_stack[self.frames,:,:]

process_steps["ExtractFrames"] = ExtractFrames
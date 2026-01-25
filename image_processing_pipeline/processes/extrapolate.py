import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class Extrapolate(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray,}
  deliverables = {"extrapolated_stack": np.ndarray, "extrapolated_frames": list}

  def _execute(self):
    """
    Scans the input stack for missing frames (e.g. every pixel has a 0 value)
    at the beginning and end and extrapolates these frames according by the closest
    frame.
    """
    # Find missing frames
    missing_frames = np.any(self.input_stack, axis=(1,2))
    first_valid_frame = np.argmax(missing_frames)
    last_valid_frame = missing_frames.size - np.argmax(missing_frames[::-1])

    # Transform into prediction for interpolated frames, removing missing frames at the start / end
    self.extrapolated_frames = ~missing_frames
    self.extrapolated_frames[first_valid_frame:last_valid_frame] = False

    # Interpolate
    self.extrapolated_stack = self.input_stack
    self.extrapolated_stack[:first_valid_frame,:,:] = self.input_stack[first_valid_frame,:,:]
    self.extrapolated_stack[last_valid_frame:,:,:] = self.input_stack[last_valid_frame-1,:,:]


process_steps["Extrapolate"] = Extrapolate
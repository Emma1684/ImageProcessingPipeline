import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class Interpolate(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray,}
  deliverables = {"interpolated_stack": np.ndarray, "interpolated_frames": list}

  options = {'mode': (str, "common_footprint")}

  def _on_set_options(self):
    if self.mode not in {"interpolate", "common_footprint", "previous", "next"}:
      raise ValueError(f"Unknown mode '{self.mode}'. Supported: interpolate, common_footprint, previous, next")
    
    if self.mode == "common_footprint":
      assert np.isin(1.0 * self.input_stack, [0., 1.]).all(), \
        "Mode 'common_footprint' requires input_stack to have only 0 & 1 or binary values."
  

  def interpolate(self, i: int, s: int, e: int) -> None:
    match self.mode:
      case "interpolate":
        w1 = (i - s + 1) / (e - s + 2)
        w2 = (e + 1 - i) / (e - s + 2)
        self.interpolated_stack[i,:] = w1 * self.input_stack[s - 1, :] + w2 * self.input_stack[e + 1, :]
      case "previous":
        self.interpolated_stack[i,:] = self.input_stack[s - 1, :]
      case "next":
        self.interpolated_stack[i,:] = self.input_stack[e + 1, :]
      case "common_footprint":
        self.interpolated_stack[i,:] = self.input_stack[s - 1, :] * self.input_stack[e + 1, :]



  def _execute(self):
    """
    Scans the input stack for missing frames (e.g. every pixel has a 0 value)
    and interpolates these frames according to the given mode. Supported modes:
    - interpolate: Interpolates with weights according to the index distance to
      the next valid frames.
    - previous/next: Replaces missing frames by the last/ next valid frame.
    - common_footprint: Replaces the missing frames by the common_footprint of the 
      surronding valid frames. This requires them to only consist of 0 and 1 values
    
    Missing frames at the very beginning and end of the stack are ignored and no
    extrapolation attempts are performed.  
    """
    # Find missing frames
    missing_frames = np.any(self.input_stack, axis=(1,2))
    first_valid_frame = np.argmax(missing_frames)
    last_valid_frame = missing_frames.size - np.argmax(missing_frames[::-1])

    # Transform into prediction for interpolated frames, removing missing frames at the start / end
    self.interpolated_frames = ~missing_frames
    self.interpolated_frames[:first_valid_frame] = False
    self.interpolated_frames[last_valid_frame:] = False

    # Find where the value changes (from False to True or True to False)
    diffs = np.diff(self.interpolated_frames.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0]

    # Interpolate
    self.interpolated_stack = self.input_stack
    if self.mode == "interpolate" and np.any(self.interpolated_frames):
      self.interpolated_stack = self.interpolated_stack.astype("float32")
    for s, e in zip(starts, ends):
      for i in range(s, e + 1):
        self.interpolate(i, s, e)
    self.interpolated_frames = self.interpolated_frames.tolist() # To support serialisation


process_steps["Interpolate"] = Interpolate
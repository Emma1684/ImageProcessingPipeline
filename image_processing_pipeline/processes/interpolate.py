import numpy as np
import pandas as pd
import warnings

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
    
    Frames at he beginning and end are automatically done by the appropiate method (next or previous).  
    """
    missing_frames = np.any(self.input_stack, axis=(1, 2))
    missing_frames_index = np.where(~missing_frames)[0] + 1

    first_valid_frame = np.argmax(missing_frames)
    last_valid_frame = len(missing_frames) - np.argmax(missing_frames[::-1]) - 1

    # Boolean mask of frames to interpolate
    self.interpolated_frames = ~missing_frames  # True where frame is missing

    # Now compute block transitions
    diffs = np.diff(self.interpolated_frames.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0]

    # Catch case where start/end of array is missing
    if self.interpolated_frames[0]:
        starts = np.insert(starts, 0, 0)
    if self.interpolated_frames[-1]:
        ends = np.append(ends, len(self.interpolated_frames) - 1)

    # Warn if any missing
    if len(missing_frames_index) > 0:
        warnings.warn(
            f"Missing frames '{missing_frames_index}'. "
            "Some values were interpolated - does not need to be rectified further",
            UserWarning
        )

    self.interpolated_stack = self.input_stack
    if self.mode == "interpolate" and np.any(self.interpolated_frames):
        self.interpolated_stack = self.interpolated_stack.astype("float32")

    # Run interpolation
    for s, e in zip(starts, ends):
        for i in range(s, e + 1):
            if i < first_valid_frame:
                original_mode = self.mode
                self.mode = "next"
                self.interpolate(i, s, e)
                self.mode = original_mode
            elif i > last_valid_frame:
                original_mode = self.mode
                self.mode = "previous"
                self.interpolate(i, s, e)
                self.mode = original_mode
            else:
                self.interpolate(i, s, e)

    # Convert for serialization
    self.interpolated_frames = self.interpolated_frames.tolist()

process_steps["Interpolate"] = Interpolate
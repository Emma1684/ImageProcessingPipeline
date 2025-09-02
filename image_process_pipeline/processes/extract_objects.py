import numpy as np
import scipy.ndimage as nd

from image_process_pipeline.framework.process_step import AbstractProcessStep, process_steps

class ExtractObjects(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray}
  deliverables = {r"object_stack_\d+": np.ndarray, r"offset_\d+": tuple}

  def _on_verify_deliverables(self):
    self.stacks  = sorted({k for k in self.deliverables if "stack" in k})
    self.offsets = sorted({k for k in self.deliverables if "offset" in k})
    assert len(self.stacks) == len(self.offsets), \
      f"Mismatch between the number of delivered stacks ({len(self.stacks)}) and " +\
      f"number of delivered stacks ({len(self.offsets)})."
    
    for s, o in zip(self.stacks, self.offsets):
      assert s.split("_")[-1] == o.split("_")[-1], \
        f"Mismatch of paring index. Tried to pair deliverables {s} and {o}."

  def _execute(self):
    """
    Extracts a variable, but at execution time constant, number of objects from the stack.

    Currently individual objects must be separated from one another by 0 pixels, but have
    to have a pixel overlap amongs the stack direction.
    """
    labelled, n_labels = nd.label(self.input_stack)
    n_expected = len(self.deliverables) // 2
    assert n_labels == n_expected, \
      f"Mismatch between objects found ({n_labels}) and expected number ({n_expected})"
    ranges = nd.find_objects(labelled)

    for n, (stack, offset) in enumerate(zip(self.stacks, self.offsets)):
      setattr(self, stack, self.input_stack[:, ranges[n][1], ranges[n][2]])
      setattr(self, offset, (ranges[n][1].start, ranges[n][2].start))

process_steps["ExtractObjects"] = ExtractObjects
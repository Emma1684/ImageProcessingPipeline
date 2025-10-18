import re
import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class CombineOffsets(AbstractProcessStep):
  inputs = {r"offset_\d+": tuple}
  deliverables = {"combined_offset": tuple,}

  options = {"extra_horizontal": (int, 0), "extra_vertical": (int, 0)}

  def _execute(self):
    """
    Combines multiple offsets through element-wise addition.
    """
    v_offset = self.extra_vertical
    h_offset = self.extra_horizontal

    for field_name in self.inputs_actual:
      if not re.match(r"offset_\d+", field_name): continue

      offset = getattr(self, field_name)
      v_offset += offset[0]
      h_offset += offset[1]

    self.combined_offset = (v_offset, h_offset)


process_steps["CombineOffsets"] = CombineOffsets
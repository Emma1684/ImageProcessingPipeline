import re
import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class NumberAdder(AbstractProcessStep):
  inputs = {r"number_\d+": float | int}
  deliverables = {"sum": float | int}

  options = {"extra_summand": (int, 0)}

  def _execute(self):
    """
    Combines multiple offsets through element-wise addition.
    """
    self.sum = self.extra_summand

    for field_name in self.inputs_actual:
      if not re.match(r"number_\d+", field_name): continue
      self.sum += getattr(self, field_name)


process_steps["NumberAdder"] = NumberAdder
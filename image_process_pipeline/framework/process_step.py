import re

from abc import ABC, abstractmethod

from image_process_pipeline.framework.typed_data_interface import TypedDataInterface

process_steps = {}

class AbstractProcessStep(ABC, TypedDataInterface):
  inputs: dict[str, type] = {}
  deliverables: dict[str, type] = {}

  options: dict[str, tuple[type, any]] = {}

  def __init__(self,
               inputs: dict = None,
               options: dict = None,
               delivers_id_map: dict = None):
    self.delivers_id_map = delivers_id_map or {}
    self.verify_and_add(self.inputs, inputs or {}, source="Inputs")
    self._on_set_inputs() # Provide hook for sub classes
    self.verify_and_add(self.options, options or {}, source="Options")
    self._on_set_options() # Provide hook for sub classes
    self.verify_ids(self.deliverables, self.delivers_id_map, source="Deliverables")
    self._on_verify_deliverables() # Provide hook for sub classes

  def _on_set_inputs(self):
    """Hook for subclasses to react to inputs being set."""
    pass
  
  def _on_set_options(self):
    """Hook for subclasses to react to options being set."""
    pass
  
  def _on_verify_deliverables(self):
    """Hook for subclasses to react to options being set."""
    pass

  def execute(self):
    self._execute()
    self._validate_deliverables()
    return {self.delivers_id_map[d]: getattr(self, d) for d in self.deliverables}
  
  @abstractmethod
  def _execute(self):
    raise NotImplementedError("Subclasses must implement _execute method")
  
  def _validate_deliverables(self):
    """Check that deliverables exist as attributes and match expected types."""
    for key, expected_type in self.deliverables.items():
      if not hasattr(self, key):
        raise AttributeError(f"Deliverable '{key}' is missing as an attribute.")
      val = getattr(self, key)
      if not isinstance(val, expected_type):
        raise TypeError(
          f"Deliverable '{key}' must be of type {expected_type.__name__}, "
          f"got {type(val).__name__}"
      )
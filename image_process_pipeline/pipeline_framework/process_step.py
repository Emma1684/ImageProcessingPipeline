from abc import ABC, abstractmethod

from chip_analysis.pipeline_framework.data_manager import DataManager

process_steps = {}

class AbstractProcessStep(ABC):
  inputs: dict[str, type] = {}
  deliverables: dict[str, type] = {}

  options: dict[str, tuple[type, any]] = {}

  def __init__(self,
               data_manager: DataManager,
               inputs: dict = None,
               options: dict = None,
               delivers_id_map: dict = None):
    # TODO: remove data_manager from here. Inputs should be passed directly and deliverables returned.
    self.data_manager = data_manager
    self.delivers_id_map = delivers_id_map or {}
    # TODO: validate delivers_id_map keys
    self._verify_and_set_inputs(inputs or {})
    self._verify_and_set_options(options or {})
  
  def _verify_and_set_inputs(self, provided_inputs: dict):
    """Check provided inputs against required schema and set them as attributes."""
    required_keys = set(self.inputs.keys())
    provided_keys = set(provided_inputs.keys())

    # Check exact match
    if required_keys != provided_keys:
      missing = required_keys - provided_keys
      extra = provided_keys - required_keys
      msg = []
      if missing:
        msg.append(f"Missing inputs: {', '.join(missing)}")
      if extra:
        msg.append(f"Unexpected inputs: {', '.join(extra)}")
      raise ValueError("Input validation failed. " + "; ".join(msg))

    # Check types and assign attributes
    for key, expected_type in self.inputs.items():
      obj = self.data_manager.get(provided_inputs[key])
      if not isinstance(obj, expected_type):
        raise TypeError(
          f"Input '{key}' must be of type {expected_type.__name__}, "
          f"got {type(obj).__name__}"
        )
      setattr(self, key, obj)
    
    self._on_set_inputs()

  def _on_set_inputs(self):
    """Hook for subclasses to react to inputs being set."""
    pass

  def _verify_and_set_options(self, provided_options: dict):
    """Check provided options against schema, apply defaults, and set as attributes."""
    for key, (expected_type, default_value) in self.options.items():
      if key in provided_options:
        value = provided_options[key]
        if not isinstance(value, expected_type):
          raise TypeError(
            f"Option '{key}' must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
          )
        setattr(self, key, value)
      else:
        setattr(self, key, default_value)

    # Catch unexpected options
    unexpected = set(provided_options.keys()) - set(self.options.keys())
    if unexpected:
      raise ValueError(f"Unexpected options provided: {', '.join(unexpected)}")
  
    self._on_set_options()
  
  def _on_set_options(self):
    """Hook for subclasses to react to options being set."""
    pass

  def execute(self):
    self._execute()
    self._validate_deliverables()
    self.data_manager.register({
      self.delivers_id_map[d]: getattr(self, d) for d in self.deliverables
    })
  
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
import re

from typing import Union

class TypedDataInterface:
  def __init__(self):
    self.added_fields = {}

  def verify_ids(self, reference: dict[str, Union[type, tuple]], data: dict, source: str = "Field", extra_okay: bool = False):
    """
    Check the keys in the reference against the data.
    
    Regex matches the reference keys and modifies the reference dict to the
    matched instances.
    """
    required_keys = set(reference.keys())
    provided_keys = set(data.keys())

    missing = required_keys - provided_keys
    extra = provided_keys - required_keys
    # Assume missing keys are regex patterns and match
    for required_key in missing.copy():
      type_backup = reference[required_key]
      if isinstance(type_backup, tuple): # Default value present, ignore
        missing.discard(required_key)
        continue
          
      for provided_key in extra.copy():
        if re.match(required_key, provided_key):
          # Register matches and update the dict / sets
          if required_key in reference:
            del reference[required_key]
          reference[provided_key] = type_backup
          missing.discard(required_key) # `Discard` to prevent errors 
          extra.remove(provided_key) # Explicitly catch double deletion

    # Check exact match
    msg = []
    if missing:
      msg.append(f"Missing {source}: {', '.join(missing)}")
    if extra and not extra_okay:
      msg.append(f"Unexpected {source}: {', '.join(extra)}")
    if msg:
      raise ValueError(f"{source} validation failed. " + "; ".join(msg))

  def verify_and_add(self, reference: dict[str, Union[type, tuple]], data: dict, source: str = "Field", extra_okay: bool = False):
    """Check provided inputs against required schema and set them as attributes."""
    self.verify_ids(reference, data, source=source, extra_okay=extra_okay)

    # Check types and assign attributes
    self.added_fields.setdefault(source, set())
    for key, expected_type in reference.items():
      if isinstance(expected_type, tuple): # tuple[type, default_value]
        expected_type, default_value = expected_type

      obj = data[key] if key in data else default_value
      if not isinstance(obj, expected_type):
        raise TypeError(
          f"{source} '{key}' must be of type {expected_type.__name__}, "
          f"got {type(obj).__name__}"
        )
      setattr(self, key, obj)
      self.added_fields[source].add(key)

    if extra_okay:
      extras = {k: data[k] for k in data if k not in reference}
      return extras
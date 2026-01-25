import yaml
import numpy as np
import tifffile as tiff

from abc import ABC, abstractmethod
from pathlib import Path

from image_processing_pipeline.framework.data_manager import DataManager

class AbstractProcessData(ABC):
  def __init__(self, data, name: str):
    self.data = data
    self.name = name

  @abstractmethod
  def _serialise(self, dir: Path):
    """
    Serialise the data.

    This method should either:
    - return `self.data` directly if it is already serialisable, or
    - implement a custom serialisation routine (e.g. save to a file)
      and return the path to the serialised data.
    """
    raise NotImplementedError("Subclasses must implement serialise method")

  def to_yaml(self, dir: Path, serialised_data):
    """
    Write a YAML file containing:
      - data: the serialised data or path
      - type: the fully qualified type name of the original data
    """
    yaml_path = dir / f"{self.name}.yaml"
    with yaml_path.open("w") as f:
      yaml.safe_dump(
        {
          "data": serialised_data,
          "type": f"{type(self.data).__module__}.{type(self.data).__qualname__}"
        },
        f
      )
  
  def serialise(self, dir: Path):
    """
    Ensure dir exists, serialise data, and write YAML metadata.
    """
    dir.mkdir(parents=True, exist_ok=True)
    if not dir.is_dir():
      raise NotADirectoryError(f"{dir} is not a directory")

    serialised_data = self._serialise(dir)
    self.to_yaml(dir, serialised_data)
  
  @abstractmethod
  def load(yaml_file: Path):
    raise NotImplementedError("Subclasses must implement load method")


class CollectableProcessData(AbstractProcessData): pass

class ProcessData(CollectableProcessData):
  def _serialise(self, dir: Path):
    return self.data

  @staticmethod
  def load(yaml_file: Path):
    """
    Load data from a YAML file, casting it to the stored type.
    """
    with yaml_file.open("r") as f:
      meta = yaml.safe_load(f)

    type_str = meta["type"]
    data = meta["data"]

    # Dynamically import type
    module_name, _, class_name = type_str.rpartition(".")
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)

    return cls(data)


class ProcessTiffData(AbstractProcessData):
  def __init__(self, data: np.ndarray, name: str):
    if not isinstance(data, np.ndarray):
      raise TypeError("ProcessTiffData expects a numpy.ndarray")
    if data.ndim not in (2, 3):
      raise ValueError("ProcessTiffData only supports 2D or 3D numpy arrays")
    super().__init__(data, name)

  def _serialise(self, dir: Path):
    """
    Save the numpy array as a TIFF file.
    """
    tif_path = dir / f"{self.name}.tif"
    if "int" in str(self.data.dtype):
      int_type = "uint8" if np.max(self.data) < 256 else "uint16"
      tiff.imwrite(tif_path, self.data.astype(int_type), photometric='minisblack')
    elif "float" in str(self.data.dtype):
      tiff.imwrite(tif_path, self.data.astype("float32"), photometric='minisblack')
    else:
      raise TypeError(
        f"Cannot serialise result {self.name} of type {self.data.dtype}. " +
        "Supported are float and int types."
      )
    return str(tif_path)

  @staticmethod
  def load(yaml_file: Path):
    """
    Load TIFF file back into numpy array.
    """
    with yaml_file.open("r") as f:
      meta = yaml.safe_load(f)
    
    tif_path = Path(meta["data"])
    return tiff.imread(tif_path)


# --- Registry System ---

class ProcessDataSerialiser:
  _instance = None

  def __new__(cls):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
      cls._instance._registry = {}
    return cls._instance

  def register(self, py_type: type, data_cls: type[ProcessData]):
    self._registry[py_type] = data_cls

  def get_data_cls(self, py_type: type):
    return self._registry.get(py_type, ProcessData)

  def save(self, data: dict, details: dict, output_dir: Path):
    """
    Save entries of `data` with a suitable AbstractProcessData wrapper.
    """
    target_dir = output_dir / details["RelativeOutputPath"]
    target_dir.mkdir(exist_ok=True, parents=True)

    if "CollectTo" in details:
      collection = {}
      for k, v in data.items():
        wrapper_cls = self.get_data_cls(type(v))
        if issubclass(wrapper_cls, CollectableProcessData):
          collection[k] = v
        else:
          wrapper = wrapper_cls(v, k)
          wrapper.serialise(target_dir)
      collectionWrapper = ProcessData(collection, details["CollectTo"])
      collectionWrapper.serialise(target_dir)
    else:
      for k, v in data.items():
        wrapper_cls = self.get_data_cls(type(v))
        wrapper = wrapper_cls(v, k)
        wrapper.serialise(target_dir)

  def load(self, yaml_file: Path):
    """
    Load using the ProcessData subclass stored in the YAML.
    """
    with yaml_file.open("r") as f:
      meta = yaml.safe_load(f)


    type_str = meta["type"]
    module_name, _, class_name = type_str.rpartition(".")
    module = __import__(module_name, fromlist=[class_name])
    data_cls = getattr(module, class_name)
    wrapper_cls = self.get_data_cls(data_cls)
    return wrapper_cls.load(yaml_file)

# --- Register standard mappings ---
process_data_serialiser = ProcessDataSerialiser()
process_data_serialiser.register(np.ndarray, ProcessTiffData)
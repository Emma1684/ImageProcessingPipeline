from importlib.metadata import PackageNotFoundError, version

from image_processing_pipeline.framework.process_pipeline import ProcessPipeline
from image_processing_pipeline.framework.visualiser import Visualiser

try:
  __version__ = version("image_process_pipeline")
except PackageNotFoundError:
  __version__ = "unknown"
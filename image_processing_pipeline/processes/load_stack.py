import numpy as np
import tifffile as tiff

from pathlib import Path

from image_processing_pipeline.framework.process_step import process_steps
from image_processing_pipeline.processes.cull_boundary import CullBoundary

class LoadStack(CullBoundary): # Inherit from CullBoundary to reuse its options
  inputs = {"input_path": Path}
  deliverables = {"loaded_stack": np.ndarray,"former_image_shape": tuple, "culled_image_offset": tuple}

  # Options and option verification inherited from CullBoundary

  def _on_set_inputs(self):
    with tiff.TiffFile(self.input_path) as tif:
      assert len(tif.series) == 1, f"Can only load tif files with a single series, got {tif.series} instead."
      self.former_image_shape = tif.pages[0].shape

  def _execute(self):
    """
    Load a stack from a multipage tiff file.
    """
    top, bottom = self.top, self.bottom
    left, right = self.left, self.right

    with tiff.TiffFile(self.input_path) as tif:
      self.loaded_stack = np.array([
        page.asarray()[top:-bottom, left:-right] for page in tif.pages
      ], dtype=tif.pages[0].dtype)

    self.culled_image_offset = (top, left)
      

process_steps["LoadStack"] = LoadStack
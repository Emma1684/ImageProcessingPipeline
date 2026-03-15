import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class VisuliseBlobMask(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray, "blob_mask_stack": np.ndarray,"background_mask_stack": np.ndarray}
  deliverables = {"overlay_mask": np.ndarray}


  def _execute(self):
    """
    Generates a red overlay for detected blobs, and a green overlay for detected background, and overlays both onto the inputted image.
    """
    blob_color=(1.0, 0.0, 0.0)
    bg_color=(0.0, 1.0, 0.0)
    alpha_blob=0.125
    alpha_bg=0.1

    overlay_list = []

    for i in range(self.input_stack.shape[0]):
      # Convert grayscale to RGB
      frame_rgb = np.stack([self.input_stack[i]]*3, axis=-1)
      frame_rgb = frame_rgb / frame_rgb.max()
      overlay = frame_rgb.copy()

      # Background mask
      if self.background_mask_stack[i] is not None:
        mask = self.background_mask_stack[i] > 0
        for c in range(3):
          overlay[mask, c] = (1-alpha_bg)*overlay[mask, c] + alpha_bg*bg_color[c]

      # Blob mask
      if self.blob_mask_stack[i] is not None:
        mask = self.blob_mask_stack[i] > 0
        for c in range(3):
          overlay[mask, c] = (1-alpha_blob)*overlay[mask, c] + alpha_blob*blob_color[c]
      
      overlay_list.append(overlay)
    self.overlay_mask = np.stack(overlay_list, axis=0)



process_steps["VisualiseBlobMask"] = VisuliseBlobMask

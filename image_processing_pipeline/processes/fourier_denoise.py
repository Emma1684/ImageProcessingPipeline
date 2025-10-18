import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class FourierDenoise(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray}
  deliverables = {"denoised_stack": np.ndarray,}

  options = {"denoise_level": (float, 1.)}

  def _on_set_options(self):
    assert 0. < self.denoise_level <= 1., "Denoise level must be in the range (0, 1]."

  def _execute(self):
    """
    """
    ft = np.fft.fft2(self.input_stack)
    ft[np.abs(ft) < self.denoise_level * np.max(np.abs(ft))] = 0
    self.denoised_stack = np.abs(np.fft.ifft2(ft))

process_steps["FourierDenoise"] = FourierDenoise
import numpy as np
import cv2


import matplotlib.pyplot as plt

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class GenerateBlobMask(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray, "mask_stack": np.ndarray}
  deliverables = {"blob_mask": np.ndarray, "background_mask": np.ndarray}
  options = {"pixel_expansion": (int, 3), "threshold": (float, 0.25) }


  @staticmethod
  def remove_global_background(frame, blur_small, blur_large):
    smooth = cv2.GaussianBlur(frame, (0, 0), blur_small)
    bg = cv2.GaussianBlur(smooth, (0, 0), blur_large)

    diff = smooth - bg
    diff[diff < 0] = 0

    scale = np.percentile(diff, 75)
    scale = max(scale, 1e-2)
    diff = diff / scale

    return diff


  @staticmethod
  def dog_response(frame, sigma_small=1.0, sigma_large=10.0):
    return cv2.GaussianBlur(frame, (0,0), sigma_small) - cv2.GaussianBlur(frame, (0,0), sigma_large)

  @staticmethod
  def detect_blobs(frame, roi_mask, sigma_small, sigma_large, 
          thresholdpc,  min_blob_area, dilate_px, min_frame_std):
  
    # Do not process if there is no spatial large variataion across the frame (i.e. diffuse chamber).
    # Value set to 0.6. Usually in diffuse std is around 0.5, and spikes up to 2.5ish when condensates are present
    if np.std(frame) < min_frame_std:
      return np.zeros_like(frame, dtype=np.uint8), np.zeros_like(frame)
    
    #Removes darkest pixels (10% darkest) from processing - focus on blobs and brighter areas
    valid_pixels = frame > np.percentile(frame, 10)
    
    #calculates difference of gaussians, no need to remove the percentile
    #dog = self.dog_response(frame, sigma_small, sigma_large)
    dog = cv2.GaussianBlur(frame, (0,0), sigma_small) - cv2.GaussianBlur(frame, (0,0), sigma_large)
    roi_values = dog[roi_mask > 0]
    threshold = np.percentile(roi_values, thresholdpc)

    mask = ((roi_mask>0) & valid_pixels & (dog>threshold)).astype(np.uint8)
    
    # remove small blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    mask_clean = np.zeros_like(mask)
    for i in range(1, num_labels):
      if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
        mask_clean[labels == i] = 255
    
    # dilate
    if dilate_px>0:
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
      mask_clean = cv2.dilate(mask_clean, kernel, iterations=dilate_px)
    
    return mask_clean, dog

  def _execute(self):
    """
    Takes an input stack and an initial input mask. 
    The process first removes the global background. Subsequently uses 
    differences of gaussians (dog) to detect regions of intensity within the image. 
    The first detection is peformed with more stringent conditions 
    (higher thresholding, minimal size) to detect concentarted regions (i.e. blobs).
    
    The second detection is less stringent, and detects the blobs and hazy regions around
    them. The final background mask is the inverse of this mask, and regions that are positive
    within the initial input mask.
    
    The final blob mask takes only regions that are positive within the blob mask, and 
    the initial input mask.
    """
  
    image_stack = np.asarray(self.input_stack, dtype=np.float32)
    region_masks = np.asarray(self.mask_stack, dtype=np.uint8)

    num_frames, H, W = image_stack.shape

  
    blob_mask = np.zeros((num_frames, H, W), dtype=np.uint8)
    background_mask = np.zeros((num_frames, H, W), dtype=np.uint8)

    # Background removal
    diff_stack = np.array([
      self.remove_global_background(f, blur_small=1.0, blur_large=80.0)
      for f in image_stack
    ])

    # Process frames
    for i in range(num_frames):
      # Detects blobs. Settings, sigma small = 0.5, sigma large = 2.0, threshold = 80   
      mask, _ = self.detect_blobs(
        diff_stack[i],
        region_masks[i],
        sigma_small=0.5,
        sigma_large=5.0,
        thresholdpc=80,
        min_blob_area=1,
        dilate_px=1,
        min_frame_std=0.6
      )

      roi_blob_mask = ((mask > 0) & (region_masks[i] > 0)).astype(np.uint8)
      blob_mask[i] = roi_blob_mask

    for i in range(num_frames):
      # Detects blobs and haze around them,   
      mask, _ = self.detect_blobs(
        diff_stack[i],
        region_masks[i],
        sigma_small=0.5,
        sigma_large=10.0,
        thresholdpc=40,
        min_blob_area=1,
        dilate_px=2,
        min_frame_std=0.6
      )

      clean_roi = (region_masks[i] > 0) & (mask == 0)
      background_mask[i] = clean_roi.astype(np.uint8)
  

    # Assign deliverables ONLY at the end
    self.blob_mask = blob_mask
    self.background_mask = background_mask


process_steps["GenerateBlobMask"] = GenerateBlobMask

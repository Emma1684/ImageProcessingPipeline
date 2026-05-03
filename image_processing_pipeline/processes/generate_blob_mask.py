import numpy as np
import cv2

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class GenerateBlobMask(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray, "mask_stack": np.ndarray}
  deliverables = {"blob_mask": np.ndarray, "background_mask": np.ndarray}
  options = {"blur_small": (float, 1.0), "blur_large": (float, 50.0),
             "sigma_small": (float, 1.0), "sigma_large": (float, 5.0),
             "std_sustain": (int,3), "std_threshold": (float, 0.55),
             "core_thresh" : (float, 1.5), 
             "halo_max_px" : (float,10.0), "halo_intensity_factor": (float,2.5),
             "smooth_sigma":(float,1.2)}


  @staticmethod
  def remove_global_background(frame, blur_small, blur_large):
      smooth = cv2.GaussianBlur(frame, (0, 0), blur_small)
      bg = cv2.GaussianBlur(smooth, (0, 0), blur_large)
      diff = smooth - bg
      diff[diff < 0] = 0
      scale = max(np.percentile(diff, 75), 1e-6)
      return diff / scale

  @staticmethod
  def dog_response(frame, sigma_small, sigma_large):
    return cv2.GaussianBlur(frame, (0,0), sigma_small) - cv2.GaussianBlur(frame, (0,0), sigma_large)

  @staticmethod
  def detect_onset_from_std(diff_stack, std_sustain, std_threshold):

      T = diff_stack.shape[0]
      std_vals = np.zeros(T, dtype=np.float32)
      
      for t in range(T):
          std_vals[t] = np.std(diff_stack[t])
      onset = T  # default: no onset detected
      for t in range(0, T - std_sustain + 1):
          if np.all(std_vals[t : t + std_sustain] > std_threshold
                    ):
              onset = t
              break    
      
      return onset
  
  @staticmethod
  def detect_core_from_dog(dog, roi_mask, core_thresh):
      roi = roi_mask > 0
      core = np.zeros_like(dog, dtype=np.uint8)
      vals = dog[roi]
       
      mu = np.mean(vals)
      sigma = np.std(vals)

      thr = mu + core_thresh * sigma
      core = (dog > thr) & roi

      return core
    
  

  @staticmethod
  def segment_frame_simple(orig_frame, roi_mask, core_mask,
                         halo_max_px,halo_intensity_factor,smooth_sigma):

    roi = roi_mask
    core = core_mask

    # Smooth image
    sm = cv2.GaussianBlur(orig_frame, (0, 0), smooth_sigma)

    # Distance from core
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 3)

    # Estimate background stats (far from core)
    far = roi & (dist > halo_max_px)
    bg_vals = sm[far] if np.any(far) else sm[roi]

    bg_med = np.median(bg_vals)
    bg_std = np.std(bg_vals) + 1e-6

    # Halo condition: close + brighter than background
    halo = (
        roi &
        (~core) &
        (dist <= halo_max_px) &
        (sm > bg_med + halo_intensity_factor * bg_std)
    )

    # Clean halo: only keep regions touching core
    core_dil = cv2.dilate(core.astype(np.uint8), np.ones((3,3), np.uint8)) > 0

    n, labels = cv2.connectedComponents(halo.astype(np.uint8))
    halo_clean = np.zeros_like(halo)

    for i in range(1, n):
        comp = labels == i
        if np.any(comp & core_dil):
            halo_clean |= comp

    # Convert to uint8
    core_u8 = (core.astype(np.uint8) * 1)
    halo_u8 = (halo_clean.astype(np.uint8) * 1)

    halo_u8[core_u8 > 0] = 0
    bg = roi & (core_u8 == 0) & (halo_u8 == 0)
    bg_u8 = (bg.astype(np.uint8) * 1)

    return core_u8, halo_u8, bg_u8
  
                          
  
  def _execute(self):  
    
    image_stack = np.asarray(self.input_stack, dtype=np.float32)
    region_masks = np.asarray(self.mask_stack, dtype=np.uint8)  
    
    #Creating empty arrays etc
    T, H, W = image_stack.shape
    roi_stack = region_masks > 0
    diff_stack = np.zeros_like(image_stack, dtype=np.float32)
    core_masks = np.zeros((T, H, W), dtype=np.uint8)
    halo_masks = np.zeros((T, H, W), dtype=np.uint8)
    bg_masks = np.zeros((T, H, W), dtype=np.uint8)

    for t in range(T):
        diff = self.remove_global_background(image_stack[t].astype(np.float32), self.blur_small, self.blur_large)
        diff_stack[t] = diff
    
    onset = self.detect_onset_from_std(diff_stack, self.std_sustain, self.std_threshold)
    
    for t in range(T):
        roi_mask = roi_stack[t]
        roi = roi_mask > 0

        if t < onset:
            bg_masks[t][roi] = 1
            continue

        dog = self.dog_response(diff_stack[t], self.sigma_small, self.sigma_large)
        
        core_dog = self.detect_core_from_dog(dog, roi_mask, self.core_thresh)
        
        core, halo, bg = self.segment_frame_simple(
            image_stack[t].astype(np.float32),roi_mask,core_dog,
            self.halo_max_px, self.halo_intensity_factor, self.smooth_sigma)

        core_masks[t] = core
        halo_masks[t] = halo
        bg_masks[t] = bg

    self.blob_mask = core_masks
    self.background_mask = bg_masks

process_steps["GenerateBlobMask"] = GenerateBlobMask

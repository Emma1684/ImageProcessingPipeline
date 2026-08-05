import numpy as np
import cv2


from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class GenerateBlobMask(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray, "mask_stack": np.ndarray}
  deliverables = {"blob_mask": np.ndarray, "background_mask": np.ndarray, "blob_num":list}
  options = { "blur_sigma": (float, 3.5), "core_thresh" : (float, 1.5), 
             "std_sustain": (int,5), "std_threshold": (float, 0.55),
             "halo_max_px" : (float,10.0), "halo_intensity_factor": (float,2.5),


  @staticmethod
  def remove_global_background(frame, blur_sigma):
      bg = cv2.GaussianBlur(frame, (0, 0), blur_sigma)
      diff = frame - bg
      diff[diff < 0] = 0
      scale = max(np.percentile(diff, 75), 1e-6)
      return diff / scale


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
      print(onset)
      return onset
  
  @staticmethod
  def detect_core(diff_stack, roi, core_threshold, size_threshold):
    
    core = np.zeros_like(diff_stack, dtype=np.uint8)
    vals = diff_stack[roi]
    
    mu = np.mean(vals)
    sigma = np.std(vals)
    thr = mu + core_threshold * sigma

    core = (diff_stack > thr) & roi

    #Size limit cores
    n, core_labels = cv2.connectedComponents(core.astype(np.uint8))
    core_clean = np.zeros_like(core, dtype=bool)

    for i in range(1, n):  # skip background
        comp = (core_labels == i)
        if np.count_nonzero(comp) >= size_threshold:
            core_clean |= comp

    n, labels = cv2.connectedComponents(core_clean.astype(np.uint8))
    return core_clean, n  
  

  @staticmethod
  def segment_frame_simple(orig_frame, roi, core,
                         halo_max_px,halo_intensity_factor):

    # Distance from core
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 3)

    # Estimate background stats (far from core)
    far = roi & (dist > halo_max_px)

    if np.any(far):
        bg_vals = orig_frame[far]
        dim_bg_vals = bg_vals[bg_vals < np.percentile(bg_vals, 50)]
    else:
        bg_vals = orig_frame[roi]
        dim_bg_vals = bg_vals[bg_vals < np.percentile(bg_vals, 25)]
        
          
    bg_med = np.median(dim_bg_vals)
    bg_std = np.std(dim_bg_vals) + 1e-6

    # Halo condition: close + brighter than background
    halo = (
        roi &
        (~core) &
        (dist <= halo_max_px) &
        (orig_frame > bg_med + halo_intensity_factor * bg_std)
    )

    # Clean halo: only keep regions touching core
    core_dil = cv2.dilate(core.astype(np.uint8), np.ones((3,3), np.uint8)) > 0

    n, labels = cv2.connectedComponents(halo.astype(np.uint8))
    halo_clean = np.zeros_like(halo)

    for i in range(1, n):
        comp = labels == i
        if np.any(comp & core_dil):
            halo_clean |= comp

    bg = roi & (~core) & (~halo_clean)

    # Convert to int32
    core_int32 = (core.astype(np.int32))
    halo_int32 = (halo_clean.astype(np.int32))
    bg_int32 = (bg.astype(np.int32))
    return core_int32, halo_int32, bg_int32
  
                       
  
  def _execute(self):  
    
    image_stack = np.asarray(self.input_stack, dtype=np.float32)
    #region_masks = np.asarray(self.mask_stack, dtype=np.int32)
    #image_stack = self.input_stack.astype(np.float32)
    region_masks = self.mask_stack.astype(np.int32)
    roi_stack = region_masks > 0 #boolstack

    T, H, W = image_stack.shape
    
    diff_stack = np.zeros_like(image_stack, dtype=np.float32)
    core_masks = np.zeros((T, H, W))
    halo_masks = np.zeros((T, H, W))
    bg_masks = np.zeros((T, H, W))
    num_blob = []

    for t in range(T):
        diff = self.remove_global_background(image_stack[t], self.blur_sigma)
        diff_stack[t] = diff
        
    onset = self.detect_onset_from_std(diff_stack, self.std_sustain, self.std_threshold)
    
    for t in range(T):
        roi_mask = roi_stack[t]

        if t < onset:
            bg_masks[t]=roi_mask
            num_blob.append(0)
            continue
                
        core_dog, nblob = self.detect_core(diff_stack[t], roi_mask, self.core_thresh, self.size_threshold)
        
        core, halo, bg = self.segment_frame_simple(
            image_stack[t],roi_mask,core_dog,
            self.halo_max_px, self.halo_intensity_factor)

        core_masks[t] = core
        halo_masks[t] = halo
        bg_masks[t] = bg
        num_blob.append(nblob)

    self.blob_mask = core_masks
    self.background_mask = bg_masks
    self.blob_num = num_blob

process_steps["GenerateBlobMask"] = GenerateBlobMask

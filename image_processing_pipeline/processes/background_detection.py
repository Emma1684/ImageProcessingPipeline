import numpy as np
import cv2

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

class GenerateBlobMask(AbstractProcessStep):
    inputs = {"input_stack": np.ndarray, "mask_stack": np.ndarray }
    deliverables = {"blob_mask": np.ndarray,}
    options = {"pixel_expansion": (int, 5), "threshold": (float, 1.0) }


    @staticmethod
    def remove_temporal_baseline(stack, percentile=10, n_frames=4):
        """
        Estimate per-pixel baseline from early frames and subtract.
        """
        baseline = np.percentile(stack[0:n_frames], percentile, axis=0)
        out = stack - baseline
        out[out < 0] = 0
        return out


    @staticmethod
    def remove_global_background_chemical(
        frame,
        blur_small=1.5,
        blur_large=80.0,
        min_scale=1e-2
    ):
        """
        Spatial background removal that does NOT chase rising intensity.
        Output is normalized to O(1).
        """

        # Light denoising
        smooth = cv2.GaussianBlur(frame, (0, 0), blur_small)

        # Strong low-pass background
        bg = cv2.GaussianBlur(smooth, (0, 0), blur_large)

        # Local contrast
        diff = smooth - bg
        diff[diff < 0] = 0

        # Robust, stable normalization
        scale = np.percentile(diff, 75)
        scale = max(scale, min_scale)

        return diff / scale


    @staticmethod
    def dog_response(frame, sigma_small, sigma_large):
        """
        Returns the difference of gaussians (small_sigma - large_sigma).
        """
        return (
            cv2.GaussianBlur(frame, (0, 0), sigma_small)
            - cv2.GaussianBlur(frame, (0, 0), sigma_large)
        )


    def detect_blobs(
        self,
        frame,
        roi_mask,
        sigma_small=3.0,
        sigma_large=10.0,
        threshold=0.6,
        min_blob_area=4,
        dilate_px=3,
        min_frame_std=0.1
    ):
        """
        Detect bright blobs in normalized contrast images.
        """

        # ---- Safety: skip flat frames ----
        if np.std(frame) < min_frame_std:
            return np.zeros_like(frame, dtype=np.uint8), np.zeros_like(frame)

        # ---- DoG enhancement ----
        dog = self.dog_response(frame, sigma_small, sigma_large)
        dog -= np.median(dog)
        dog[dog < 0] = 0

        # ---- Threshold + ROI ----
        mask = ((roi_mask > 0) & (dog > threshold)).astype(np.uint8)

        # ---- Remove small objects ----
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clean = np.zeros_like(mask)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
                clean[labels == i] = 255

        # ---- Dilate ----
        if dilate_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            clean = cv2.dilate(clean, kernel, iterations=dilate_px)

        return clean, dog


    def _execute(self):
        """
        Finds condensates and generates a mask around them.

        Options
        - pixel_expansion: Determines how many pixels around the detected condensate the mask extends
        - threshold: Determines sensitivity level of detection
        """
        # ---- Step 1: temporal baseline ----
        stack = self.remove_temporal_baseline(self.image_stack)

        # ---- Step 2: spatial background ----
        diff_stack = np.array([
            self.remove_global_background_chemical(
                f,
                blur_small=1.0,
                blur_large=80.0
            )
            for f in stack
        ])

        # ---- Step 3: blob detection ----
        self.blob_mask = np.empty_like(diff_stack, dtype=np.uint8)

        num_frames = self.image_stack.shape[0]
        for i in range(num_frames):
            mask, _ = self.detect_blobs(
                diff_stack[i],
                self.mask_stack[i],
                sigma_small=3.0,
                sigma_large=10.0,
                threshold=5,
                min_blob_area=4,
                dilate_px=3,
                min_frame_std=1
            )
            self.blob_mask[i] = mask

process_steps["GenerateBlobMask"] = GenerateBlobMask

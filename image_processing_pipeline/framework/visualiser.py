import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Visualiser:
  @staticmethod
  def show_image_stack(image_stack: np.ndarray, title: str = "Image Stack", cmap: str = "gray", layout: str = "row"):
    """Display a stack of images in a grid layout."""
    if image_stack.ndim == 2:
      image_stack = image_stack[np.newaxis, ...]
    if image_stack.ndim != 3:
      raise ValueError("image_stack must be a 3D numpy array (num_images, height, width) or 2D (height, width).")
    
    num_images = image_stack.shape[0]
    image_shape = image_stack.shape[1:]
    if layout == "row":
      _fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 0.5 + 4 * image_shape[0] / image_shape[1]))
    elif layout == "square":
      cols = int(np.ceil(np.sqrt(num_images)))
      rows = int(np.ceil(num_images / cols))
      _fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    
    if num_images == 1:
      axes = [axes]
    else:
      axes = axes.flatten()

    for i in range(num_images):
      im = axes[i].imshow(image_stack[i], cmap=cmap)
      axes[i].set_title(f"Image {i+1}")
      # plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
      divider = make_axes_locatable(axes[i])
      cax = divider.append_axes("right", size="5%", pad=0.05)
        
      plt.colorbar(im, cax=cax)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
  
  @staticmethod
  def show_histograms(image_stack: np.ndarray, title: str = "Histograms", bins: int = 100, yscale: str = "log", layout: str = "row"):
    """Display histograms of pixel intensities for each image in the stack."""
    if image_stack.ndim == 2:
      image_stack = image_stack[np.newaxis, ...]
    if image_stack.ndim != 3:
      raise ValueError("image_stack must be a 3D numpy array (num_images, height, width) or 2D (height, width).")
    
    num_images = image_stack.shape[0]
    if layout == "row":
      _fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
    elif layout == "square":
      cols = int(np.ceil(np.sqrt(num_images)))
      rows = int(np.ceil(num_images / cols))
      _fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    if num_images == 1:
      axes = [axes]
    else:
      axes = axes.flatten()

    for i in range(num_images):
      axes[i].hist(image_stack[i].ravel(), bins=bins, color='blue', alpha=0.7)
      axes[i].set_title(f"Frame {i+1}")
      #axes[i].set_xlabel("Intensity")
      #axes[i].set_ylabel("Frequency")
      axes[i].set_yscale(yscale)

      axes[i].tick_params(axis='x', labelsize=7)
      axes[i].tick_params(axis='y', labelsize=7)
      #axes[i].tick_params(axis='x', labelrotation=45)
      
      xmax = image_stack[i].max()

      # round min down to nearest power-of-10 or to zero
      xmin_rounded = 0 

      # round max up to nearest 10
      xmax_rounded = int(np.ceil(xmax / 1000) * 1000)

      # middle rounded
      xmid_rounded = int((xmin_rounded + xmax_rounded) / 2)
      axes[i].set_xticks([xmin_rounded, xmid_rounded, xmax_rounded])
      axes[i].set_xticklabels([str(xmin_rounded), str(xmid_rounded), str(xmax_rounded)])



    for i in range(num_images, len(axes)):
      axes[i].axis('off')

     # Increase space between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
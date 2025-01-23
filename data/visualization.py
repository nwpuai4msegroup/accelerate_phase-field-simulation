from typing import Union
import cv2
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt


def show_video_line(data, frame_index=None, ncols=1, vmax=0.6, vmin=0.0, cmap='gray', norm=None, cbar=False,
                    format='png', out_path=None, use_rgb=False):
    """Generate images with a video sequence or show/save a single frame."""
    # Ensure the frame index is valid
    if frame_index is not None and (frame_index < 0 or frame_index >= data.shape[0]):
        raise ValueError("Frame index out of range.")

    # Adjust data shape if needed
    if len(data.shape) > 3:
        data = data.swapaxes(1, 2).swapaxes(2, 3)

    # Single frame display and save
    if frame_index is not None:
        fig, ax = plt.subplots(figsize=(3.25, 3))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        frame = data[frame_index]
        if use_rgb:
            im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            im = ax.imshow(frame, cmap=cmap, norm=norm)

        ax.axis('off')
        im.set_clim(vmin, vmax)

        if cbar:
            fig.colorbar(im, ax=ax, shrink=0.1)

        plt.show()

        if out_path is not None:
            fig.savefig(out_path, format=format, pad_inches=0, bbox_inches='tight')
        plt.close()
        return  # End function early after showing single frame

    # Multi-frame display code remains mostly unchanged
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(3.25 * ncols, 3))
    plt.subplots_adjust(wspace=0.01, hspace=0)

    images = []
    if ncols == 1:
        if use_rgb:
            im = axes.imshow(cv2.cvtColor(data[0], cv2.COLOR_BGR2RGB))
        else:
            im = axes.imshow(data[0], cmap=cmap, norm=norm)
        images.append(im)
        axes.axis('off')
        im.set_clim(vmin, vmax)
    else:
        for t, ax in enumerate(axes.flat):
            if t >= data.shape[0]:  # Ensure we don't exceed data length
                break
            if use_rgb:
                im = ax.imshow(cv2.cvtColor(data[t], cv2.COLOR_BGR2RGB),
                               cmap='gray')  # This cmap should match RGB usage
            else:
                im = ax.imshow(data[t], cmap=cmap, norm=norm)
            images.append(im)
            ax.axis('off')
            im.set_clim(vmin, vmax)

    if cbar and ncols > 1:
        cbaxes = fig.add_axes([0.9, 0.15, 0.04 / ncols, 0.7])
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.1, cax=cbaxes)

    plt.show()

    if out_path is not None:
        fig.savefig(out_path, format=format, pad_inches=0, bbox_inches='tight')
    plt.close()



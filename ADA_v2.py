# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 12:50:05 2025

@author: Akhil
"""
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.feature import peak_local_max
from scipy.spatial.distance import pdist
from skimage.io import imread
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity, adjust_gamma, equalize_adapthist
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os

class AtomAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Atom Center Analyzer")

        self.load_btn = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_btn.pack(pady=5)

        self.process_btn = tk.Button(root, text="Process and Analyze", command=self.process_image, state='disabled')
        self.process_btn.pack(pady=5)

        self.output = tk.Text(root, height=10, width=60)
        self.output.pack(pady=5)

        self.image = None
        self.coordinates = None

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")])
        if not file_path:
            return
        self.image_path = file_path
        self.image = imread(file_path, as_gray=True)
        self.process_btn.config(state='normal')
        self.output.insert(tk.END, f"Loaded image: {file_path}\n")

    def process_image(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded.")
            return

        project_name = simpledialog.askstring("Project Name", "Enter a name for the project (used for saved files):")
        if not project_name:
            project_name = "project"

        gray_image = self.image.copy()
        height, width = gray_image.shape

        p2, p98 = np.percentile(gray_image, (2, 98))
        contrast_stretched = rescale_intensity(gray_image, in_range=(p2, p98), out_range=(0, 1))
        adaptive_equalized = equalize_adapthist(contrast_stretched)
        gamma_corrected = adjust_gamma(adaptive_equalized, gamma=0.8)
        smoothed_image = gaussian(gamma_corrected, sigma=1)

        cropped_image = smoothed_image[int(height*0.1):int(height*0.9), int(width*0.1):int(width*0.9)]
        mean_val = np.mean(cropped_image)
        std_val = np.std(cropped_image)
        adaptive_threshold = mean_val + 1.5 * std_val
        raw_coordinates = peak_local_max(cropped_image, min_distance=3, threshold_abs=adaptive_threshold, indices=True)
        raw_coordinates[:, 0] += int(height * 0.1)
        raw_coordinates[:, 1] += int(width * 0.1)

        unique_coords = []
        intensity_map = gray_image
        radius = 3
        for y, x in raw_coordinates:
            overlapping = False
            for yc, xc in unique_coords:
                if np.hypot(y - yc, x - xc) < radius:
                    overlapping = True
                    if intensity_map[y, x] > intensity_map[yc, xc]:
                        unique_coords.remove((yc, xc))
                        unique_coords.append((y, x))
                    break
            if not overlapping:
                unique_coords.append((y, x))

        coordinates = np.array(unique_coords)
        self.coordinates = coordinates

        pixel_size_nm = simpledialog.askfloat("Pixel Size", "Enter pixel size in nm (default: 0.05):", initialvalue=0.05)
        if not pixel_size_nm:
            pixel_size_nm = 0.05

        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.imshow(gamma_corrected, cmap='gray')
        ax1.scatter(coordinates[:, 1], coordinates[:, 0], s=20, edgecolors='red', facecolors='none')
        ax1.set_title("Detected Atom Centers")
        ax1.axis('off')
        plt.tight_layout()
        plt.show()

        if len(coordinates) <= 1:
            self.output.insert(tk.END, "Not enough atom centers detected to compute distances.\n")
            return

        dists = pdist(coordinates) * pixel_size_nm

        # Ask user-defined distance range
        range_str = simpledialog.askstring("Distance Range", "Enter distance range for histogram (e.g., 0.0,0.5):", initialvalue="0.0,0.5")
        try:
            range_min, range_max = [float(r.strip()) for r in range_str.split(",")]
        except:
            self.output.insert(tk.END, "Invalid input. Using default range 0.0 to 0.5 nm.\n")
            range_min, range_max = 0.0, 0.5

        filtered_dists = dists[(dists >= range_min) & (dists <= range_max)]
        bin_heights, bin_edges = np.histogram(filtered_dists, bins=30)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        peaks, _ = find_peaks(bin_heights, prominence=1)
        self.output.insert(tk.END, f"Estimated number of peaks: {len(peaks)}\n")

        def multi_Gaussian(x, *params):
            y = np.zeros_like(x)
            for i in range(0, len(params), 3):
                amp = params[i]
                mu = params[i+1]
                sigma = params[i+2]
                y += amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
            return y

        initial_guess = []
        for peak_idx in peaks:
            amp = bin_heights[peak_idx]
            mu = bin_centers[peak_idx]
            sigma = 0.005
            initial_guess.extend([amp, mu, sigma])

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(filtered_dists, bins=30, color='lightgray', edgecolor='black', label='Histogram')

        fitted = False
        try:
            popt, _ = curve_fit(multi_Gaussian, bin_centers, bin_heights, p0=initial_guess)
            fitted_curve = multi_Gaussian(bin_centers, *popt)
            ax2.plot(bin_centers, fitted_curve, 'r-', lw=2, label='Gaussian Fit')
            for i in range(1, len(popt), 3):
                self.output.insert(tk.END, f"Peak {i//3 + 1}: {popt[i]:.4f} nm\n")
            fitted = True
        except Exception as e:
            self.output.insert(tk.END, f"Gaussian fitting failed: {str(e)}\n")

        ax2.set_title(f"Atom Pair Distance Distribution ({range_min}–{range_max} nm)")
        ax2.set_xlabel("Distance (nm)")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        plt.tight_layout()
        plt.show()

        save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
        if not save_dir:
            self.output.insert(tk.END, "Save canceled.\n")
            return

        plot_path = os.path.join(save_dir, f"{project_name}_distance_distribution_plot.png")
        fig2.savefig(plot_path)
        self.output.insert(tk.END, f"Plot saved to: {plot_path}\n")

        txt_path = os.path.join(save_dir, f"{project_name}_distance_distribution_data.txt")
        with open(txt_path, 'w') as f:
            f.write("Distance(nm)\tFrequency\n")
            for center, height in zip(bin_centers, bin_heights):
                f.write(f"{center:.6f}\t{height}\n")
        self.output.insert(tk.END, f"Data saved to: {txt_path}\n")

        atom_image_path = os.path.join(save_dir, f"{project_name}_atom_centers_detected.png")
        fig1.savefig(atom_image_path)
        self.output.insert(tk.END, f"Atom center image saved to: {atom_image_path}\n")

if __name__ == '__main__':
    root = tk.Tk()
    app = AtomAnalyzerGUI(root)
    root.mainloop()

import os
import scipy.io
import matplotlib.pyplot as plt

# --- 1. Setup: Define file path and patient ID ---
base_path = r"./PlaTiF Dataset/Patient Data_Part 1"
patient_id = "Patient_ID_001"
file_path = os.path.join(base_path, f"{patient_id}.mat")

# --- 2. Load Data ---
mat_data = scipy.io.loadmat(file_path)
patient_data = mat_data[patient_id]

# --- 3. Access Image Data from the 'im0' struct ---
im0_data = patient_data[0, 0]["im0"]
original_img = im0_data[0, 0]["OriginalImage"]
bw_mask = im0_data[0, 0]["BW"]
masked_img = im0_data[0, 0]["maskedImage"]
Schatzker_label = im0_data[0, 0]["label"][0, 0]

# --- 4. Visualize the Data (with conditional plotting) ---

# Check if Coronal_CT exists to determine subplot layout
if "Coronal_CT" in patient_data.dtype.names:
    num_cols = 4
    fig, axes = plt.subplots(1, num_cols, figsize=(20, 5))  # Wider figure for 4 plots
    coronal_ct_image = patient_data[0, 0]["Coronal_CT"]
else:
    num_cols = 3
    fig, axes = plt.subplots(1, num_cols, figsize=(18, 6))  # Original figure size

fig.suptitle(f"Data for {patient_id}", fontsize=16)

# Plot the common images
axes[0].imshow(original_img, cmap="gray")
axes[0].set_title(f"Schatzker Classification Label: {Schatzker_label}")
axes[0].axis("off")

axes[1].imshow(bw_mask, cmap="gray")
axes[1].set_title("Tibia Bone Plateau Mask")
axes[1].axis("off")

axes[2].imshow(masked_img, cmap="gray")
axes[2].set_title("Segmented Tibia Bone")
axes[2].axis("off")

# Plot the optional fourth image only if it exists
if num_cols == 4:
    axes[3].imshow(coronal_ct_image, cmap="gray")
    axes[3].set_title("A Coronal CT Slice")
    axes[3].axis("off")

plt.tight_layout()
plt.show()

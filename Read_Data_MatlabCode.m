clc
clear
close all

% --- 1. Setup: Define file path and patient ID ---
basePath = 'C:\Users\Ali\Desktop\PlaTiF Dataset';
patientID = 'Patient_ID_001';
filePath = fullfile(basePath, [patientID, '.mat']);

% --- 2. Load Data ---
dataStruct = load(filePath);
patientData = dataStruct.(patientID);

% --- 3. Access Image Data from the 'im0' struct ---
im0_data = patientData.im0;
originalImg = im0_data.OriginalImage;
bwMask = im0_data.BW;
maskedImg = im0_data.maskedImage;
SchatzkerLabel = im0_data.label;

% --- 4. Visualize the Data (with conditional plotting) ---
figure('Name', ['Data for ', patientID]);

% Check if Coronal_CT exists to determine subplot layout
if isfield(patientData, 'Coronal_CT')
    numCols = 4;
    coronalCT_image = patientData.Coronal_CT;
else
    numCols = 3;
end

% Subplot 1: Original Image
subplot(1, numCols, 1);
imshow(originalImg, []);
title(['Schatzker Classification Label: ', num2str(SchatzkerLabel)]);

% Subplot 2: Tibia Mask (BW)
subplot(1, numCols, 2);
imshow(bwMask);
title('Tibia Bone Plateau Mask');

% Subplot 3: Masked Image
subplot(1, numCols, 3);
imshow(maskedImg, []);
title('Segmented Tibia Bone');

% Plot the optional fourth image only if it exists
if numCols == 4
    subplot(1, numCols, 4);
    imshow(coronalCT_image, []);
    title('A Coronal CT Slice');
end
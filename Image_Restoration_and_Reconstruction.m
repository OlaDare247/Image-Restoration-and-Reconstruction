
% Clear workspace
clear; clc; close all;
% 1. Generate Uniform Noise
% Let's create a 256x256 uniform noise matrix between 0 and 1
noise_uniform = rand(256, 256);
% 2. Display the Uniform Noise
figure, imagesc(noise_uniform), colormap gray, axis image, colorbar;
title('Uniform Noise Image');
% 3. Display the Histogram of the Noise
figure, histogram(noise_uniform(:), 50);
title('Histogram of Uniform Noise');
xlabel('Pixel Intensity');
ylabel('Frequency');
% 4. Select a Patch (e.g., 100x100 patch from top-left corner)
patch = noise_uniform(1:100, 1:100);
% 5. Calculate the Average (Mean) of the Patch
mean_patch = mean(patch(:));
% 6. Calculate the Standard Deviation of the Patch
std_patch = std(patch(:));
% Display the results in the MATLAB command window
fprintf('Mean of the patch: %.4f\n', mean_patch);
fprintf('Standard Deviation of the patch: %.4f\n', std_patch);
% Clear workspace and close figures
clear; clc; close all;
% 1. Load the grayscale image
A = imread('peppers_gray.png');
A = im2double(A); % Convert to double for calculations
% 2. Add Gaussian noise (mean = 0, variance = 0.01)
B = imnoise(A, 'gaussian', 0, 0.01);
% 3. Create averaging filters of sizes 3x3, 5x5, and 9x9
h3 = fspecial('average', [3 3]);
h5 = fspecial('average', [5 5]);
h9 = fspecial('average', [9 9]);
% 4. Apply the averaging filters
C3 = filter2(h3, B, 'same');
C5 = filter2(h5, B, 'same');
C9 = filter2(h9, B, 'same');
% 5. Display the denoised images
figure;
subplot(1,3,1), imagesc(C3), colormap gray, axis image, colorbar;
title('Averaged with 3x3 Filter');
subplot(1,3,2), imagesc(C5), colormap gray, axis image, colorbar;
title('Averaged with 5x5 Filter');
subplot(1,3,3), imagesc(C9), colormap gray, axis image, colorbar;
title('Averaged with 9x9 Filter');
% 6. Display histograms of the denoised images
figure;
subplot(1,3,1), histogram(C3(:), 50);
title('Histogram - 3x3 Filter');
subplot(1,3,2), histogram(C5(:), 50);
title('Histogram - 5x5 Filter');
subplot(1,3,3), histogram(C9(:), 50);
title('Histogram - 9x9 Filter');
% 7. Compute SNR between original (A) and restored images
SNR_3 = snr(C3, A - C3);
SNR_5 = snr(C5, A - C5);
SNR_9 = snr(C9, A - C9);
% 8. Display SNR results in command window
fprintf('SNR for 3x3 filter: %.2f dB\\n', SNR_3);
fprintf('SNR for 5x5 filter: %.2f dB\\n', SNR_5);
fprintf('SNR for 9x9 filter: %.2f dB\\n', SNR_9);
% Clear workspace and close figures
clear; clc; close all;
% Step 1: Load grayscale image and add Salt & Pepper noise

A = imread('peppers_gray.png');
A = im2double(A);
% Add Salt & Pepper Noise (density 0.2)
B = imnoise(A, 'salt & pepper', 0.2);
% Step 2: Apply Order Statistics Filters (min, max, median)
% Median Filter (already known)
C_median = medfilt2(B, [3 3]);
% Minimum Filter (min value in window)
C_min = ordfilt2(B, 1, true(3));
% Maximum Filter (max value in window)
C_max = ordfilt2(B, 9, true(3)); % 3x3 window has 9 elements
% Step 3: Apply Midpoint Filter
% Midpoint = (min + max)/2
C_midpoint = (C_min + C_max) / 2;
% Step 4: Display Restored Images
figure;
subplot(2,2,1), imagesc(C_median), colormap gray, axis image, colorbar;
title('Median Filtered Image (3x3)');
subplot(2,2,2), imagesc(C_min), colormap gray, axis image, colorbar;
title('Minimum Filtered Image (3x3)');
subplot(2,2,3), imagesc(C_max), colormap gray, axis image, colorbar;
title('Maximum Filtered Image (3x3)');
subplot(2,2,4), imagesc(C_midpoint), colormap gray, axis image, colorbar;
title('Midpoint Filtered Image (3x3)');
% Step 5: Compute SNR for each restored image
SNR_median = snr(C_median, A - C_median);
SNR_min = snr(C_min, A - C_min);
SNR_max = snr(C_max, A - C_max);
SNR_midpoint = snr(C_midpoint, A - C_midpoint);
% Step 6: Display SNR Results
fprintf('SNR for Median Filtered Image: %.2f dB\\n', SNR_median);
fprintf('SNR for Minimum Filtered Image: %.2f dB\\n', SNR_min);
fprintf('SNR for Maximum Filtered Image: %.2f dB\\n', SNR_max);
fprintf('SNR for Midpoint Filtered Image: %.2f dB\\n', SNR_midpoint);
% Modelling Atmospheric Turbulence
% Clear workspace
clear; clc; close all;
% Load and preprocess image
A = imread('157055.jpg');
A = rgb2gray(A);
A = im2double(A);
% FFT of original image
FA = fftshift(fft2(A));
% Create meshgrid for turbulence model
mid = floor(size(A)/2);
[y, x] = meshgrid(-mid(2):mid(2), -mid(1):mid(1));
r2 = x.^2 + y.^2;
% Different K values to test
K_values = [0.0025, 0.001, 0.0005];
for i = 1:length(K_values)
 
k = K_values(i);
 
% Step 1: Degradation function H
 
H = exp(-k * r2.^(5/6));
 
% Step 2: Apply degradation in frequency domain
 
HFA = H .* FA;
 
% Step 3: Inverse FFT to get degraded image
 
G = real(ifft2(ifftshift(HFA)));
 
% Show degraded image
 
figure, imagesc(G), colormap gray, axis image, colorbar;
 
title(sprintf('Degraded Image (K = %.4f)', k));
 
%% Step 4: Inverse Filtering (Direct)
 
% Avoid division by zero with a small epsilon
 
epsilon = 1e-4;
 
H_inv = H;


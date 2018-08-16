% Create the Lena matrix for CS evaluation

load brain512 %use same mask and sampling
load phantom_lego_4 %assume prime sized acquisition

[p, p] = size(image); %where p is prime
%Wavelets only do power of two, so remove extra row/col
image = image(1:p-1,1:p-1);%prime size to make it dyandic (assuming 257)
[rows, cols] = size(image);
fprintf('Image Size: %dx%d\n', rows, cols);

%load mask
%mask = logical(imread('sampling_fractal.png'));
%pdf = double(imread('sampling_fractal.png'));
%mask = logical(imread('sampling_random.png'));
%pdf = double(imread('sampling_random.png'));
mask = logical(imread('sampling_random_lines.png'));
pdf = double(imread('sampling_random_lines.png'));
[rows, cols] = size(mask);
fprintf('Mask Size: %dx%d\n', rows, cols);

%count lines used in CS
sample_count = sum(mask(:)~=0);
line_count = sum(mask(1,:));
fprintf('Sample Count: %d, Line Count: %d\n', sample_count, line_count);

%Fourier transform
image = fftshift(image); %shift
%mask = fftshift(mask);
%pdf = fftshift(pdf);
dftSpace = Cartesian_kData;
powerSpect = abs(dftSpace);
image = fftshift(image); %unshift

%overwrite brain kspace with lena
data = dftSpace(1:p-1,1:p-1); %prime size to make it dyandic (assuming 257)
[rows, cols] = size(data);
fprintf('DFT Size: %dx%d\n', rows, cols);

%save mat file
save('lego256.mat', 'image', 'mask', 'pdf', 'data');

figure;
subplot(1,3,1);
imshow(image, []);
colormap(gray); colorbar; caxis;
title('Image');
subplot(1,3,2);
imshow(mask, []);
colormap(gray); colorbar; caxis;
title('Mask');
subplot(1,3,3);
imshow(log(powerSpect), []);
colormap(gray); colorbar; caxis;
title('FFT');

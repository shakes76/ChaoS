% Create the Lena matrix for CS evaluation

load brain512 %use same mask and sampling

%image = imread('lena512.pgm');
%image = imread('sampling_lena.png');
image = imread('sampling_phantom.png');
%image = imread('sampling_camera.png');
%image = imread('sampling_lena_prime.png');
[rows, cols] = size(image);
fprintf('Image Size: %dx%d\n', rows, cols);

%load mask
%mask = logical(imread('sampling_fractal.png'));
%pdf = double(imread('sampling_fractal.png'));
mask = logical(imread('sampling_random.png'));
pdf = double(imread('sampling_random.png'));
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
dftSpace = fft2(image);
dftSpace = fftshift(dftSpace);
powerSpect = abs(dftSpace);
image = fftshift(image); %unshift

%add noise
noisyDFTSpace = awgn(dftSpace, 30);

%overwrite brain kspace with lena
data = noisyDFTSpace;

%save mat file
%save('lena512.mat', 'image', 'mask', 'pdf', 'data');
%save('lena256.mat', 'image', 'mask', 'pdf', 'data');
save('phantom256.mat', 'image', 'mask', 'pdf', 'data');
%save('camera256.mat', 'image', 'mask', 'pdf', 'data');

figure;
subplot(1,4,1);
imshow(image, []);
colormap(gray); colorbar; caxis;
title('Image');
subplot(1,4,2);
imshow(mask, []);
colormap(gray); colorbar; caxis;
title('Mask');
subplot(1,4,3);
imshow(log10(abs(noisyDFTSpace-dftSpace)), []);
colormap(gray); colorbar; caxis;
title('Noise');
subplot(1,4,4);
imshow(log(powerSpect), []);
colormap(gray); colorbar; caxis;
title('FFT');

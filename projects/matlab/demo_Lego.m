% This script demonstare a CS reconstruction from 
% Randomly undersmapled phase encodes of 2D FSE
% of a brain image.


addpath(strcat(pwd,'/utils'));

if exist('FWT2_PO') <2
	error('must have Wavelab installed and in the path');
end

load lego256;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L1 Recon Parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = size(data); 	% image Size
DN = size(data); 	% data Size
TVWeight = 0.002; 	% Weight for TV penalty, default: 0.002
xfmWeight = 0.005;	% Weight for Transform L1 penalty, default: 0.005
Itnlim = 8;		% Number of sub iterations, default: 8
iter = 48; %Iterations for main loop, default: 64
iterCount = 0;

%generate Fourier sampling operator
FT = p2DFT(mask, N, 1, 2);

% scale data
%im_dc = FT'*(data.*mask./pdf);
im_dc = FT'*(data.*mask);
data = data/max(abs(im_dc(:)));
im_dc = im_dc/max(abs(im_dc(:)));
image = double(image);
image = image/max(image(:));
%imageMax = max(image(:));

%generate transform operator
XFM = Wavelet('Daubechies',4,4);	% Wavelet

% initialize Parameters for reconstruction
param = init;
param.FT = FT;
param.XFM = XFM;
param.TV = TVOP;
param.data = data;
param.TVWeight =TVWeight;     % TV penalty 
param.xfmWeight = xfmWeight;  % L1 wavelet penalty
param.Itnlim = Itnlim;

figure(100), imshow(abs(im_dc),[]);drawnow;

res = XFM*im_dc;

errorList = [];
ssims = zeros(iter,1);
psnrs = zeros(iter,1);
iters = zeros(iter,1);

% do iterations
tic
for n=1:iter
	%res = fnlCg(res,param);
    [res, err] = fnlCgErr(res,param);
    errorList = [errorList; err];
    
	im_res = XFM'*res;
    recon = real(im_res);
    ssimval = ssim(recon, image);
    peaksnr = psnr(recon, image);
    iterCount = n*Itnlim;
    ssims(n) = ssimval;
    psnrs(n) = peaksnr;
    iters(n) = iterCount;
    disp(sprintf('%d   , ssim: %f, PSNR: %f, iteration: %d', n,ssimval,peaksnr,iterCount));
    
	figure(100), imshow(abs(im_res),[]), drawnow
end
toc

save('result_lego.mat', 'im_res', 'im_dc', 'errorList', 'ssims', 'psnrs', 'iters', 'N', 'TVWeight', 'xfmWeight', 'Itnlim', 'iter');
%save('result.mat', 'im_res', 'im_dc', 'errorList', 'N', 'TVWeight', 'xfmWeight', 'Itnlim', 'iter');

figure, imshow(image,[]);
figure, imshow(log10(abs(data.*mask)),[]);
figure, imshow(abs(cat(2,im_dc,im_res)),[]);
figure, imshow(abs(cat(2,im_dc(155:304,110:209), im_res(155:304,110:209))),[0,1],'InitialMagnification',200);
%figure, imshow(abs(cat(2,im_dc(165/2:314/2,120/2:219/2), im_res(165/2:314/2,120/2:219/2))),[0,1],'InitialMagnification',200);
title(' zf-w/dc              l_1 Wavelet');
figure, plot(1:length(errorList), errorList);





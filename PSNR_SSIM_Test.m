%clc; clear all;

%% Test Configuration
dir_name = 'D:/Test_Images_Location/Set5/x2/SR/'; % SR Location
dir_label = 'D:/Test_Images_Location/Set5/x2/HR/'; % HR Location
scale = 2; % Up-scale value (2,3,4)

%% PSNR/SSIM Test

images = dir(strcat(dir_name,'*.png'));
labels = dir(strcat(dir_label,'*.png'));

avg_psnr = 0;   
avg_ssim = 0;
psnr_list = zeros(length(images));
ssim_list = zeros(length(images));

for i = 1:length(images)
    img = rgb2ycbcr(im2double(uint8(imread(strcat(dir_name,images(i).name)))));
    lab = rgb2ycbcr(im2double(uint8(imread(strcat(dir_label,labels(i).name)))));
    
    [ih, iw, ic] = size(img);

    img2 = img(scale+1:ih-scale,scale+1:iw-scale,:);
    lab2 = lab(scale+1:ih-scale,scale+1:iw-scale,:);
    
    
    [psnrv, ~] = psnr(img2(:,:,1), lab2(:,:,1));
    [ssimv, ~] = ssim(img2(:,:,1), lab2(:,:,1));
    psnr_list(i) = psnrv;
    ssim_list(i) = ssimv;
    avg_psnr = avg_psnr + psnrv;
    avg_ssim = avg_ssim + ssimv;
    
end

disp('PSNR');
dispavgpsnr=avg_psnr/length(images);
disp(round(dispavgpsnr,2));
disp('SSIM')
dispavgssim=avg_ssim/length(images);
disp(round(dispavgssim,4));


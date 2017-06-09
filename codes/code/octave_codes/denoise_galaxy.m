%
%

clear
bb=8; % block size
RR=4; % redundancy factor
K=RR*bb^2; % number of atoms in the dictionary

sigma=0.005;
pathForImages = '/media/arunkannawadi/Acads_CMU/Cosmology/SparseMethods/kSVD/images/'; 
imageName = 'barbara.png'; %'pure_galaxy1.fits';
IMin0 = imread(strcat([pathForImages,imageName]));
%IMin0 = fitsread(strcat([pathForImages,imageName]));

% Set small values to 0.0
IMin0(IMin0<1e-7) = 0.0;

% Convert to grayscale
if (max(IMin0(:))<2)
    IMin = IMin0*255;
    a = 1
else
    IMin = double(IMin0);
end

%Add noise
IMnoise = IMin + sigma*randn(size(IMin));
PSNRIn = 20*log10(255/sqrt(mean((IMnoise(:)-IMin(:)).^2)));

%==========================================================================
%   P E R F O R M   D E N O I S I N G   U S I N G   A   D I C T  I O N A R Y
%                  T R A I N E D   O N   N O I S Y   I M A G E
%==========================================================================
[IoutAdaptive,output] = denoiseImageKSVD(IMnoise, sigma,K);

PSNROut = 20*log10(255/sqrt(mean((IoutAdaptive(:)-IMin(:)).^2)));
figure;
subplot(1,3,1); imshow(IMin,[]); title('Original clean image');
subplot(1,3,2); imshow(IMnoise,[]); title(strcat(['Noisy image, ',num2str(PSNRIn),'dB']));
subplot(1,3,3); imshow(IoutAdaptive,[]); title(strcat(['Clean Image by Adaptive dictionary, ',num2str(PSNROut),'dB']));

figure;
I = displayDictionaryElementsAsImage(output.D, floor(sqrt(K)), floor(size(output.D,2)/floor(sqrt(K))),bb,bb);
title('The dictionary trained on patches from the noisy image');



% Given a dictionary, denoise a noisy image

% clear

param.K = K;
param.numIteration = numIterOfKsvd ;

param.L = L;

param.errorFlag = 0; % decompose signals to have a fixed number of coefficients. irrespective of the error
param.preserveDCAtom = 0;
param.initialDictionary = DCT(:,1:param.K );
param.InitializationMethod =  'GivenMatrix';

pathForImages = '';
testImageName = 'pure_galaxy1.fits';

% load circular_dictionary;

testImage = fitsread(testImageName);
[NN1, NN2] = size(testImage);

% Add noise as a test part
sigma = 0.2*(max(testImage(:))-min(testImage(:)));
Image = imnoise(testImage,'gaussian',0.0,sigma);

%% function denoise()
% denoise the image using the resulted dictionary
bb = 16; % block size
RR = 4; % redundancy factor
K = RR*bb^2; % number of atoms in the dictionary
maxBlocksToConsider = 260000;
C = 1.15;
numIterOfKsvd = 5;
reduceDC = 1;
slidingDis = 1;

waitBarOn = 0; % not in Coma or any other clusters

errT = sigma*C;
IMout=zeros(NN1,NN2);
Weight=zeros(NN1,NN2);
%blocks = im2col(Image,[NN1,NN2],[bb,bb],'sliding');
tic
while (prod(floor((size(Image)-bb)/slidingDis)+1)>maxBlocksToConsider)
    slidingDis = slidingDis+1;
end
[blocks,idx] = my_im2col(Image,[bb,bb],slidingDis);
'While block executed'
toc

% go with jumps of 30000
for jj = 1:30000:size(blocks,2)
    tic
    if (waitBarOn)
        waitbar(((param.numIteration*size(blocks,2))+jj)/newCounterForWaitBar);
    end
    jumpSize = min(jj+30000-1,size(blocks,2));
    if (reduceDC)
        vecOfMeans = mean(blocks(:,jj:jumpSize));
        blocks(:,jj:jumpSize) = blocks(:,jj:jumpSize) - repmat(vecOfMeans,size(blocks,1),1);
    end
    
    %Coefs = mexOMPerrIterative(blocks(:,jj:jumpSize),Dictionary,errT);
    Coefs = OMPerr(Dictionary,blocks(:,jj:jumpSize),errT);
    if (reduceDC)
        blocks(:,jj:jumpSize)= Dictionary*Coefs + ones(size(blocks,1),1) * vecOfMeans;
    else
        blocks(:,jj:jumpSize)= Dictionary*Coefs ;
    end
    'One for loop evaluated'
    toc
end

count = 1;
Weight = zeros(NN1,NN2);
IMout = zeros(NN1,NN2);
[rows,cols] = ind2sub(size(Image)-bb+1,idx);
for i  = 1:length(cols)
    col = cols(i); row = rows(i);        
    block =reshape(blocks(:,count),[bb,bb]);
    IMout(row:row+bb-1,col:col+bb-1)=IMout(row:row+bb-1,col:col+bb-1)+block;
    Weight(row:row+bb-1,col:col+bb-1)=Weight(row:row+bb-1,col:col+bb-1)+ones(bb);
    count = count+1;
end;

if (waitBarOn)
    close(h);
end
IOut = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight);

save denoised_image
%% end % function denoise


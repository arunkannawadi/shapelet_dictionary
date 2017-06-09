% Dont know what to call it yet

clear

%% function [Dictionary, output] = TrainDictionary(pathForTrainImages, trainImageName)
bb = 16; % block size
RR = 4; % redundancy factor
K = RR*bb^2; % number of atoms in the dictionary
L = 4; % number of coefficients in an atom
maxNumBlocksToTrainOn = 65000;
C = 1.15;
numIterOfKsvd = 5;
reduceDC = 1;

pathForImages = '';
trainImageName = 'pure_galaxy1.fits';

trainImage = fitsread(strcat([pathForImages,trainImageName]));
[NN1,NN2] = size(trainImage);

if(prod([NN1,NN2]-bb+1)> maxNumBlocksToTrainOn)
    randPermutation =  randperm(prod([NN1,NN2]-bb+1));
    selectedBlocks = randPermutation(1:maxNumBlocksToTrainOn);

    blkMatrix = zeros(bb^2,maxNumBlocksToTrainOn);
    for i = 1:maxNumBlocksToTrainOn
        [row,col] = ind2sub(size(trainImage)-bb+1,selectedBlocks(i));
        currBlock = trainImage(row:row+bb-1,col:col+bb-1);
        blkMatrix(:,i) = currBlock(:);
    end
else
    blkMatrix = im2col(trainImage,[bb,bb],'sliding');
end

param.K = K;
param.numIteration = numIterOfKsvd ;

param.L = L;

param.errorFlag = 0; % decompose signals to have a fixed number of coefficients. irrespective of the error
param.preserveDCAtom = 0;

Pn=ceil(sqrt(K));
DCT=zeros(bb,Pn);
for k=0:1:Pn-1,
    V=cos([0:1:bb-1]'*k*pi/Pn);
    if k>0, V=V-mean(V); end;
    DCT(:,k+1)=V/norm(V);
end;
DCT=kron(DCT,DCT);

param.initialDictionary = DCT(:,1:param.K );
param.InitializationMethod =  'GivenMatrix';

if (reduceDC)
    vecOfMeans = mean(blkMatrix);
    blkMatrix = blkMatrix-ones(size(blkMatrix,1),1)*vecOfMeans;
end

[Dictionary,output] = KSVD(blkMatrix,param);
output.D = Dictionary;

%% end % function TrainDictionary

pathForImages = '';
trainImageName = 'pure_galaxy1.fits';

%[Dictionary, output] = TrainDictionary(pathForImages,trainImageName)
% Save and load the dictionary
dictionary_name = 'circular_dictionary'
save(dictionary_name, Dictionary)

%load(dictionary_name)

%% function denoise()
% denoise the image using the resulted dictionary
bb = 16; % block size
RR = 4; % redundancy factor
K = RR*bb^2; % number of atoms in the dictionary
maxNumBlocksToTrainOn = 65000;
C = 1.15;
numIterOfKsvd = 5;
reduceDC = 1;

errT = sigma*C;
IMout=zeros(NN1,NN2);
Weight=zeros(NN1,NN2);
%blocks = im2col(Image,[NN1,NN2],[bb,bb],'sliding');
while (prod(floor((size(Image)-bb)/slidingDis)+1)>maxBlocksToConsider)
    slidingDis = slidingDis+1;
end
[blocks,idx] = my_im2col(Image,[bb,bb],slidingDis);

if (waitBarOn)
    newCounterForWaitBar = (param.numIteration+1)*size(blocks,2);
end


% go with jumps of 30000
for jj = 1:30000:size(blocks,2)
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
%% end % function denoise




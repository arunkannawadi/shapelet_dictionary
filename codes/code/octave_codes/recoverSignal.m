% recover signals
function [Irec] = recoverSignal(output,bb,ImSize,vecOfMeans)
  Dict = output.D;
  Coefs = output.CoefMatrix;
  blkMatrix = Dict*Coefs;
  blkSel = [];
  NN1 = ImSize(1); NN2 = ImSize(2);
  for k=0:bb:NN1-1
      blkSel = [blkSel [1:bb:NN1]+(NN2-bb+1)*k];
  end
  outputBlk = blkMatrix;
  %outputBlk = bsxfun(@rdivide,blkMatrix,max(blkMatrix));
  Irec = col2im(outputBlk(:,blkSel),[bb,bb],[NN1,NN2],'distinct');
  
  
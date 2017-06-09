function error = ComputeError(path,prefix,D,S)
Files = dir(strcat([path,prefix,'*']));
Filenames = {Files.name};
N =size(Filenames,2);
signal = zeros(size(D,1),N);

for i=1:N
    Image = fitsread(strcat([path,Filenames{i}]));
    signal(:,i) = Image(:);
end

if prod(size(S))==1 % ugly!
    % then it is really L
    S = OMP(D,signal,S);
end

error = Error(signal, D,S);

function error = Error(signal,D,S)
% Compute the Frobenius norm between the original and reconstructed signal

N = prod(size(signal));
%error = sqrt(sum(sum((signal-D*S).^2))/N)/norm(signal,'fro');
error = norm(signal-D*S,'fro')/norm(signal,'fro');
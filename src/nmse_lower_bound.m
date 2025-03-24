function delta = nmse_lower_bound(N, M, K, C)
%UNTITLED Summary of this function goes here
%  M : list of number of measurements
delta = C*(K./M.*log(2.*N./K).*log(2.*N./M+2.*M./N)).^(1/5);
end

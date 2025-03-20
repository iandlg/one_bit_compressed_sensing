function [nmse,output_snr] = get_stats(x,xhat)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% mse = norm(x-xhat)^2;
nmse = norm(x/norm(x)-xhat/norm(xhat))^2;
output_snr = snr(x, x-xhat);
end

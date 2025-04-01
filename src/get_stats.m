function [nmse,output_snr, ham_err, ang_err] = get_stats(x,xhat,y, yhat)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% mse = norm(x-xhat)^2;

nmse = norm(x/norm(x)-xhat/norm(xhat))^2;
output_snr = snr(x, x-xhat);
ham_err = sum(y~=yhat)/length(y);
ang_err = acos(x'*xhat)/pi;
end

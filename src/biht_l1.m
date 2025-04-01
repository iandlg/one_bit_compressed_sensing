function [xhat,stats] = biht_l1(y, Phi, K, maxiter, htol)
    % This small matlab demo tests the Binary Iterative Hard Thresholding algorithm
    % developed in:
    %
    %  "Robust 1-bit CS via binary stable embeddings"
    %  L. Jacques, J. Laska, P. Boufounos, and R. Baraniuk
    %  
    % More precisely, using paper notations, two versions of BIHT are tested
    % here on sparse signal reconstruction:
    %
    %  * the standard BIHT associated to the (LASSO like) minimization of
    % 
    %        min || [ y o A(u) ]_- ||_1 s.t. ||u||_0 \leq K    (1)
    % Negative function [.]_-


    A = @(in) sgn(Phi*in);

    [M, N] = size(Phi);
    xhat = zeros(N,1);
    hd = Inf;

    tau = 1/(sqrt(M)*norm(Phi));
    
    ii=0;
    while(htol < hd)&&(ii < maxiter)
	    % Get gradient
	    g = Phi'*(A(xhat) - y);
	    
	    % Step
	    a = xhat - tau/2*g;
	    
	    % Best K-term (threshold)
	    [trash, aidx] = sort(abs(a), 'descend');
	    a(aidx(K+1:end)) = 0;
	    
        % Update x
	    xhat = a;
    
	    % Measure hammning distance to original 1bit measurements
	    hd = nnz(y - A(xhat));
	    ii = ii+1;
    end
    
    % Now project to sphere
    xhat = xhat/norm(xhat);
    names = ["iter", "hamming_err"];
    values = [ii, nnz(y - A(xhat))];
    stats = dictionary(names, values);
end

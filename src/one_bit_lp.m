function x_opt = one_bit_lp(y, A, ~)
    [m, n] = size(A);
    D = zeros(n-1, n);
    for i = 1:n-1
        D(i, i) = 1;
        D(i, i+1) = -1;
    end
    lambda = 1;
    cvx_clear
    cvx_begin quiet
        
       variable x(n)
           
       minimize (norm(x, 1) + lambda * norm(D * x, 1)) 
       subject to 
           y.*(A*x) >= 0;
           sum(y.*(A*x))/m >= 1;

    cvx_end

    x_opt = x/norm(x, 2);

end

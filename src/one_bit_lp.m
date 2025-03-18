function x_opt = one_bit_lp(y, A)

    [m,n] = size(A);

    cvx_begin quiet
    
       variable (x(n))
           
       minimize (norm(x, 1))
       subject to 
           y.*(A*x) >= 0;
           y.*(A*x)/m >= 1;

    cvx_end

    x_opt = x/norm(x, 2);

end

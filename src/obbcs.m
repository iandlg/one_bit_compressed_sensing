function [x_est, stats] =obbcs(t,A,maxiter,tor)
    % This file implements the algorithm discribed in 
    % "F. Li, J. Fang, H. Li, et al. Robust One-Bit Bayesian Compressed
    % Sensing with Sign-Flip Errors. Signal Processing Letters, IEEE, 2015,
    % 22(7):857–861"
    % Model : t=sign(A*x);
    % Inputs:
    % t --- measurements 
    % A---measurement matrix 
    % max_iter --- maximal iteration 
    % tor  ---convergence conddition 
    % Outputs: 
    % x_est --- normalized estimate signal 
    
    % init 
    [M,N]=size(A);
    par=1e-10;
    a=par;
    b=par;
    at=a+.5;
    Ealpha=ones(N,1)*par;
    iter=0;
    mux_old=zeros(N,1);
    mux=ones(N,1);
    epsilong=ones(M,1);
    while( iter < maxiter  && norm(mux_old-mux) > tor )
       iter = iter +1 ;
       mux_old=mux;
       
       % Expectation step 
       % Update mux & sigmax 
        sigmax=inv(diag(Ealpha)+2*A'*diag(f_lambda(epsilong))*A);
        mux=.5*sigmax*A'*(2*t-1);
        
        % Update Ealpha
        bt=b+.5*(diag(sigmax)+mux.*mux);
        Ealpha=at./bt;
        
        % Max Step 
        % Update epsilong 
        B=mux*mux'+sigmax;
        epsilong=sqrt(diag(A*B*A'));
    end
    stats = dictionary("iter", iter);
    x_est=mux/norm(mux);
end



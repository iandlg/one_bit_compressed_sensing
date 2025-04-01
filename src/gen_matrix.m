function W = gen_matrix(rows,cols)
%Generate a rows by cols random sensing matrix with unit norm columns
    W = randn(rows, cols);  % Draw from standard normal distribution
    W = W ./ vecnorm(W, 2, 1);  % Normalize columns to unit L2 norm
end

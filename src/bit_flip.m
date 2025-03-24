function y_flipped = bit_flip(y,k_flips)
    %flips k elements in y
    %   Detailed explanation goes here
    y_flipped = y;
    [M,~] = size(y);
    ind_flip=randperm(M);
    flip_loc=ind_flip(1:k_flips);
    y_flipped(flip_loc)=~(y_flipped(flip_loc));
end

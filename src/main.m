%% import MNIST dataset
clc; clear; close all;
load('../data/mnist.mat')

% image (training.images(:,:,18)*255);
[height, width, image_num] = size(training.images);
training.labels(18);

pos = @(in) in.*(in > 0); 

%% Flatten and normalize images
processed_images = reshape(training.images, height*width, image_num);
% norm = mean(processed_images,'all');
% std = std(processed_images,[], 'all');
im_norm = norm(processed_images);
% processed_images = processed_images/im_norm;

%% Create sensing matrices
% List of row sizes for the projection matrices
cols = height*width;  % Image vector dimension
MNratios = linspace(0,3,10);
rows_list = floor(cols * MNratios);
rows_list(1) = 50; 

% Generate random matrices and normalize columns
projection_matrices = cell(1, length(rows_list));

for i = 1:length(rows_list)
    rows = rows_list(i);
    W = randn(rows, cols);  % Draw from standard normal distribution
    W = W ./ vecnorm(W, 2, 1);  % Normalize columns to unit L2 norm
    projection_matrices{i} = W;
end

%% Test on random vector
% Macros 
M=100;
N=50;
K=5;
L=3;
maxiter=300;
htol = 0;
tor = 1e-6;

[y,Phi,x]=gen_data(M,N,K,L);


[x_obbcs, ~] = obbcs(y, Phi,maxiter,tor);
[x_biht, ~] = biht_l1(y, Phi, K, maxiter, htol);
x_oblp = one_bit_lp(y, Phi);

% Plot
figure(1); clf;
stem(x);hold on;
stem(x_biht);
stem(x_obbcs);
stem(x_oblp)
grid on; hold off

%% Test on single image
idx = 18;
x = processed_images(:,idx);
K = nnz(x);
snr_obbcs = zeros(1, length(rows_list));
snr_biht = zeros(1, length(rows_list));

figure(2); clf;
% subplot(2,10,1);
% imshow(reshape(x,height, width));
% plot(x);

for i = 1:length(rows_list)
    disp(['Processing sensing matrix ', num2str(i),'/', num2str(length(rows_list))]);
    % Get measurement
    Phi = projection_matrices{i};

    % Signal reconstruction
    [x_biht, ~] = biht_l1(sgn(Phi*x), Phi, K, maxiter, htol);
    % [x_obbcs, ~] = obbcs(sgn(Phi*x), Phi, maxiter, tor);
    x_biht = rescale(pos(x_biht)); % x_obbcs = rescale(pos(x_obbcs));

    % Plot
    subplot(2,10,i);
    imshow(reshape(x_biht,height,width))
    subplot(2,10,10 + i);
    imshow(reshape(x_obbcs,height,width))

    % Collect metrics
    snr_biht(i) = snr(x, x-x_biht);
    snr_obbcs(i) = snr(x, x-x_obbcs);
    % plot(rescale(xhat, 0,1))
end
%%
% Plot SNR
figure(3); clf;
plot(rows_list, snr_obbcs); hold on;
plot(rows_list, snr_biht);
legend("OBBCS", "BIHT");

%% Test on multiple images
N = 10;
snr_obbcs = zeros(1, length(rows_list));
snr_biht = zeros(1, length(rows_list));

for i = 1:length(rows_list)
    snr_biht_list = zeros(1,N);
    snr_obbcs_list = zeros(1,N);

    disp(['Processing sensing matrix ', num2str(i),'/', num2str(length(rows_list))]);

    for j = 1:N
        % Get measurement
        Phi = projection_matrices{i};
        x = processed_images(:,j);
        K = nnz(x);

        % Signal reconstruction
        [x_biht, ~] = biht_l1(sgn(Phi*x), Phi, K, maxiter, htol);  % sgn sets to 0 1
        % [x_obbcs, ~] = obbcs(sgn(Phi*x), Phi, maxiter, tor);        % sgn sets to 0 1
        x_biht = rescale(pos(x_biht)); % x_obbcs = rescale(pos(x_obbcs));

        % Plot
        subplot(10,10,j+10*(i-1));
        imshow(reshape(x_biht,height,width))

    
        % Collect metrics
        snr_biht_list(j) = snr(x, x-x_biht);
        disp(snr_biht_list(j))
        % snr_obbcs_list(j) = snr(x, x-x_obbcs);
        % plot(rescale(xhat, 0,1))
    end
    snr_biht(i) = mean(snr_biht_list);
    snr_obbcs(i) = mean(snr_obbcs_list);


    
end

%%
% Plot average SNRs
figure(4); clf;
load("snr_obbcs.mat","snr_obbcs")
plot(MNratios, snr_obbcs); hold on;
plot(MNratios, snr_biht);
xlabel("M/N ratios");
ylabel("SNR (dB)")
legend("OBBCS", "BIHT");
grid on;
exportgraphics(gcf, "../output/snr_to_ratios.png", "Resolution",300);
hold off;


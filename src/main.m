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
MNratios = linspace(0,2,10);
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
N=51;
K=5;
L=3;
maxiter=300;
htol = 0;
tor = 1e-6;

[y,Phi,x]=gen_data(M,N,K,L);


[obbcs_dat.xhat, ~] = obbcs(y, Phi,maxiter,tor);
[biht_dat.xhat, ~] = biht_l1(y, Phi, K, maxiter, htol);
obpl_dat.xhat = one_bit_lp(y, Phi);

[obbcs_dat.nmse, obbcs_dat.snr] = get_stats(x, obbcs_dat.xhat);
[biht_dat.nmse, biht_dat.snr] = get_stats(x, biht_dat.xhat);
[obpl_dat.nmse, obpl_dat.snr] = get_stats(x, obpl_dat.xhat);


% Plot
figure(1); clf;
stem(x);hold on;
stem(obbcs_dat.xhat);
stem(biht_dat.xhat);
stem(obpl_dat.xhat)
grid on; hold off

disp(['OBBCS : NMSE = ', num2str(obbcs_dat.nmse), ''])

%% Test on single image
idx = 18;
x = processed_images(:,idx);
K = nnz(x);
maxiter=300;
htol = 0;
tor = 1e-6;

% obbcs.xhat = zeros(1, height*width);
obbcs_dat.snr = zeros(1, length(rows_list));
obbcs_dat.nmse = zeros(1, length(rows_list));

% biht.xhat = zeros(1, height*width);
biht_dat.snr = zeros(1, length(rows_list));
biht_dat.nsme = zeros(1, length(rows_list));

% obpl.xhat = zeros(1, height*width);
obpl.snr = zeros(1, length(rows_list));
obpl.nmse = zeros(1, length(rows_list));

figure(2); clf;
% subplot(2,10,1);
% imshow(reshape(x,height, width));
% plot(x);

for i = 1:length(rows_list)
    disp(['Processing sensing matrix ', num2str(i),'/', num2str(length(rows_list))]);
    % Get measurement
    Phi = projection_matrices{i};
    y = sgn(Phi*x);

    % Signal reconstruction
    [biht_dat.xhat, ~] = biht_l1(y, Phi, K, maxiter, htol);
    [obbcs_dat.xhat, ~] = obbcs(y, Phi, maxiter, tor);
    obpl_dat.xhat = one_bit_lp(y, Phi);
    
    % Rescale
    biht_dat.xhat = rescale(pos(biht_dat.xhat));
    obbcs_dat.xhat = rescale(pos(obbcs_dat.xhat));
    obpl_dat.xhat = rescale(pos(obpl_dat.xhat));

    % Plot
    subplot(3,10,i);
    imshow(reshape(biht_dat.xhat,height,width))
    subplot(3,10,10 + i);
    imshow(reshape(obbcs_dat.xhat,height,width))
    subplot(3,10,20+i);
    imshow(reshape(obpl_dat.xhat, height, width))

    % Collect metrics
    [biht_dat.nmse(i), biht_dat.snr(i)] = get_stats(x, biht_dat.xhat);
    [obbcs_dat.nmse(i), obbcs_dat.snr(i)] = get_stats(x, obbcs_dat.xhat);
    [obpl_dat.nmse(i), obpl_dat.snr(i)] = get_stats(x, obpl_dat.xhat);
    % plot(rescale(xhat, 0,1))
end
%%
% Plot SNR
figure(3); clf;
plot(MNratios, obbcs_dat.snr); hold on;
plot(MNratios, biht_dat.snr);
plot(MNratios, obpl_dat.snr);
xlabel("MN ratios")
ylabel("SNR (dB)") 
legend("OBBCS", "BIHT","OBPL");
exportgraphics(gcf, "../output/snr_to_ratios_1img.png", "Resolution",300);
hold off;


% Plot NMSE
figure(4); clf;
plot(MNratios, obbcs_dat.nmse); hold on;
plot(MNratios, biht_dat.nmse);
plot(MNratios, obpl_dat.nmse);
xlabel("MN ratios")
ylabel("NMSE") 
legend("OBBCS", "BIHT","OBPL");
exportgraphics(gcf, "../output/nmse_to_ratios_1img.png", "Resolution",300);
hold off;


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


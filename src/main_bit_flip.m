%% import MNIST dataset
clc; clear; close all;

% Put the mnist.mat file in the ../data directory
data_dir = "../data";
mnsit_dir = fullfile(data_dir, "mnist.mat");
load(mnsit_dir)

output_dir = "../out";
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% image (training.images(:,:,18)*255);
[height, width, image_num] = size(training.images);
training.labels(18);

pos = @(in) in.*(in > 0); 

%% Flatten and normalize images
processed_images = reshape(training.images, height*width, image_num);
im_norm = vecnorm(processed_images,2,1);
processed_images = processed_images./im_norm;

%% Test on random vector with bitflip
% Macros 
M=100;
N=51;
K=5;
L=3;
k_flip = 3;
maxiter=300;
htol = 0;
tor = 1e-6;

[y,Phi,x]=gen_data(M,N,K,L);
y = bit_flip(y,k_flip);              % perform bit flip

[obbcs_dat.xhat, ~] = obbcs(y, Phi,maxiter,tor);
[biht_dat.xhat, ~] = biht_l1(y, Phi, K, maxiter, htol);
oblp_dat.xhat = one_bit_lp(y, Phi, 1);
obbp_dat.xhat = one_bit_bp(y, Phi, 1);

[obbcs_dat.nmse, obbcs_dat.snr] = get_stats(x, obbcs_dat.xhat);
[biht_dat.nmse, biht_dat.snr] = get_stats(x, biht_dat.xhat);
[oblp_dat.nmse, oblp_dat.snr] = get_stats(x, oblp_dat.xhat);
[obbp_dat.nmse, obbp_dat.snr] = get_stats(x, obbp_dat.xhat);


% Plot
figure(1); clf;
stem(x);hold on;
stem(obbcs_dat.xhat);
stem(biht_dat.xhat);
stem(oblp_dat.xhat)
stem(obbp_dat.xhat)
legend("original",['OBBCS : NMSE = ', num2str(obbcs_dat.nmse,3), ''] ...
    ,['BIHT : NMSE = ', num2str(biht_dat.nmse,3), ''], ...
    ['OBLP : NMSE = ', num2str(oblp_dat.nmse,3), ''], ...
    ['OBBP : NMSE = ', num2str(obbp_dat.nmse,3), ''])
grid on;
title(["Comparison of algorithm perfomance on random sparse vector with", num2str(k_flip), "bit flips."])
output_file_path = fullfile(output_dir, "random_vector_test_bitflip.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

disp(['OBBCS : NMSE = ', num2str(obbcs_dat.nmse), ''])
disp(['BIHT : NMSE = ', num2str(biht_dat.nmse), ''])
disp(['OBLP : NMSE = ', num2str(oblp_dat.nmse), ''])
disp(['OBBP : NMSE = ', num2str(obbp_dat.nmse), ''])

%% Create the sensing matrice + bit flip numbers
% List of row sizes for the projection matrices
cols = height*width;  % Image vector dimension
measurements = floor(1.1*cols); % 

% make the measuremet matrix 
Phi = randn(measurements, cols);
Phi = Phi./vecnorm(Phi, 2 ,1);

% Generate the bit flip array
% flips from 0-30% of measurements in 8 steps
flips = linspace(0,0.2,8);
flips = floor(flips*measurements);
%% Test on single image with bitflip
idx = 18;
x = processed_images(:,idx);
K = nnz(x);
maxiter=300;
htol = 0;
tor = 1e-6;

obbcs_dat.snr = zeros(1, length(flips));
obbcs_dat.nmse = zeros(1, length(flips));

biht_dat.snr = zeros(1, length(flips));
biht_dat.nsme = zeros(1, length(flips));

oblp_dat.snr = zeros(1, length(flips));
oblp_dat.nmse = zeros(1, length(flips));

obbp_dat.snr = zeros(1, length(flips));
obbp_dat.nmse = zeros(1, length(flips));

figure(2); clf; hold on;
% subplot(2,10,1);
% imshow(reshape(x,height, width));
% plot(x);

for i = 1:length(flips)
    disp([num2str(i),'/',num2str(length(flips)), ...
        ' - Processing reconstruction with ', num2str(flips(i)),' bit flips']);
    % Get measurement
    y = sgn(Phi*x);
    
    % Flip bits
    y_flip = bit_flip(y, flips(i));

    % Signal reconstruction
    [biht_dat.xhat, ~] = biht_l1(y_flip, Phi, K, maxiter, htol);
    [obbcs_dat.xhat, ~] = obbcs(y_flip, Phi, maxiter, tor);
    oblp_dat.xhat = one_bit_lp(y_flip, Phi);
    obbp_dat.xhat = one_bit_bp(y_flip, Phi);
    
    % Rescale
    biht_dat.xhat = pos(biht_dat.xhat);
    obbcs_dat.xhat =pos(obbcs_dat.xhat);
    oblp_dat.xhat = pos(oblp_dat.xhat);
    obbp_dat.xhat = pos(obbp_dat.xhat);

    % Plot
    subplot(4,10,i); 
    imshow(reshape(rescale(biht_dat.xhat),height,width))
    title('BIHT', 'FontSize', 8, 'FontWeight', 'normal', 'HorizontalAlignment', 'center');
    subplot(4,10,10 + i); 
    imshow(reshape(rescale(obbcs_dat.xhat),height,width))
    title('OBBCS', 'FontSize', 8, 'FontWeight', 'normal', 'HorizontalAlignment', 'center');
    subplot(4,10,20+i);
    imshow(reshape(rescale(oblp_dat.xhat), height, width))
    title('OBLP', 'FontSize', 8, 'FontWeight', 'normal', 'HorizontalAlignment', 'center');
    subplot(4,10,30+i); 
    imshow(reshape(rescale(obbp_dat.xhat), height, width))
    title('OBBP', 'FontSize', 8, 'FontWeight', 'normal', 'HorizontalAlignment', 'center');
    
    % Collect metrics
    [biht_dat.nmse(i), biht_dat.snr(i)] = get_stats(x, biht_dat.xhat);
    [obbcs_dat.nmse(i), obbcs_dat.snr(i)] = get_stats(x, obbcs_dat.xhat);
    [oblp_dat.nmse(i), oblp_dat.snr(i)] = get_stats(x, oblp_dat.xhat);
    [obbp_dat.nmse(i), obbp_dat.snr(i)] = get_stats(x, obbp_dat.xhat);
end

output_file_path = fullfile(output_dir, "image_reconstruction_comparison_bitflip.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

%% Plot snr and nmse for single image
% Plot SNR
figure(3); clf;
plot(flips/measurements*100, obbcs_dat.snr); hold on;
plot(flips/measurements*100, biht_dat.snr);
plot(flips/measurements*100, oblp_dat.snr);
plot(flips/measurements*100, obbp_dat.snr);
xlabel("Percentage of flipped bits")
ylabel("SNR (dB)") 
legend("OBBCS", "BIHT","OBLP", "OBBP");
grid on;
output_file_path = fullfile(output_dir, "snr_to_ratios_1img_bitflip.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;


% Plot NMSE
figure(4); clf;
plot(flips/measurements*100, obbcs_dat.nmse); hold on;
plot(flips/measurements*100, biht_dat.nmse);
plot(flips/measurements*100, oblp_dat.nmse);
plot(flips/measurements*100, obbp_dat.nmse);
xlabel("MN ratios")
ylabel("NMSE") 
legend("OBBCS", "BIHT","OBLP", "OBBP", "NMSE upper bound");
grid('on')
output_file_path = fullfile(output_dir, "nmse_to_ratios_1img_bitflip.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

%% Test on multiple images bitflip
N = 4;

obbcs_dat.snr = zeros(1, length(flips));
obbcs_dat.nmse = zeros(1, length(flips));

biht_dat.snr = zeros(1, length(flips));
biht_dat.nsme = zeros(1, length(flips));

oblp_dat.snr = zeros(1, length(flips));
oblp_dat.nmse = zeros(1, length(flips));

obbp_dat.snr = zeros(1, length(flips));
obbp_dat.nmse = zeros(1, length(flips));

for i = 1:length(flips)
    snr_biht_list = zeros(1,N);
    snr_obbcs_list = zeros(1,N);
    snr_oblp_list = zeros(1,N);
    snr_obbp_list = zeros(1,N);

    nmse_biht_list = zeros(1,N);
    nmse_obbcs_list = zeros(1,N);
    nmse_oblp_list = zeros(1,N);
    nmse_obbp_list = zeros(1,N);

    disp([num2str(i),'/',num2str(length(flips)), ...
        ' - Processing reconstruction with ', num2str(flips(i)),' bit flips']);

    for j = 1:N
        % Get measurement
        x = processed_images(:,j);
        y = sgn(Phi*x);
        y_flip = bit_flip(y, flips(i));
        K = nnz(x);

        % Signal reconstruction
        [biht_dat.xhat, ~] = biht_l1(y_flip, Phi, K, maxiter, htol);
        [obbcs_dat.xhat, ~] = obbcs(y_flip, Phi, maxiter, tor);
        oblp_dat.xhat = one_bit_lp(y_flip, Phi);
        obbp_dat.xhat = one_bit_bp(y_flip, Phi);
        
        % Rescale
        biht_dat.xhat = pos(biht_dat.xhat);
        obbcs_dat.xhat = pos(obbcs_dat.xhat);
        oblp_dat.xhat = pos(oblp_dat.xhat);
        obbp_dat.xhat = pos(obbp_dat.xhat);
    
        % Collect metrics
        [nmse_biht_list(i), snr_biht_list(i)] = get_stats(x, biht_dat.xhat);
        [nmse_obbcs_list(i), snr_obbcs_list(i)] = get_stats(x, obbcs_dat.xhat);
        [nmse_oblp_list(i), snr_oblp_list(i)] = get_stats(x, oblp_dat.xhat);
        [nmse_obbp_list(i), snr_obbp_list(i)] = get_stats(x, obbp_dat.xhat);
    end

    % Average out metrics
    obbcs_dat.snr(i) = mean(snr_obbcs_list);
    biht_dat.snr(i) = mean(snr_biht_list);
    oblp_dat.snr(i) = mean(snr_oblp_list);
    obbp_dat.snr(i) = mean(snr_obbp_list);

    obbcs_dat.nmse(i) = mean(nmse_obbcs_list);
    biht_dat.nmse(i)= mean(nmse_biht_list);
    oblp_dat.nmse(i) = mean(nmse_oblp_list);
    obbp_dat.nmse(i) = mean(nmse_obbp_list);

end

output_file_path = fullfile(output_dir, 'average_metrics_bitflip.mat');
save(output_file_path, "obbcs_dat", "biht_dat", "oblp_dat", "obbp_dat");

%% plot average snr and nmse for image
% Plot average SNRs
output_file_path = fullfile(output_dir, 'average_metrics_bitflip.mat');
load(output_file_path);

figure(4); clf;
plot(flips/measurements*100, obbcs_dat.snr); hold on;
plot(flips/measurements*100, biht_dat.snr);
plot(flips/measurements*100, oblp_dat.snr);
plot(flips/measurements*100, obbp_dat.snr);
xlabel("Percentage of flipped bits");
ylabel("SNR (dB)")
legend("OBBCS", "BIHT","OBLP", "OBBP");
grid on;
output_file_path = fullfile(output_dir, "avg_snr_to_ratios_bitflip.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

% Plot average NMSEs
figure(5); clf;
plot(flips/measurements*100, obbcs_dat.nmse); hold on;
plot(flips/measurements*100, biht_dat.nmse);
plot(flips/measurements*100, oblp_dat.nmse);
plot(flips/measurements*100, obbp_dat.nmse);
xlabel("Percentage of flipped bits");
ylabel("NMSE")
legend("OBBCS", "BIHT","OBLP", "OBBP");
grid on;
output_file_path = fullfile(output_dir, "avg_nmse_to_ratios_bitflip.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

%% import MNIST dataset
clc; clear; close all;

% Put the mnist.mat file in the ../data directory
data_dir = fullfile(".." , "data");
mnsit_dir = fullfile(data_dir, "mnist.mat");
load(mnsit_dir)

output_dir = fullfile("..", "out");
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

[obbcs_dat.nmse, obbcs_dat.snr, obbcs_dat.hamerr, obbcs_dat.angerr] = ...
    get_stats(x, obbcs_dat.xhat, y, sgn(Phi*obbcs_dat.xhat));
[biht_dat.nmse, biht_dat.snr, biht_dat.hamerr, biht_dat.angerr] = ...
    get_stats(x, biht_dat.xhat, y , sgn(Phi*biht_dat.xhat));
[oblp_dat.nmse, oblp_dat.snr, oblp_dat.hamerr, oblp_dat.angerr] = ...
    get_stats(x, oblp_dat.xhat, y, sgn(Phi*oblp_dat.xhat));
[obbp_dat.nmse, obbp_dat.snr, obbp_dat.hamerr, obbp_dat.angerr] = ...
    get_stats(x, obbp_dat.xhat, y, sgn(Phi*obbp_dat.xhat));


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
measurements = floor(1.5*cols); % 

% Generate the bit flip array
% flips from 0-30% of measurements in 8 steps
steps = 10;
flips = linspace(0,0.2,steps);
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
obbcs_dat.hamerr = zeros(1, length(flips));
obbcs_dat.angerr = zeros(1, length(flips));

biht_dat.snr = zeros(1, length(flips));
biht_dat.nsme = zeros(1, length(flips));
biht_dat.hamerr = zeros(1, length(flips));
biht_dat.angerr = zeros(1, length(flips));

oblp_dat.snr = zeros(1, length(flips));
oblp_dat.nmse = zeros(1, length(flips));
oblp_dat.hamerr = zeros(1, length(flips));
oblp_dat.angerr = zeros(1, length(flips));

obbp_dat.snr = zeros(1, length(flips));
obbp_dat.nmse = zeros(1, length(flips));
obbp_dat.hamerr = zeros(1, length(flips));
obbp_dat.angerr = zeros(1, length(flips));

figure(2); clf; hold on;
% subplot(2,10,1);
% imshow(reshape(x,height, width));
% plot(x);

for i = 1:length(flips)
    disp([num2str(i),'/',num2str(length(flips)), ...
        ' - Processing reconstruction with ', num2str(flips(i)),' bit flips']);

    % Get measurement matrix
    Phi = gen_matrix(measurements, cols);

    % Get measurement
    y = sgn(Phi*x);
    
    % Flip bits
    y_flip = bit_flip(y, flips(i));

    % Signal reconstruction
    [biht_dat.xhat, ~] = biht_l1(y_flip, Phi, K, maxiter, htol);
    [obbcs_dat.xhat, ~] = obbcs(y_flip, Phi, maxiter, tor);
    oblp_dat.xhat = one_bit_lp(y_flip, Phi);
    obbp_dat.xhat = one_bit_bp(y_flip, Phi);
    
    % % Rescale
    % biht_dat.xhat = pos(biht_dat.xhat);
    % obbcs_dat.xhat = pos(obbcs_dat.xhat);
    % oblp_dat.xhat = pos(oblp_dat.xhat);
    % obbp_dat.xhat = pos(obbp_dat.xhat);

    % Plot
    subplot(4,steps,i); 
    imshow(reshape(rescale(biht_dat.xhat),height,width))
    title('BIHT', 'FontSize', 8, 'FontWeight', 'normal', 'HorizontalAlignment', 'center');
    subplot(4,steps,steps + i); 
    imshow(reshape(rescale(obbcs_dat.xhat),height,width))
    title('OBBCS', 'FontSize', 8, 'FontWeight', 'normal', 'HorizontalAlignment', 'center');
    subplot(4,steps,2*steps+i);
    imshow(reshape(rescale(oblp_dat.xhat), height, width))
    title('OBLP', 'FontSize', 8, 'FontWeight', 'normal', 'HorizontalAlignment', 'center');
    subplot(4,steps,3*steps+i); 
    imshow(reshape(rescale(obbp_dat.xhat), height, width))
    title('OBBP', 'FontSize', 8, 'FontWeight', 'normal', 'HorizontalAlignment', 'center');
    
    % Collect metrics
    [biht_dat.nmse(i), biht_dat.snr(i), biht_dat.hamerr(i), biht_dat.angerr(i)] = ...
        get_stats(x, biht_dat.xhat, y, sgn(Phi*biht_dat.xhat));
    [obbcs_dat.nmse(i), obbcs_dat.snr(i), obbcs_dat.hamerr(i), obbcs_dat.angerr(i)] = ...
        get_stats(x, obbcs_dat.xhat, y, sgn(Phi*obbcs_dat.xhat));
    [oblp_dat.nmse(i), oblp_dat.snr(i), oblp_dat.hamerr(i), oblp_dat.angerr(i)] = ...
        get_stats(x, oblp_dat.xhat, y, sgn(Phi*oblp_dat.xhat));
    [obbp_dat.nmse(i), obbp_dat.snr(i), obbp_dat.hamerr(i), obbp_dat.angerr(i)] = ...
        get_stats(x, obbp_dat.xhat, y, sgn(Phi*obbp_dat.xhat));
end

output_file_path = fullfile(output_dir, "image_reconstruction_comparison_bitflip.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

%% Plot snr and nmse for single image
% Plot SNR
figure(3); clf;
plot(flips/measurements*100, obbcs_dat.snr, LineWidth=1.2, Marker="+"); hold on;
plot(flips/measurements*100, biht_dat.snr, LineWidth=1.2, Marker="o");
plot(flips/measurements*100, oblp_dat.snr, LineWidth=1.2, Marker="*");
plot(flips/measurements*100, obbp_dat.snr, LineWidth=1.2, Marker="diamond");
xlabel("Percentage of flipped bits")
ylabel("SNR (dB)") 
legend("OBBCS", "BIHT","OBLP", "OBBP","Location","best");
grid on;
output_file_path = fullfile(output_dir, "1img_snr_to_flips.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;


% Plot NMSE
figure(4); clf;
plot(flips/measurements*100, obbcs_dat.nmse, LineWidth=1.2, Marker="+"); hold on;
plot(flips/measurements*100, biht_dat.nmse, LineWidth=1.2, Marker="o");
plot(flips/measurements*100, oblp_dat.nmse, LineWidth=1.2, Marker="*");
plot(flips/measurements*100, obbp_dat.nmse, LineWidth=1.2, Marker="diamond");
xlabel("Percentage of flipped bits")
ylabel("NMSE") 
legend("OBBCS", "BIHT","OBLP", "OBBP","Location","best");
grid('on')
output_file_path = fullfile(output_dir, "1img_nmse_to_flips.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

% Plot normalized Hamming error
figure(5); clf;
plot(flips/measurements*100, obbcs_dat.hamerr, LineWidth=1.2, Marker="+"); hold on;
plot(flips/measurements*100, biht_dat.hamerr, LineWidth=1.2, Marker="o");
plot(flips/measurements*100, oblp_dat.hamerr, LineWidth=1.2, Marker="*");
plot(flips/measurements*100, obbp_dat.hamerr, LineWidth=1.2, Marker="diamond");
xlabel("Percentage of flipped bits")
ylabel("Normalized Hamming error") 
legend("OBBCS", "BIHT","OBLP", "OBBP","Location","best");
grid('on')
output_file_path = fullfile(output_dir, "1img_hamming_err_to_flips.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

% Plot normalized angular error
figure(6); clf;
plot(flips/measurements*100, obbcs_dat.angerr, LineWidth=1.2, Marker="+"); hold on;
plot(flips/measurements*100, biht_dat.angerr, LineWidth=1.2, Marker="o");
plot(flips/measurements*100, oblp_dat.angerr, LineWidth=1.2, Marker="*");
plot(flips/measurements*100, obbp_dat.angerr, LineWidth=1.2, Marker="diamond");
xlabel("Percentage of flipped bits")
ylabel("Normalized angular error") 
legend("OBBCS", "BIHT","OBLP", "OBBP","Location","best");
grid('on')
output_file_path = fullfile(output_dir, "1img_angular_err_to_flips.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

%% Test on multiple images bitflip
N = 4;

obbcs_dat.snr = zeros(1, length(flips));
obbcs_dat.nmse = zeros(1, length(flips));
obbcs_dat.hamerr = zeros(1, length(flips));
obbcs_dat.angerr = zeros(1, length(flips));

biht_dat.snr = zeros(1, length(flips));
biht_dat.nsme = zeros(1, length(flips));
biht_dat.hamerr = zeros(1, length(flips));
biht_dat.angerr = zeros(1, length(flips));

oblp_dat.snr = zeros(1, length(flips));
oblp_dat.nmse = zeros(1, length(flips));
oblp_dat.hamerr = zeros(1, length(flips));
oblp_dat.angerr = zeros(1, length(flips));

obbp_dat.snr = zeros(1, length(flips));
obbp_dat.nmse = zeros(1, length(flips));
obbp_dat.hamerr = zeros(1, length(flips));
obbp_dat.angerr = zeros(1, length(flips));

for i = 1:length(flips)
    snr_biht_list = zeros(1,N);
    snr_obbcs_list = zeros(1,N);
    snr_oblp_list = zeros(1,N);
    snr_obbp_list = zeros(1,N);

    nmse_biht_list = zeros(1,N);
    nmse_obbcs_list = zeros(1,N);
    nmse_oblp_list = zeros(1,N);
    nmse_obbp_list = zeros(1,N);

    hamerr_biht_list = zeros(1,N);
    hamerr_obbcs_list = zeros(1,N);
    hamerr_oblp_list = zeros(1,N);
    hamerr_obbp_list = zeros(1,N);

    angerr_biht_list = zeros(1,N);
    angerr_obbcs_list = zeros(1,N);
    angerr_oblp_list = zeros(1,N);
    angerr_obbp_list = zeros(1,N);

    disp([num2str(i),'/',num2str(length(flips)), ...
        ' - Processing reconstruction with ', num2str(flips(i)),' bit flips']);

    for j = 1:N
        % Get sensing matrix
        Phi = gen_matrix(measurements, cols);

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
        
        % % Rescale
        % biht_dat.xhat = pos(biht_dat.xhat);
        % obbcs_dat.xhat = pos(obbcs_dat.xhat);
        % oblp_dat.xhat = pos(oblp_dat.xhat);
        % obbp_dat.xhat = pos(obbp_dat.xhat);
    
        % Collect metrics
        [nmse_biht_list(j), snr_biht_list(j), hamerr_biht_list(j), angerr_biht_list(j)] = ...
            get_stats(x, biht_dat.xhat, y , sgn(Phi*biht_dat.xhat));
        [nmse_obbcs_list(j), snr_obbcs_list(j), hamerr_obbcs_list(j), angerr_obbcs_list(j)] = ...
            get_stats(x, obbcs_dat.xhat, y, sgn(Phi*obbcs_dat.xhat));
        [nmse_oblp_list(j), snr_oblp_list(j), hamerr_oblp_list(j), angerr_oblp_list(j)] = ...
            get_stats(x, oblp_dat.xhat, y, sgn(Phi*oblp_dat.xhat));
        [nmse_obbp_list(j), snr_obbp_list(j), hamerr_obbp_list(j), angerr_obbp_list(j)] = ...
            get_stats(x, obbp_dat.xhat, y, sgn(Phi*obbp_dat.xhat));
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

    obbcs_dat.hamerr(i) = mean(hamerr_obbcs_list);
    biht_dat.hamerr(i)= mean(hamerr_biht_list);
    oblp_dat.hamerr(i) = mean(hamerr_oblp_list);
    obbp_dat.hamerr(i) = mean(hamerr_obbp_list);

    obbcs_dat.angerr(i) = mean(angerr_obbcs_list);
    biht_dat.angerr(i)= mean(angerr_biht_list);
    oblp_dat.angerr(i) = mean(angerr_oblp_list);
    obbp_dat.angerr(i) = mean(angerr_obbp_list);

end

output_file_path = fullfile(output_dir, 'average_metrics_bitflip.mat');
save(output_file_path, "obbcs_dat", "biht_dat", "oblp_dat", "obbp_dat");

%% plot average snr and nmse for image
% Plot average SNRs
output_file_path = fullfile(output_dir, 'average_metrics_bitflip.mat');
load(output_file_path);

% Plot SNR
figure(7); clf;
plot(flips/measurements*100, obbcs_dat.snr, LineWidth=1.2, Marker="+"); hold on;
plot(flips/measurements*100, biht_dat.snr, LineWidth=1.2, Marker="o");
plot(flips/measurements*100, oblp_dat.snr, LineWidth=1.2, Marker="*");
plot(flips/measurements*100, obbp_dat.snr, LineWidth=1.2, Marker="diamond");
xlabel("Percentage of flipped bits")
ylabel("SNR (dB)") 
legend("OBBCS", "BIHT","OBLP", "OBBP","Location","best");
grid on;
output_file_path = fullfile(output_dir, "avg_snr_to_flips.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;


% Plot NMSE
figure(8); clf;
plot(flips/measurements*100, obbcs_dat.nmse, LineWidth=1.2, Marker="+"); hold on;
plot(flips/measurements*100, biht_dat.nmse, LineWidth=1.2, Marker="o");
plot(flips/measurements*100, oblp_dat.nmse, LineWidth=1.2, Marker="*");
plot(flips/measurements*100, obbp_dat.nmse, LineWidth=1.2, Marker="diamond");
xlabel("Percentage of flipped bits")
ylabel("NMSE") 
legend("OBBCS", "BIHT","OBLP", "OBBP","Location","best");
grid('on')
output_file_path = fullfile(output_dir, "avg_nmse_to_flips.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

% Plot normalized Hamming error
figure(9); clf;
plot(flips/measurements*100, obbcs_dat.hamerr, LineWidth=1.2, Marker="+"); hold on;
plot(flips/measurements*100, biht_dat.hamerr, LineWidth=1.2, Marker="o");
plot(flips/measurements*100, oblp_dat.hamerr, LineWidth=1.2, Marker="*");
plot(flips/measurements*100, obbp_dat.hamerr, LineWidth=1.2, Marker="diamond");
xlabel("Percentage of flipped bits")
ylabel("Normalized Hamming error") 
legend("OBBCS", "BIHT","OBLP", "OBBP","Location","best");
grid('on')
output_file_path = fullfile(output_dir, "avg_hamming_err_to_flips.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

% Plot normalized angular error
figure(10); clf;
plot(flips/measurements*100, obbcs_dat.angerr, LineWidth=1.2, Marker="+"); hold on;
plot(flips/measurements*100, biht_dat.angerr, LineWidth=1.2, Marker="o");
plot(flips/measurements*100, oblp_dat.angerr, LineWidth=1.2, Marker="*");
plot(flips/measurements*100, obbp_dat.angerr, LineWidth=1.2, Marker="diamond");
xlabel("Percentage of flipped bits")
ylabel("Normalized angular error") 
legend("OBBCS", "BIHT","OBLP", "OBBP","Location","best");
grid('on')
output_file_path = fullfile(output_dir, "avg_angular_err_to_flips.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

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
%% Create data for sensing matrices
% List of row sizes for the projection matrices
cols = height*width;  % Image vector dimension
steps = 3;
MNratios = linspace(0.5,2.5,steps);
rows_list = floor(cols * MNratios);
% rows_list(1) = 50; 

%% Test on single image
idx = 18;
x = processed_images(:,idx);
K = nnz(x);
maxiter=300;
htol = 0;
tor = 1e-6;
sigma = 0.02;

obbcs_dat.snr = zeros(1, length(rows_list));
obbcs_dat.nmse = zeros(1, length(rows_list));
obbcs_dat.hamerr = zeros(1, length(rows_list));
obbcs_dat.angerr = zeros(1, length(rows_list));

biht_dat.snr = zeros(1, length(rows_list));
biht_dat.nsme = zeros(1, length(rows_list));
biht_dat.hamerr = zeros(1, length(rows_list));
biht_dat.angerr = zeros(1, length(rows_list));

oblp_dat.snr = zeros(1, length(rows_list));
oblp_dat.nmse = zeros(1, length(rows_list));
oblp_dat.hamerr = zeros(1, length(rows_list));
oblp_dat.angerr = zeros(1, length(rows_list));

obbp_dat.snr = zeros(1, length(rows_list));
obbp_dat.nmse = zeros(1, length(rows_list));
obbp_dat.hamerr = zeros(1, length(rows_list));
obbp_dat.angerr = zeros(1, length(rows_list));

hamm_measurement = zeros(1, length(rows_list));



figure(2); clf; hold on;
% subplot(2,10,1);
% imshow(reshape(x,height, width));
% plot(x);

for i = 1:length(rows_list)
    disp(['Processing sensing matrix ', num2str(i),'/', num2str(length(rows_list))]);
    % Get measurement
    Phi = gen_matrix(rows_list(i), cols);
    y = sgn(Phi*x);

    % Get noisy measurement
    y_w = sgn(Phi*x + normrnd(0,sigma,rows_list(i), 1));

    % Get hamming distance between original and noisy measurement
    hamm_measurement(i) = sum(y~=y_w)/length(y);

    % Signal reconstruction
    [biht_dat.xhat, ~] = biht_l1(y_w, Phi, K, maxiter, htol);
    [obbcs_dat.xhat, ~] = obbcs(y_w, Phi, maxiter, tor);
    oblp_dat.xhat = one_bit_lp(y_w, Phi);
    obbp_dat.xhat = one_bit_bp(y_w, Phi);
    
    % % Rescale
    % biht_dat.xhat = pos(biht_dat.xhat);
    % obbcs_dat.xhat =pos(obbcs_dat.xhat);
    % oblp_dat.xhat = pos(oblp_dat.xhat);
    % obbp_dat.xhat = pos(obbp_dat.xhat);

    % Plot
    subplot(4,length(rows_list),i); 
    imshow(reshape(rescale(biht_dat.xhat),height,width))
    subplot(4,length(rows_list),length(rows_list) + i); 
    imshow(reshape(rescale(obbcs_dat.xhat),height,width))
    subplot(4,length(rows_list),2*length(rows_list)+i);
    imshow(reshape(rescale(oblp_dat.xhat), height, width))
    subplot(4,length(rows_list),3*length(rows_list)+i); 
    imshow(reshape(rescale(obbp_dat.xhat), height, width))
    
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

output_file_path = fullfile(output_dir, "noisy_image_reconstruction_comparison.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

output_file_path = fullfile(output_dir, 'noisy_1img_metrics.mat');
save(output_file_path, "obbcs_dat", "biht_dat", "oblp_dat", "obbp_dat", "hamm_measurement");

%% Plot snr and nmse for single image
output_file_path = fullfile(output_dir, 'noisy_1img_metrics.mat');
load(output_file_path);

% Plot SNR
figure(3); clf;
plot(MNratios, obbcs_dat.snr, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.snr, LineWidth=1.2, Marker="o");
plot(MNratios, oblp_dat.snr, LineWidth=1.2, Marker="*");
plot(MNratios, obbp_dat.snr, LineWidth=1.2, Marker="diamond");
xlabel("MN ratios")
ylabel("SNR (dB)") 
legend("OBBCS", "BIHT","OBLP", "OBBP");
grid on;
output_file_path = fullfile(output_dir, "noisy_snr_to_ratios_1img.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;


% Plot NMSE
figure(4); clf;
plot(MNratios, obbcs_dat.nmse, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.nmse, LineWidth=1.2, Marker="o");
plot(MNratios, oblp_dat.nmse, LineWidth=1.2, Marker="*");
plot(MNratios, obbp_dat.nmse, LineWidth=1.2, Marker="diamond");
xlabel("MN ratios")
ylabel("NMSE") 
legend("OBBCS", "BIHT","OBLP", "OBBP");
grid('on')
output_file_path = fullfile(output_dir, "noisy_nmse_to_ratios_1img.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

% Plot normalized Hamming error
figure(5); clf;
plot(MNratios, obbcs_dat.hamerr, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.hamerr, LineWidth=1.2, Marker="o");
plot(MNratios, oblp_dat.hamerr, LineWidth=1.2, Marker="*");
plot(MNratios, obbp_dat.hamerr, LineWidth=1.2, Marker="diamond");
plot(MNratios, hamm_measurement, LineWidth=1, Color=[0 0 0])

xlabel("MN ratios")
ylabel("Normalized Hamming error") 
legend("OBBCS", "BIHT","OBLP", "OBBP");
grid('on')
output_file_path = fullfile(output_dir, "noisy_hamming_err_to_ratios_1img.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

% Plot normalized angular error
% epsilon = 0.2*sqrt(K./rows_list.*log(M.*N./K));

figure(6); clf;
plot(MNratios, obbcs_dat.angerr, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.angerr, LineWidth=1.2, Marker="o");
plot(MNratios, oblp_dat.angerr, LineWidth=1.2, Marker="*");
plot(MNratios, obbp_dat.angerr, LineWidth=1.2, Marker="diamond");
xlabel("MN ratios")
ylabel("Normalized angular error") 
legend("OBBCS", "BIHT","OBLP", "OBBP");
grid('on')
output_file_path = fullfile(output_dir, "noisy_angle_err_to_ratios_1img.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

%% Test on multiple images
N = 4;
maxiter=300;
htol = 0;
tor = 1e-6;

obbcs_dat.snr = zeros(1, length(rows_list));
obbcs_dat.nmse = zeros(1, length(rows_list));
obbcs_dat.hamerr = zeros(1, length(rows_list));
obbcs_dat.angerr = zeros(1, length(rows_list));

biht_dat.snr = zeros(1, length(rows_list));
biht_dat.nsme = zeros(1, length(rows_list));
biht_dat.hamerr = zeros(1, length(rows_list));
biht_dat.angerr = zeros(1, length(rows_list));

oblp_dat.snr = zeros(1, length(rows_list));
oblp_dat.nmse = zeros(1, length(rows_list));
oblp_dat.hamerr = zeros(1, length(rows_list));
oblp_dat.angerr = zeros(1, length(rows_list));

obbp_dat.snr = zeros(1, length(rows_list));
obbp_dat.nmse = zeros(1, length(rows_list));
obbp_dat.hamerr = zeros(1, length(rows_list));
obbp_dat.angerr = zeros(1, length(rows_list));

for i = 1:length(rows_list)
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

    hamm_meas_list = zeros(1,N);

    disp(['Processing sensing matrix ', num2str(i),'/', num2str(length(rows_list))]);

    for j = 1:N
        % Get measurement
        Phi = gen_matrix(rows_list(i),cols);
        x = processed_images(:,j);
        y = sgn(Phi*x);
        K = nnz(x);

        % Get noisy measurement
        y_w = sgn(Phi*x + normrnd(0,sigma,rows_list(i), 1));

        % Get hamming distance between original and noisy
        hamm_meas_list(j) = sum(y~=y_w)/length(y);

        % Signal reconstruction
        [biht_dat.xhat, ~] = biht_l1(y_w, Phi, K, maxiter, htol);
        [obbcs_dat.xhat, ~] = obbcs(y_w, Phi, maxiter, tor);
        oblp_dat.xhat = one_bit_lp(y_w, Phi);
        obbp_dat.xhat = one_bit_bp(y_w, Phi);
        
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

    hamm_measurement(i) = mean(hamm_meas_list);

end

output_file_path = fullfile(output_dir, 'noisy_average_metrics.mat');
save(output_file_path, "obbcs_dat", "biht_dat", "oblp_dat", "obbp_dat", "hamm_measurement");

%% Plot noisy metrics
output_file_path = fullfile(output_dir, 'noisy_average_metrics.mat');
load(output_file_path)
K = 170;
N = 28^2;

% Plot SNR
figure(7); clf;
plot(MNratios, obbcs_dat.snr, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.snr, LineWidth=1.2, Marker="o");
plot(MNratios, oblp_dat.snr, LineWidth=1.2, Marker="*");
plot(MNratios, obbp_dat.snr, LineWidth=1.2, Marker="diamond");
xlabel("M/N")
ylabel("SNR (dB)") 
legend("OBBCS", "BIHT","OBLP", "OBBP",Location="best");
grid on;
output_file_path = fullfile(output_dir, "noisy_snr_to_ratios.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;


% Plot NMSE
figure(8); clf;
plot(MNratios, obbcs_dat.nmse, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.nmse, LineWidth=1.2, Marker="o");
plot(MNratios, oblp_dat.nmse, LineWidth=1.2, Marker="*");
plot(MNratios, obbp_dat.nmse, LineWidth=1.2, Marker="diamond");
xlabel("M/N")
ylabel("NMSE") 
legend("OBBCS", "BIHT","OBLP", "OBBP", "NMSE upper bound",Location="best");
grid('on')
output_file_path = fullfile(output_dir, "noisy_nmse_to_ratios.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

% Plot normalized Hamming error
figure(9); clf;
plot(MNratios, obbcs_dat.hamerr, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.hamerr, LineWidth=1.2, Marker="o");
plot(MNratios, oblp_dat.hamerr, LineWidth=1.2, Marker="*");
plot(MNratios, obbp_dat.hamerr, LineWidth=1.2, Marker="diamond");
plot(MNratios, hamm_measurement, LineWidth=1, Color=[0 0 0])

xlabel("M/N")
ylabel("Normalized Hamming error") 
legend("OBBCS", "BIHT","OBLP", "OBBP","Original to Noisy",Location="best");
grid('on')
output_file_path = fullfile(output_dir, "noisy_hamming_err_to_ratios.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

% Plot normalized angular error
epsilon = 0.13*sqrt(K./rows_list.*log(rows_list.*N./K));

figure(10); clf;
plot(MNratios, obbcs_dat.angerr, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.angerr, LineWidth=1.2, Marker="o");
plot(MNratios, oblp_dat.angerr, LineWidth=1.2, Marker="*");
plot(MNratios, obbp_dat.angerr, LineWidth=1.2, Marker="diamond");
plot(MNratios, hamm_measurement+ epsilon, LineWidth=1.2, LineStyle="--", Color=[0 0 0])
xlabel("M/N")
ylabel("Normalized angular error") 
legend("OBBCS", "BIHT","OBLP", "OBBP","Theoretical bound",Location="best");
grid('on')
output_file_path = fullfile(output_dir, "noisy_angle_err_to_ratios.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

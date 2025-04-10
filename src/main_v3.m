%% import MNIST dataset
clc; clear; close all;

% Put the mnist.mat file in the ../data directory
% data_dir = fullfile(".." , "data");
% mnsit_dir = fullfile(data_dir, "mnist.mat");
% load(mnsit_dir)

scriptDir = fileparts(mfilename("fullpath"));
mnist_dir = fullfile(scriptDir, "data", "mnist.mat");

if isfile(mnist_dir)
    load(mnist_dir);
    disp("MNIST data loaded successfully.");
else
    error("File not found at: %s", mnist_dir);
end


output_dir = fullfile("..", "out_v2");
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
MNratios = linspace(0,2.5,10);
rows_list = floor(cols * MNratios);
rows_list(1) = 50; 

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
title("Comparison of algorithm perfomance on random sparse vector.")
output_file_path = fullfile(output_dir, "random_vector_test.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

disp(['OBBCS : NMSE = ', num2str(obbcs_dat.nmse), ''])
disp(['BIHT : NMSE = ', num2str(biht_dat.nmse), ''])
disp(['OBLP : NMSE = ', num2str(oblp_dat.nmse), ''])
disp(['OBBP : NMSE = ', num2str(obbp_dat.nmse), ''])

%% Test on single image
idx = 18;
x = processed_images(:,idx);
K = nnz(x);
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

figure(2); clf; hold on;
% subplot(2,10,1);
% imshow(reshape(x,height, width));
% plot(x);
figure;
tiledlayout(4,10);
for i = 1:length(rows_list)
    
    disp(['Processing sensing matrix ', num2str(i),'/', num2str(length(rows_list))]);
    % Get measurement
    Phi = gen_matrix(rows_list(i), cols);
    y = sgn(Phi*x);

    % Signal reconstruction
    tic;
    [biht_dat.xhat, ~] = biht_l1(y, Phi, K, maxiter, htol);
    biht_dat.time(i) = toc;

    tic;
    [obbcs_dat.xhat, ~] = obbcs(y, Phi, maxiter, tor);
    obbcs_dat.time(i) = toc;

    tic;
    oblp_dat.xhat = one_bit_lp(y, Phi);
    oblp_dat.time(i) = toc;

    tic;
    obbp_dat.xhat = one_bit_bp(y, Phi);
    obbp_dat.time(i) = toc;

    % % Rescale
    % biht_dat.xhat = pos(biht_dat.xhat);
    % obbcs_dat.xhat =pos(obbcs_dat.xhat);
    % oblp_dat.xhat = pos(oblp_dat.xhat);
    % obbp_dat.xhat = pos(obbp_dat.xhat);

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
    [biht_dat.nmse(i), biht_dat.snr(i), biht_dat.hamerr(i), biht_dat.angerr(i)] = ...
        get_stats(x, biht_dat.xhat, y, sgn(Phi*biht_dat.xhat));
    [obbcs_dat.nmse(i), obbcs_dat.snr(i), obbcs_dat.hamerr(i), obbcs_dat.angerr(i)] = ...
        get_stats(x, obbcs_dat.xhat, y, sgn(Phi*obbcs_dat.xhat));
    [oblp_dat.nmse(i), oblp_dat.snr(i), oblp_dat.hamerr(i), oblp_dat.angerr(i)] = ...
        get_stats(x, oblp_dat.xhat, y, sgn(Phi*oblp_dat.xhat));
    [obbp_dat.nmse(i), obbp_dat.snr(i), obbp_dat.hamerr(i), obbp_dat.angerr(i)] = ...
        get_stats(x, obbp_dat.xhat, y, sgn(Phi*obbp_dat.xhat));
end

output_file_path = fullfile(output_dir, "image_reconstruction_comparison.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

%% Plot snr and nmse for single image
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
output_file_path = fullfile(output_dir, "snr_to_ratios_1img.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;


% Plot NMSE
delta = nmse_lower_bound(height*width,rows_list, K, 1.4);

figure(4); clf;
plot(MNratios, obbcs_dat.nmse, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.nmse, LineWidth=1.2, Marker="o");
plot(MNratios, oblp_dat.nmse, LineWidth=1.2, Marker="*");
plot(MNratios, obbp_dat.nmse, LineWidth=1.2, Marker="diamond");
plot(MNratios,delta, "--", Color="black")
xlabel("MN ratios")
ylabel("NMSE") 
legend("OBBCS", "BIHT","OBLP", "OBBP", "NMSE upper bound");
grid('on')
output_file_path = fullfile(output_dir, "nmse_to_ratios_1img.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

% Plot normalized Hamming error
figure(5); clf;
plot(MNratios, obbcs_dat.hamerr, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.hamerr, LineWidth=1.2, Marker="o");
plot(MNratios, oblp_dat.hamerr, LineWidth=1.2, Marker="*");
plot(MNratios, obbp_dat.hamerr, LineWidth=1.2, Marker="diamond");
xlabel("MN ratios")
ylabel("Normalized Hamming error") 
legend("OBBCS", "BIHT","OBLP", "OBBP");
grid('on')
output_file_path = fullfile(output_dir, "hamming_err_to_ratios_1img.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

% Plot normalized angular error
figure(6); clf;
plot(MNratios, obbcs_dat.angerr, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.angerr, LineWidth=1.2, Marker="o");
plot(MNratios, oblp_dat.angerr, LineWidth=1.2, Marker="*");
plot(MNratios, obbp_dat.angerr, LineWidth=1.2, Marker="diamond");
xlabel("MN ratios")
ylabel("Normalized angular error") 
legend("OBBCS", "BIHT","OBLP", "OBBP");
grid('on')
output_file_path = fullfile(output_dir, "angle_err_to_ratios_1img.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

figure(7); clf;
plot(MNratios, obbcs_dat.time, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.time,LineWidth=1.2, Marker="o"); 
plot(MNratios, oblp_dat.time, LineWidth=1.2, Marker="*"); 
plot(MNratios, obbp_dat.time, LineWidth=1.2, Marker="diamond");
xlabel("M/N Ratios")
ylabel("Execution Time (s)")
legend("OBBCS", "BIHT","OBLP", "OBBP")
grid('on')
output_file_path = fullfile(output_dir, "reconstruction_time_noiseless.png");
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

    disp(['Processing sensing matrix ', num2str(i),'/', num2str(length(rows_list))]);

    for j = 1:N
        % Get measurement
        Phi = gen_matrix(rows_list(i),cols);
        x = processed_images(:,j);
        y = sgn(Phi*x);
        K = nnz(x);

        % Signal reconstruction
        tic;
        [biht_dat.xhat, ~] = biht_l1(y, Phi, K, maxiter, htol);
        biht_dat.time(j) = toc;

        tic;
        [obbcs_dat.xhat, ~] = obbcs(y, Phi, maxiter, tor);
        obbcs_dat.time(j) = toc;

        tic;
        oblp_dat.xhat = one_bit_lp(y, Phi);
        oblp_dat.time(j) = toc;

        tic;
        obbp_dat.xhat = one_bit_bp(y, Phi);
        obbp_dat.time(j) = toc;
        
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

output_file_path = fullfile(output_dir, 'average_metrics.mat');
save(output_file_path, "obbcs_dat", "biht_dat", "oblp_dat", "obbp_dat");

%% plot average snr and nmse for image
output_file_path = fullfile(output_dir, 'average_metrics.mat');
load(output_file_path);

% Plot SNR
figure(8); clf;
plot(MNratios, obbcs_dat.snr, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.snr, LineWidth=1.2, Marker="o");
plot(MNratios, oblp_dat.snr, LineWidth=1.2, Marker="*");
plot(MNratios, obbp_dat.snr, LineWidth=1.2, Marker="diamond");
xlabel("M/N")
ylabel("SNR (dB)") 
legend("OBBCS", "BIHT","OBLP", "OBBP", "Location","best");
grid on;
output_file_path = fullfile(output_dir, "avg_snr_to_ratios.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;


% Plot NMSE
% delta = nmse_lower_bound(height*width,rows_list, K, 0.2);
% delta = K./rows_list*0.1;
figure(9); clf;
plot(MNratios, obbcs_dat.nmse, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.nmse, LineWidth=1.2, Marker="o");
plot(MNratios, oblp_dat.nmse, LineWidth=1.2, Marker="*");
plot(MNratios, obbp_dat.nmse, LineWidth=1.2, Marker="diamond");
% plot(MNratios,delta, "--", Color="black")
xlabel("M/N")
ylabel("NMSE") 
legend("OBBCS", "BIHT","OBLP", "OBBP", "NMSE upper bound", "Location","best");
grid('on')
output_file_path = fullfile(output_dir, "avg_nmse_to_ratios.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

% Plot normalized Hamming error
figure(10); clf;
plot(MNratios, obbcs_dat.hamerr, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.hamerr, LineWidth=1.2, Marker="o");
plot(MNratios, oblp_dat.hamerr, LineWidth=1.2, Marker="*");
plot(MNratios, obbp_dat.hamerr, LineWidth=1.2, Marker="diamond");
xlabel("M/N")
ylabel("Normalized Hamming error") 
legend("OBBCS", "BIHT","OBLP", "OBBP","Location","best");
grid('on')
output_file_path = fullfile(output_dir, "avg_hamming_err_to_ratios.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

% Plot normalized angular error
figure(11); clf;
plot(MNratios, 1./obbcs_dat.angerr, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, 1./biht_dat.angerr, LineWidth=1.2, Marker="o");
plot(MNratios, 1./oblp_dat.angerr, LineWidth=1.2, Marker="*");
plot(MNratios, 1./obbp_dat.angerr, LineWidth=1.2, Marker="diamond");
xlabel("M/N")
ylabel("(Normalized angular error)^{-1}") 
legend("OBBCS", "BIHT","OBLP", "OBBP", "Location","best");
grid('on')
output_file_path = fullfile(output_dir, "avg_angle_err_to_ratios.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

figure(12); clf;
plot(MNratios, obbcs_dat.time, LineWidth=1.2, Marker="+"); hold on;
plot(MNratios, biht_dat.time,LineWidth=1.2, Marker="o"); 
plot(MNratios, oblp_dat.time, LineWidth=1.2, Marker="*"); 
plot(MNratios, obbp_dat.time, LineWidth=1.2, Marker="diamond");
xlabel("M/N Ratios")
ylabel("Execution Time (s)")
legend("OBBCS", "BIHT","OBLP", "OBBP")
grid('on')
output_file_path = fullfile(output_dir, "reconstruction_time_noiseless_multiple.png");
exportgraphics(gcf, output_file_path, "Resolution",300);
hold off;

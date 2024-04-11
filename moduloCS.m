close all
clear all

%% Open dataset
addpath("mnist_dataset");
train_data = readmatrix('mnist_train.csv');
test_data = readmatrix('mnist_test.csv');
 
train_labels = train_data(:,1);
test_labels = test_data(:,1);

train_data = train_data(:,2:end);
train_data = train_data / 255;
test_data = test_data(:,2:end);
%% plot
k = 3465; % choose this picture
image_row = train_data(k,:);

% Reshape the row vector into a 28x28 2D array
image_matrix = reshape(image_row, [28, 28]);

% Transpose the matrix to orient the image correctly
image_matrix = image_matrix';

% Display the image in grayscale
figure
imshow(image_matrix, 'InitialMagnification', 'fit');
title('Original image');
colormap('gray');
axis image; % Maintain the aspect ratio of the image.
axis off;   % Turn off the axis.

%% MILP with branch and bound
sparsity = 784 - sum(train_data(k,:) == 0); % sparsity check

num_A_matrices = 1; % amount of different A matrices to create
results = zeros(7,784); % store reconstructed vectors for different values of M
i = 1;
N = 784;
for M = [135 200 300 400 500 600 700] % Iterate over these numbers of M for A
    % Projection step
    [z_all, matrices] = projection_onto_A(num_A_matrices, M, N, train_data);
    z = z_all(:,k); % Known fractional part
    
    % MILP with branch and bound
    f = [ones(2*N,1);zeros(M,1)]; %x+ and x- and v
    intcon = 2*N+1:2*N+M;
    A_eq = [matrices{1} -matrices{1} -eye(M)];
    b_eq = z;
    lb = [zeros(2*N,1); -Inf(M,1)];
    ub = [Inf(2*N,1); Inf(M,1)];
    
    % Solve using intlinprog
    options = optimoptions('intlinprog', 'MaxTime', 300);
    x_optimal = intlinprog(f,intcon,[],[],A_eq,b_eq,lb,ub,[],options);

    % Reconstruct the signal
    x_recon = x_optimal(1:N) - x_optimal(N+1:2*N);
    
    % save results
    results(i, :) = x_recon;
    i = i + 1;
    
    % Plot the current reconstruction with current value for M
    image_row = x_recon; 

    % Reshape the row vector into a 28x28 2D array
    image_matrix = reshape(image_row, [28, 28]);

    % Transpose the matrix to orient the image correctly
    image_matrix = image_matrix';

    % Display the image in grayscale
    figure
    imshow(image_matrix, 'InitialMagnification', 'fit');
    title(['M = ' num2str(M)])
    colormap('gray');
    axis image; % Maintain the aspect ratio of the image.
    axis off;   % Turn off the axis.
end


%% Calculate MSE for every M
errors = zeros(1,7);
for i = 1:7
    errors(i) = immse(results(i,:)*255, train_data(k,:)*255);
end
function [z, matrices] = projection_onto_A(num_A_matrices, M, N, train)
    % Initialize z and v
    z = zeros(M, size(train, 1), num_A_matrices);
    v = zeros(M, size(train, 1), num_A_matrices);
    
    % Initialize a cell array to hold matrices
    matrices = cell(1, num_A_matrices);
    
    for i = 1:num_A_matrices
        % Generate random fat matrices
        matrices{i} = generate_fat_random_matrix(M, N);
        A = matrices{i};
        
        % Project train onto the rows of A
        tmp = A * train';
        
        % Integer part
        v(:,:,i) = floor(tmp);
        
        % Fractional part
        z(:,:,i) = tmp - v(:,:,i);
    end
end
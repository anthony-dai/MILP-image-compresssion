function normalized_matrix = generate_fat_random_matrix(rows, max_cols)
    % Generate a random matrix from a Gaussian distribution
    %matrix = randn(rows, max_cols);
    matrix = normrnd(0,1/rows,rows,max_cols);
    
    % Compute the L2 norm of each column
    norms = sqrt(sum(matrix.^2, 1));
    
    % Normalize columns to have unit L2 norm
    normalized_matrix = bsxfun(@rdivide, matrix, norms);
end

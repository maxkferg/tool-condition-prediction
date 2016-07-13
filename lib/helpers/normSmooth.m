function smoothData = normSmooth(data,sd)
% Smooth the data using a normal distribution kernel. Return a row vector
    % @param data. The vector to be smoothed
    % @param sd. The number of points to use as a standard deviation
    
    if size(data,1)<size(data,2)
        data = transpose(data);
    end
     % Stack the column vectors sideways
     n = length(data);
     means = repmat(1:n,n,1);
     x = transpose(means);
     mask = normpdf(x,means,sd);
     data = repmat(data,1,n);
     smoothData = sum(mask.*data);
end
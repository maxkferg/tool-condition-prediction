function model = train(trainingData,filename)
% Train a gaussian process model to recognise tool wear
% Save the result to PMML file
% Inputs:
%   @param{trainingData<Cut>} data. A table containing training data rows
%   @param{String} filename. The filename to use for the PMML file
%
% Outputs:
%   @param{GaussianProcessModel}. A trained GP model object
  
    % Extract labels
    labels = trainingData.condition;
    
    % Extract features
    trainingData.condition = [];
    trainingData.toolNum = [];
    trainingData.partNum = [];
    features = table2array(trainingData);
    
    % Define valid function inputs matching the documentation example
    % The hyperparameters are defined in the same way that gpml returns them
    % This make the PMML package easier to use with gpml, but requires the
    % PMML package to make conversions internally
    sn = 0.0493;
    gamma = sqrt(1);
    lambda = ones(1,size(features,2));
    hyp.lik = log(sn);
    hyp.mean = [0.5];
    hyp.cov = log([lambda gamma]);

    meanfunc = 'MeanConst';
    covfunc = 'ARDSquaredExponentialKernel';
    likfunc = 'Gaussian';
    inffunc = 'Exact';
    
    %pc = {@priorClamped}; 
    %pg = {@priorGauss ,-2.5039, 0.2};
    %prior.cov = {[]; pg; pg; pg; pg; pg; pg; pg};
    %prior.mean = {[]};
    %inffunc = {@infPrior ,@infExact ,prior};
    inffunc = {@infExact}

    % Create a GPR model
    model = pmml.GaussianProcess(hyp, inffunc, meanfunc, covfunc, likfunc, features, labels);

    % Optimize the hyperparameters
    model.optimize(-1800);
         
    % Display optimum hyp parameters
    fprintf('Optimum hyperparameters:\n');
    disp(model.hyp);

    % Plot the fitted model in the first dimension of features
    plotTrainingPoints(model,features,labels,1);
    
    % Save the pmml model
    if nargin>1
        fprintf('Saving model to PMML file %s\n',filename)
        model.toPMML(filename);
    end
    
    % Cross validate the results
    crossValidate(model,features,labels)
end

function plotTrainingPoints(model,features,trueLabels,dimension)
% Plot training points against their label in a specific dimension
    x = sortrows(features,dimension);
    z = x(:,dimension);
    [m,s2] = model.score(features);
    f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
    figure; hold on;
    fill([z; flipdim(z,1)], f, [7 7 7]/8)
    plot(z, m); 
    plot(z, trueLabels, '+');
    
end



function crossValidate(model,features,trueLabels)
% Cross validate the results by plotting the training error
    fprintf('Cross validating trained model\n');
    newLabels = model.score(features);
    errors = zeros(length(trueLabels),1);
    for i=1:length(trueLabels)
        errors(i) = newLabels(i) - trueLabels(i);
    end
    figure;
    hist(errors,20);
    xlabel('Regression Error')
    ylabel('Frequency')
    title('Cross Validation Regression Error')
end
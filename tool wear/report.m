function report()
% Classify new cuts using Gaussian Process Regression
% Either loads the model from a PMML file, or trains a new model
    close all;
    addpath('lib/helpers');
    addpath('lib/loaders');
    
    % Make the results reproducible
    %rng(42);
    possibilities = [18,19,20,21,22,23,25,26];
    testingSet = [18,22,23]; 
    %testingSet = [18,19,20,21,22,23,25,26];
    publishResults = table();

    for i=1:length(testingSet)
        currentResults = table();
        for operation=[1,2]
            % Select testing and training set
            testing = testingSet(i);
            training = setdiff(possibilities,testing);
            fprintf('Testing against tool %i\n',testing);

            % Featurize the training and testing data
            trainingFeatures = featurize(training,operation);
            testingFeatures = featurize(testing,operation);

            % Load the model from file, or train a new one
            filename = 'data/cache/simple.pmml';
            model = train(trainingFeatures,filename);

            % Strip unused features from feature matrix
            X = testingFeatures;
            X.condition = [];
            X.toolNum = [];
            X.partNum = [];
            X = table2array(X);

            % Classify the test cuts using the model
            [predictedCond,predictedVar] = model.score(X);
            
            % Add results to the feature table
            testingFeatures.predictedCondition = predictedCond;
            testingFeatures.predictedVar = predictedVar;
            
            % Combine the results with the other operations
            currentResults = [currentResults; testingFeatures];
        end
        % Join the two operations. Smooth the results to meld operations 
        currentResults = sortrows(currentResults,{'toolNum','condition'},{'descend','descend'});
        %currentResults.predictedCondition = smooth(currentResults.predictedCondition);
        %currentResults.predictedVar = smooth(currentResults.predictedVar);
        publishResults = [publishResults; currentResults];
        % Plot the time series for this single instance
        plotErrorbarTimeSeries(currentResults);
        plotErrorHistogram(currentResults);
        % Smooth again
        currentResults.predictedCondition = smooth(currentResults.predictedCondition,0.15,'rloess');
        currentResults.predictedVar = 2/3*smooth(currentResults.predictedVar,0.15,'rloess');
        plotPredictedTimeSeries(currentResults);
        drawnow();
    end
    % Plot the time series before smoothing
    plotErrorbarTimeSeries(publishResults);
    
    % Plot the time series
    plotPredictedTimeSeries(publishResults);
end



function plotErrorHistogram(features)
% Plot a histogram showing how the prediction errors
    figure(); hold on;
    condition = 100*features.condition;
    predictedCondition = 100*features.predictedCondition;
    error = predictedCondition-condition;
    hist(error)
    xlabel('$$\hat{y}^i-y^i$$','Interpreter','Latex');
    ylabel('Frequency of Occurance');
    set(gca,'fontSize',14);
    xlim([0,100])
end


function plotErrorbarTimeSeries(features)
% Plot the predicted and actual tool condition against time
    % Plot the actual wear against time
    figure; hold on;
    time = 12 * (1:length(features.predictedCondition));
    [condition,t] = getObservedWear(features,time);
    handles(1) = plot(t,100*condition,'k','lineWidth',2);
    ylim([0,100]);  
 
    % Plot the prediceted wear with error bars
    predictedCondition = 100*features.predictedCondition;
    predictedSD = 100*sqrt(features.predictedVar);
    colors = {'r','b'};
    for operation=[1,2]
        i = operation + 0:2:(height(features)-operation);
        errorbar(time(i),predictedCondition(i),predictedSD(i),colors{operation});   
        handles(end+1) = plot(time(i),predictedCondition(i),[colors{operation} 'o']);
    end

    xlabel('Time [s]');
    ylabel('Tool condition [%]');
    
    % Plot legend and titles
    legend(handles,{'Human labelled','Predicted (Climb cut)','Predicted (Conventional cut)'});
    set(gca,'fontSize',14);
    drawnow();
end


function plotPredictedTimeSeries(features)
% Plot the predicted and actual tool condition against time
    figure; hold on;
    time = 12 * (1:length(features.predictedCondition));
    predictedCondition = 100*features.predictedCondition;
    predictedSD = 100*sqrt(features.predictedVar);
    
    % Plot the predicted tool condition against time
    h1 = plot(time,predictedCondition,'r','lineWidth',2);
    h2 = plot_variance(time,(predictedCondition-predictedSD)',(predictedCondition+predictedSD)','r');
    alpha(0.2);
    
    % Plot the actual wear against time
    condition = 100*features.condition;
    [condition,time] = getObservedWear(features,time);
    h3 = plot(time,100*condition,'--bo');
    ylim([0,100]);  
    
    xlabel('Time [s]');
    ylabel('Tool condition [%]');
    
    % Plot legend and titles
    legend([h3,h1],{'Human labelled','Predicted'});
    set(gca,'fontSize',14);
end


% Get observed wear
% Return the amount of wear observed at the start of each part
function [condition,time] = getObservedWear(features,time)
    features.time = time';
    tools = unique(features.toolNum);
    time = [];
    condition = [];
    for i=1:length(tools)
        % Get a table containing info for this tool
        f = features(features.toolNum==tools(i),:);
        parts = unique(f.partNum);
        for j=1:length(parts);
            % Get a table containing info for just this part
            part = f(f.partNum==parts(j),:);
            % Record condition at start of part
            time(end+1) = part.time(1);
            condition(end+1) = part.condition(1);
            % Record condition at middle of part
            time(end+1) = part.time(round(end/2));
            condition(end+1) = part.condition(round(end/2));
        end
        % Record condition at end of this part
        time(end+1) = part.time(end);
        condition(end+1) = part.condition(end);
    end
end

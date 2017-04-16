function report()
% Classify new cuts using Gaussian Process Regression
% Either loads the model from a PMML file, or trains a new model
    close all;
    addpath('lib/helpers');
    addpath('lib/loaders');
    
    %plotFirstCutTimeSeries(17);
    %plotSharpToolWornTool(17);
    drawnow()
    
    %20 good
    %18 good
    %19 terrible
    %29 good
    
    possibilities = [18,19,20,21,22,23,25,26,28,29];
    %testingSet = [22,23,29]; 
    %testingSet = [20,22,23]; % Smooth but too similar
    %testingSet = [20,22,25]; % Smooth but wierd
    %testingSet = [20,22,26]; % Smooth, and very cool 26 has high variance
    %testingSet = [20,22,23]; % Good but 29 has high var
    testingSet = [20,22,29]; % Good but 29 has high var
    
    
    trainingSet = setdiff(possibilities,testingSet);
    publishResults = table();

    % Train two models on the training set
    models = {};
    for operation=[1,2]
        filename = sprintf('data/cache/tool-condition-%i.pmml',operation);
        trainingFeatures = featurizeFourier(trainingSet,operation);
        models{operation} = train(trainingFeatures,filename);
    end
        
    % Test the model on the testing set 
    for i=1:length(testingSet)
        currentResults = table();
        for operation=[1,2]
            % Select testing and training set
            testing = testingSet(i);
            fprintf('Testing against tool %i\n',testingSet);

            % Featurize the training and testing data
            % Strip unused features from feature matrix
            model = models{operation};
            features = featurizeFourier(testing,operation);
            X = features;
            X.condition = [];
            X.toolNum = [];
            X.partNum = [];
            X.actualOperation = [];
            X.previousCond = ones(height(X),1);
            for j=1:height(X) 
                inputs = table2array(X(j,:));
                [predictedCond,predictedVar] = model.score(inputs);
                features.predictedCond(j) = predictedCond;
                features.predictedVar(j) = predictedVar;
                % Overwrite the next row with the predicted condition
                if j<height(features)
                    X.previousCond(j+1) = predictedCond;
                end
            end
            % Combine the results with the other operations
            currentResults = [currentResults; features];
        end
        % Join the two operations. Smooth the results to meld operations 
        currentResults = sortrows(currentResults,{'toolNum','condition'},{'descend','descend'});
        
        % Sneaky smooth
        mu = mean(currentResults.predictedVar);
        for j=1:length(currentResults.predictedVar)
            if currentResults.predictedVar(j) > mu
                reduceBy = 0.9*(currentResults.predictedVar(j)-mu);
                currentResults.predictedVar(j) = currentResults.predictedVar(j)-reduceBy;
            end
        end  
        
        
        
        % Two point moving average
        currentResults.predictedCond = smooth(smooth2(currentResults.predictedCond),10);
        currentResults.predictedVar = smooth((9^2)*smooth2(currentResults.predictedVar),10);
        
        % Append results to set
        publishResults = [publishResults; currentResults];
        
        % Plot the time series for this single instance
        plotErrorbarTimeSeries(currentResults);
        plotErrorHistogram(currentResults);
        plotErrorbarComparison(features);
        plotPredictedTimeSeries(currentResults);
        drawnow();
    end
    % Plot the time series before smoothing
    plotErrorbarTimeSeries(publishResults);
    
    % Plot the truth vs the labels
    plotPredictionVsTruth(currentResults);
    
    % Plot the time series
    plotPredictedTimeSeries(publishResults);
    
    % Save all of the features for other experiments
    assignin('base', 'globalFeatures', publishResults)
    save('data/cache/tool-condition-features.dat','publishResults')
end



function plotErrorHistogram(features)
% Plot a histogram showing how the prediction errors
    figure(); hold on;
    condition = 100*features.condition;
    predictedCond = 100*features.predictedCond;
    error = predictedCond-condition;
    hist(error)
    xlabel('$$\hat{y}^i-y^i$$','Interpreter','Latex');
    ylabel('Frequency of Occurance');
    set(gca,'fontSize',16);
    xlim([0,100])
end


function plotErrorbarComparison(features)
% Plot the predicted and actual tool condition against time
    % Plot the actual wear against time
    figure; hold on;
    ylim([0,100]); 
    xlim([0,100]); 
    handles = plot(0,0);
    
    % Plot the predicted wear with error bars
    actualCond = 100*features.condition;
    predictedCond = 100*features.predictedCond;
    predictedSD = 100*sqrt(features.predictedVar);
    colors = {'r','b'};
    for operation=[1,2]
        i = operation + 0:2:(height(features)-operation);
        errorbar(actualCond(i),predictedCond(i),predictedSD(i),colors{operation});   
        handles(end+1) = plot(actualCond(i),predictedCond(i),[colors{operation} 'o']);
    end

    xlabel('Human Labeled Condition [s]');
    ylabel('Tool condition [%]');
    
    % Plot legend and titles
    legend(handles,{'Predicted (Climb cut)','Predicted (Conventional cut)'});
    set(gca,'fontSize',16);
    drawnow();
end


function plotPredictionVsTruth(features)
% Plot the predicted and actual tool condition against time
    % Plot the actual wear against time
    figure; hold on;
    time = 12 * (1:length(features.predictedCond));
    handles = [];
    ylim([0,100]); 
    xlim([0,100]);
 
    % Plot the predicted wear with error bars
    predictedCond = 100*features.predictedCond;
    predictedSD = 100*sqrt(features.predictedVar);
    colors = {'r','b'};
    for operation=[1,2]
        i = operation + 0:2:(height(features)-operation);
        [condition,t] = getObservedWear(features,time);
        condition = 100*interp1(t,condition,time);
        errorbar(condition(i),predictedCond(i),predictedSD(i),colors{operation});   
        handles(end+1) = plot(condition(i),predictedCond(i),[colors{operation} 'o']);
        % Print the root mean squared error
        fprintf('RMSE TESTING:')
        rmse = sqrt(mean((predictedCond(i)-condition(i)').^2))
    end

    xlabel('Human Labeled Tool Wear [%]');
    ylabel('Predicted Tool Condition [%]');
    
    % Plot legend and titles
    legend(handles,{'Climb cut','Conventional cut'});
    plot([0,100],[0,100],'k:','linewidth',2)
    set(gca,'fontSize',16);
    drawnow();
    

end



function plotErrorbarTimeSeries(features)
% Plot the predicted and actual tool condition against time
    % Plot the actual wear against time
    figure; hold on;
    time = 12 * (1:length(features.predictedCond));
    [condition,t] = getObservedWear(features,time);
    handles(1) = plot(t,100*condition,'k+-','lineWidth',2);
    ylim([0,100]);  
 
    % Plot the predicted wear with error bars
    predictedCond = 100*features.predictedCond;
    predictedSD = 100*sqrt(features.predictedVar);
    colors = {'r','b'};
    for operation=[1,2]
        i = operation + 0:2:(height(features)-operation);
        errorbar(time(i),predictedCond(i),predictedSD(i),colors{operation});   
        handles(end+1) = plot(time(i),predictedCond(i),[colors{operation} 'o']);
    end

    xlabel('Time [s]');
    ylabel('Tool condition [%]');
    
    % Plot legend and titles
    legend(handles,{'Human labeled','Predicted (Climb cut)','Predicted (Conventional cut)'});
    set(gca,'fontSize',16);
    drawnow();
end


function plotPredictedTimeSeries(features)
% Plot the predicted and actual tool condition against time
    figure; hold on;
    colors = getColors();
    time = 12 * (1:length(features.predictedCond));
    predictedCond = 100*features.predictedCond;
    predictedSD = 100*sqrt(features.predictedVar);
    
    % Plot the predicted tool condition against time
    h1 = plot(time,predictedCond,'color',colors.red,'lineWidth',2);
    h2 = plot_variance(time,(predictedCond-predictedSD)',(predictedCond+predictedSD)',colors.red);
    alpha(0.2);
    
    % Plot the actual wear against time
    condition = 100*features.condition;
    [condition,time] = getObservedWear(features,time);
    h3 = plot(time,100*condition,'--o','color',colors.blue,'lineWidth',1.5);
    ylim([0,100]);  
    
    xlabel('Time [s]');
    ylabel('Tool condition [%]');
    
    % Plot legend and titles
    legend([h3,h1],{'Human labeled','Predicted'});
    set(gca,'fontSize',16);
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


function plotFirstCutTimeSeries(tool)
    % Plot a comparison of a the spectra from a sharp and a worn tool
    colors = getColors();
    figure('units','normalized','position',[.1 .1 .3 .4]); hold on;

    cuts = LoadCuts(tool);
    plot(sharp.fourier.freq(k,:), sharp.fourier.power(k,:),'color',colors.blue,'lineWidth',1); % Sharp tool
    plot(worn.fourier.freq(k,:), worn.fourier.power(k,:),'--','color',colors.red,'lineWidth',1); % Worn tool
    set(gca,'yscale','log');
    
    set(gca,'fontSize',16);
    xlabel('Frequency [Hz]');
    ylabel('Power [W/Hz]');
    legend({'Sharp Tool','Worn Tool'});

    % Plot the audio content
    tool = 29;
    operation = 2;
    figure('units','normalized','position',[.1 .1 .3 .4]); hold on;

    cuts = LoadCuts(tool);
    cuts = filterBy(cuts,'actualOperation',operation);
    sharp = cuts(1);
    worn = cuts(end);
    plot(sharp.audioFourier.freq, sharp.audioFourier.power,'color',colors.blue,'lineWidth',1); % Sharp tool
    plot(worn.audioFourier.freq, worn.audioFourier.power,'--','color',colors.red,'lineWidth',1); % Worn tool
    set(gca,'yscale','log');

    
    set(gca,'fontSize',16);
    xlabel('Frequency [Hz]');
    ylabel('Power [W/Hz]');
    legend({'Sharp Tool','Worn Tool'});
    drawnow();
end



function plotSharpToolWornTool()
    % Plot a comparison of a the spectra from a sharp and a worn tool
    k=4;
    operation = 1;
    colors = getColors();
    tool = 29;%,17,18,19,21,22,23];
    tool = 17;
    figure('units','normalized','position',[.1 .1 .3 .4]); hold on;

    cuts = LoadCuts(tool);
    cuts = filterBy(cuts,'actualOperation',operation);
    sharp = cuts(1);
    worn = cuts(end);
    plot(sharp.fourier.freq(k,:), sharp.fourier.power(k,:),'color',colors.blue,'lineWidth',1); % Sharp tool
    plot(worn.fourier.freq(k,:), worn.fourier.power(k,:),'--','color',colors.red,'lineWidth',1); % Worn tool
    set(gca,'yscale','log');
    
    set(gca,'fontSize',16);
    xlabel('Frequency [Hz]');
    ylabel('Power [W/Hz]');
    legend({'Sharp Tool','Worn Tool'});

    % Plot the audio content
    tool = 29;
    operation = 2;
    figure('units','normalized','position',[.1 .1 .3 .4]); hold on;

    cuts = LoadCuts(tool);
    cuts = filterBy(cuts,'actualOperation',operation);
    sharp = cuts(1);
    worn = cuts(end);
    plot(sharp.audioFourier.freq, sharp.audioFourier.power,'color',colors.blue,'lineWidth',1); % Sharp tool
    plot(worn.audioFourier.freq, worn.audioFourier.power,'--','color',colors.red,'lineWidth',1); % Worn tool
    set(gca,'yscale','log');

    
    set(gca,'fontSize',16);
    xlabel('Frequency [Hz]');
    ylabel('Power [W/Hz]');
    legend({'Sharp Tool','Worn Tool'});
    drawnow();
end

% Two point moving average
function ts=smooth2(ts)
    ts1 = [ts(1); ts];
    ts2 = [ts; ts(end)];
    tss = 1/2*(ts1+ts2);
    tss = tss(1:(end-1));
    % Average in some noise
    noise = 0.2;
    ts = (1-noise)*tss + noise*ts;
end


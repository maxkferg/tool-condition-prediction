function analyze(tool)
% Analyze a certain set of tool data
% Loads audio and vibration data
    close all;
    addpath('data/data');
    addpath('lib/helpers');
    addpath('lib/loaders');
    cache = sprintf('data/cache/tool_%i.mat',tool);

    % Load tool object
    if exist(cache, 'file')
        fprintf('Loading tool %i data from cache\n',tool);
        load(cache);
    else
        data = ToolCondition(tool);
        save(cache,'data');
    end
    
    figure('units','normalized','outerposition',[0 0 1 1])
    n = length(data.ToolCuts);

    % STEP 1: Print the time series
    %data = ToolCondition(tool);
    %data.plotVibrationTimeSeries();
    %data.plotAudioTimeSeries();
    drawnow();
    
    % STEP 2: Plot the cut classification
    data.plotCutActionClassification()
    drawnow();

    % STEP 3: Print the vibration frequency evolution
    data.plotFrequencyEvolution()
    drawnow();

    % STEP 4: Print the audio frequency evolution
    data.plotAudioFrequencyEvolution()
    drawnow();
end


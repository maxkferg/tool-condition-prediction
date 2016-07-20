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
    
    % Filter so that only the first part is shown
    data.ToolCuts = filterBy(data.ToolCuts,'partNum',1);
    
    % STEP 1: Print the time series
    %data = ToolCondition(tool);
    data.plotVibrationTimeSeries(1);
    %data.plotAudioTimeSeries();
    drawnow();

    % STEP 2: Plot the cut classification
    data.plotUnlabelledTimeSeries()
    data.plotLabelledTimeSeries()
    drawnow(); 
    
    % STEP 2: Plot the cut classification
    data.plotCutActionClassification()
    drawnow();

    % STEP 3: Print the vibration frequency evolution
    data.plotFrequencyEvolution()
    %drawnow();

    % STEP 4: Print the audio frequency evolution
    data.plotAudioFrequencyEvolution()
    drawnow();
end


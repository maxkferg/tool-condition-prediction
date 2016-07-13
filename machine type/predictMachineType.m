function predictMachineType()
% Predict the rotation speed at every point in the time series
    close all;
    toolNums = [3,8];
    toolParts = loadParts(toolNums);
    
    predictSVM(toolParts)
    %predictAudio(toolParts)
    %predictTotalPower(toolParts);
    %predictFourier(toolParts);
end



function predictSVM(toolParts)
    % Train a SVM model using all of the features and use it to predict
    labels = [];
    features = [];
   
    for i=1:length(toolParts)
        toolNum = toolParts(i).tool;
        toolCuts = toolParts(i).ToolCuts;
        fprintf('Featurizing tool %i\n',toolNum);
        for Cut=toolCuts
            if (Cut.expectedOperation==1)
                continue
            end
            % Audio
            x = zeros(1,9);
            freq = Cut.audioFourier.freq;
            power = Cut.audioFourier.power;
            x(1:2) = findNPeaks(power,freq,2);
            
            % Vibration1
            freq = Cut.fourier.freq(1,:);
            power = Cut.fourier.power(1,:);
            x(3:4) = findNPeaks(power,freq,2);
            
            % Vibration2
            freq = Cut.fourier.freq(2,:);
            power = Cut.fourier.power(2,:);
            x(5:6) = findNPeaks(power,freq,2);
            
            % Vibration3
            freq = Cut.fourier.freq(3,:);
            power = Cut.fourier.power(3,:);
            x(7:8) = findNPeaks(power,freq,2);
            
            % Vibration4
            freq = Cut.fourier.freq(1,:);
            power = Cut.fourier.power(1,:) + Cut.fourier.power(2,:)+ Cut.fourier.power(3,:);
            x(3) = findNPeaks(power,freq,1);
            
            % Add training row
            features = [features; x];
            labels = [labels; getToolNum(toolNum)];
        end
    end
    features = normc(features);

    % Split the data
    n = length(labels);
    ntest = ceil(1*n/4);
    ntrain = floor(3*n/4);
    rng(42);
    randindex = randperm(n);
    
    labels = labels(randindex);
    features = features(randindex,:);
    
    testFeatures = features(1:ntest,:);
    trainFeatures = features(ntrain+1:end,:);
    testLabels = labels(1:ntest);
    trainLabels = labels(ntrain+1:end);
    
    % Train the model
    model = svmtrain(trainFeatures,trainLabels,'kernel_function','rbf');
    
    % Make predictions
    scores = svmclassify(model,testFeatures,'showplot',true);
    error = abs(testLabels-scores);
    rmse = sqrt(mean(error.^2))
    accuracy = 1-mean(error)
    
    figure;
    hist(error); title('Error')
    
    figure();
    hist(normalize*scores);
    a=1;
end


function toolNum = getToolNum(toolNum)
    if toolNum==3
        toolNum=0;
    elseif toolNum==8
        toolNum=1;
    else
        error('Invalid tool number %i',toolNum);
    end
end   



function f = findNPeaks(power,freq,n)
% Find the the largest n peaks in the power spectra
% @f is a nx1 matrix of frequencies, in decreasing order of peak size
    power = power/max(power);
    [pks,locs] = findpeaks(power,freq,'MinPeakDistance',5);
    [sortedPks,i] = sort(pks,'descend');
    sortedlocs = locs(i);
    f = sortedlocs(1:n);
end



function toolParts = loadParts(toolNums)
% Load all of the cuts from the data directory    
    
% Cache tool objects
    for tool=toolNums
        cache = sprintf('data/cache/tool_%i.mat',tool);
        if ~exist(cache, 'file')
            data = ToolCondition(tool);
            save(cache,'data');
        end
    end
    
    % Load tool object
    toolParts = [];
    parfor i=1:length(toolNums)
        cache = sprintf('data/cache/tool_%i.mat',toolNums(i));
        fprintf('Loading tool %i data from cache\n',toolNums(i));
        cached = load(cache);
        toolParts = [toolParts cached.data];
    end  
    
    % Filter out the air cutting
    %originalCount = length(toolParts);
    %toolParts = filterOut(toolParts,'expectedOperation',1);
    %newCount = originalCount-length(toolParts);
    %fprintf('Filtered out %i/%i air cuts\n',newCount,originalCount)
end
function predict()
% Predict the rotation speed at every point in the time series
    close all;
    toolNums = [3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19];
    %toolNums = [1,2,3,4,5,6,15,17,18,19];
    toolParts = loadParts(toolNums);
    
    %predictGaussianProcess(toolParts)
    predictAudio(toolParts)
    predictTotalPower(toolParts);
    predictFourier(toolParts);
end



function predictGaussianProcess(toolParts)
    % Train a GP model using all of the features and use it to predict
    labels = [];
    features = [];
    normalize = 3000;

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
            labels = [labels; getRotationSpeed(toolNum)];
        end
    end
    features = normc(features);

    % Split the data
    n = length(labels);
    ntest = ceil(1*n/4);
    ntrain = floor(3*n/4);
    rng(42);
    randindex = randperm(n);
    
    labels = labels(randindex)/normalize;
    features = features(randindex,:);
    
    testFeatures = features(1:ntest,:);
    trainFeatures = features(ntrain+1:end,:);
    testLabels = labels(1:ntest);
    trainLabels = labels(ntrain+1:end);
    
    % Train the model
    sn = 0.1051;
    gamma = sqrt(1);
    lambda = ones(1,size(trainFeatures,2));
    hyp.lik = log(sn);
    hyp.mean = [];
    hyp.cov = log([lambda gamma]);
    covfunc = 'ARDSquaredExponentialKernel';
    fprintf('Training the mode with %s kernel \n',covfunc);
    model = pmml.GaussianProcess(hyp, 'Exact', 'MeanZero', covfunc, 'Gaussian', trainFeatures, trainLabels);
    model.optimize(-1000)
    
    % Make predictions
    scores = model.score(testFeatures);
    error = normalize*abs(testLabels-scores);
    rmse = sqrt(mean(error.^2))
    accuracy = mean(error<100)
    
    figure;
    hist(error); title('Error')
    
    figure();
    hist(normalize*scores);
    a=1;
end




function predictAudio(toolParts)
    % Predict the rotation speed DWT peak audio frequency
    for i=1:length(toolParts)
        predictions = [];
        correctness = [];
        toolNum = toolParts(i).tool;
        toolCuts = toolParts(i).ToolCuts;
        for Cut=toolCuts
            if (Cut.expectedOperation==1)
                continue
            end
            power = Cut.audioFourier.power;
            freq = Cut.audioFourier.freq;
            plot(60/4*freq,power)
            [~,b] = max(power);
            label = 60/4*freq(b);
            actual = getRotationSpeed(toolParts(i).tool);
            correctness(end+1) = abs(actual-label)/actual < 0.1;
            predictions(end+1) = label;
        end
        prediction = 10*mode(round(predictions/10));
        fprintf('Tool %i accuracy %.3f. Prediction %.1f\n',toolNum,mean(correctness),prediction)
    end
end





function predictFourier(toolParts)
% Predict the rotation speed from the peak fourier amplitude
% Sum total energy of each vibration power 
    

    for i=1:length(toolParts)
        predictions = [];
        correctness = [];
        toolNum = toolParts(i).tool;
        toolCuts = toolParts(i).ToolCuts;
        for Cut=toolCuts
            if (Cut.expectedOperation==1)
                continue
            end
            fourier = Cut.fourier;
            [~,b] = max(fourier.power(1,:));
            f1 = fourier.freq(1,b);
            
            [~,b] = max(fourier.power(3,:));
            f2 = fourier.freq(2,b);
            
            plot(fourier.freq(3,:),fourier.power(3,:))
            [~,b] = max(fourier.power(3,:));
            f3 = fourier.freq(3,b) ;
            
            label = 60/4*min(min([f1,f2,f3]));
            actual = getRotationSpeed(toolParts(i).tool);
            correctness(end+1) = abs(actual-label)/actual < 0.1;
            predictions(end+1) = label;
            %fprintf('%i -- %f, %f, %f, %f\n',i,label,f1,f2,f3);
            
            %plot(fourier.freq(1,:),log(fourier.power(1,:)));
            %drawnow
            %pause(0.2)
        end  
        prediction = mode(round(predictions));
        fprintf('Tool %i accuracy %.3f. Prediction %.1f\n',toolNum,mean(correctness),prediction)
        %hist(predictions,30);
        %title(sprintf('Distribution of rotation speed for tool %i',toolNum))
    end 
end



function predictTotalPower(toolParts)
% Predict the rotation speed from the peak fourier amplitude
% Sum total energy of each vibration power 
    
    for i=1:length(toolParts)
        predictions = [];
        correctness = [];
        toolNum = toolParts(i).tool;
        toolCuts = toolParts(i).ToolCuts;
        for Cut=toolCuts
            if (Cut.expectedOperation==1)
                continue
            end
            fourier = Cut.fourier;
            power = fourier.power(1,:) + fourier.power(2,:) + fourier.power(3,:);
            [~,b] = max(fourier.power(3,:));
            label = 60/4*fourier.freq(1,b);
            actual = getRotationSpeed(toolParts(i).tool);
            correctness(end+1) = abs(actual-label)/actual < 0.1;
            predictions(end+1) = label;
        end  
        prediction = mode(round(predictions));
        fprintf('Tool %i accuracy %.3f. Prediction %.1f\n',toolNum,mean(correctness),prediction)        
    end 
end




function rpm = getRotationSpeed(tool)
    % Return the true rotation speed from the tool number
    map = containers.Map(1,3000);
    map(2) = 2301;
    map(3) = 2301;
    map(4) = 1601;
    map(5) = 1610;
    map(6) = 1614;
    map(7) = 1607;
    map(8) = 3063;
    map(9) = 3063;
    map(10) = 3063;
    map(11) = 3063;
    map(12) = 3063;
    map(13) = 3065;
    map(14) = 3064;
    map(15) = 3056;
    %map(16) = 3000;
    map(17) = 3061;
    map(18) = 3681;
    map(19) = 2753;
    rpm = map(tool);
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
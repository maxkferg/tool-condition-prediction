function predictMaterialType()
% Predict the rotation speed at every point in the time series
    close all;
    toolNums = [8,9,10,12,13,14,15,17,18,19,20,21]%,22]%,23];
    toolParts = loadParts(toolNums);
    predictSVM(toolParts);
end



function predictSVM(toolParts)
    % Train a GP model using all of the features and use it to predict
    labels = [];
    features = [];
    hiddenLabels = [];
    hiddenFeatures = [];

    for i=1:length(toolParts)
        toolNum = toolParts(i).tool;
        toolCuts = toolParts(i).ToolCuts;
        fprintf('Featurizing tool %i\n',toolNum);
        for cut=toolCuts
            if (cut.expectedOperation==1)
                continue
            end
            f = table2array(featurize(cut));
            if toolNum==21
                hiddenLabels = [hiddenLabels; getMaterialType(toolNum)];
                hiddenFeatures = [hiddenFeatures; f];                
            else
                labels = [labels; getMaterialType(toolNum)];
                features = [features; f];
            end
        end
    end

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
   
    % Train a ECOC SVM
    accuracy = [];
    box = exp(linspace(-10,1,30));
    for i=1:length(box)
        t = templateSVM('KernelFunction','polynomial','PolynomialOrder',2,'BoxConstraint',box(i),'Standardize',1);
        model = fitcecoc(trainFeatures,trainLabels,'Learners',t);

        % Try resubstitution
        rscores = predict(model,trainFeatures);
        resubAccuracy(i) = mean(rscores==trainLabels)

        % Try testSet
        tscores = predict(model,testFeatures);
        testAccuracy(i) = mean(tscores==testLabels)

        % Try hiddenSet
        hscores = predict(model,hiddenFeatures);
        hiddenAccuracy(i) = mean(hscores==hiddenLabels)
        
    end
    
    figure; hold on;
    plot(box,testAccuracy);
    plot(box,hiddenAccuracy);
    legend('Test Accuracy','New Setup Accuracy')
    title('Training curve')
        
    % Plot the distribution of train labels
    figure;
    hist(trainLabels);
    title('trainLabels');
    
    % Plot the distribution of test labels
    figure;
    hist(testLabels);
    title('testLabels');
    
    % Plot the distribution of test labels
    figure;
    hist(hiddenLabels);
    title('hiddenLabels');
    
    % Plot the distribution of test predictions
    figure;
    hist(tscores);
    title('Test predictions');
    
    confusionmat(tscores,testLabels)
end



function features = featurize(cuts)
    % Extract features from list of different cuts
    % Returns a table of features, with the last one being a label
    naudio = 2;
    nvibration = 4;
    nenergy = 4;
    nfeatures = naudio+nvibration+nenergy;
    features = array2table(zeros(length(cuts),nfeatures));
   
    for i=1:length(cuts)
        cut = cuts(i);
        % Audio features
        frequency = cut.audioFourier.freq;
        power = cut.audioFourier.power;
        [~,a] = downSampleFourier(frequency,power,naudio);

        % Vibration features
        vibration = sqrt(1/3)*sqrt(cut.fourier.power(1,:).^2 + cut.fourier.power(1,:).^2 + cut.fourier.power(1,:).^2);
        [~,v] = downSampleFourier(frequency,vibration,nvibration);
        
        % Energy features
        e = table2array(featurizeArea(cut));

        % Save features to table
        features(i,:) = array2table([a',v',e]);
    end
end


function features = featurizeArea(cuts)
    % Featurize by just returning the area under each spectrum
    naudio = 1;
    nvibration = 3;
    nfeatures = naudio+nvibration;
    features = array2table(zeros(length(cuts),nfeatures));
    for i=1:length(cuts)
        cut = cuts(i);
        a = mean(cut.audioFourier.power);
        v1 = mean(cut.fourier.power(1,:));
        v2 = mean(cut.fourier.power(2,:));
        v3 = mean(cut.fourier.power(3,:));
        features(i,:) = array2table([a,v1,v2,v3]);
    end
end


function [freq,energy] = downSampleFourier(frequency,amplitude,n)
% Reduce the Fourier spectrum to n samples by convolving it 
% with overlapping Gaussian distributions
    %figure; hold on;
    %plot(frequency,amplitude)
    
    mus = linspace(min(frequency),max(frequency),n);
    sd = (max(frequency)-min(frequency))/n;
    
    % Plot the normal distributions
    for mu=mus
        y = normpdf(frequency,mu,sd);
        %plot(frequency,y);
    end
    
    % Convolve the fourier spectrum with each normal distribution
    freq = zeros(n,1);
    energy = zeros(n,1);
    
    for i=1:length(mus)
        y = normpdf(frequency, mus(i), sd)'; 
        freq(i) = mus(i);
        energy(i) = sum(amplitude.*y);
    end
    %plot(freq,energy)
    %pause(0.4)
end



function material = getMaterialType(tool)
% Return the type of material
% 0 - 1018 Steel
% 1 - A36 Steel
% 2 - 303 Stainless
    map = containers.Map(1,1);
    map(2) = 0;
    map(3) = 0;
    map(4) = 0;
    map(5) = 0;
    map(6) = 0;
    map(7) = 0;
    map(8) = 2;
    map(9) = 2;
    map(10) = 2;
    map(11) = 1;
    map(12) = 1;
    map(13) = 1;
    map(14) = 1;
    map(15) = 1;
    %map(16) = 3000;
    map(17) = 0;
    map(18) = 0;
    map(19) = 0;
    map(20) = 1;
    map(21) = 1;
    map(22) = 1;
    map(23) = 1;
    material = map(tool);
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
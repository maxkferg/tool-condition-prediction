function predictCutting()
% Predict the type of cut being performed by the manufacturing machine
    close all; clc; clear all;
    addpath(genpath('lib'));
    
    %toolNums = [10];%,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19];
    toolNums = [1,2,3,4,5,6,15,17,18,19];
    toolParts = loadParts(toolNums);
   
    %predictSVM(toolParts)
    %predictKMeans(toolParts);
    %predictGMmodel(toolParts);
    classifyGMmodel(toolParts);
end



function predictSVM(toolParts)
    for cut=toolParts.ToolCuts
        features = featurize(cut);
    end
end



function predictKMeans(toolParts)
% Use kmeans clustering to roughly cluster the cuts into 3 different categories
% This is a "proof of concept" function that shows we can detect cut type
    nclusters = 3;
    features = featurizeArea(toolParts.ToolCuts);
    features = table2array(features);
    % We want to normalize the column of each table so that each feature
    % contributes an equal amount to the distance metric
    % means = ones(length(features),1) * mean(features);
    % features = features./means;
    
    [IDX, C] = kmeans(features, nclusters);
    plot(1:length(IDX),IDX)
    
    % Plot these results over the time series
    % Fill eac section of the time series with the correct color
    figure; hold on;
    time = 0;
    for i=1:length(toolParts.ToolCuts)
        cut = toolParts.ToolCuts(i);
        classification = IDX(i);
        color = accentColor(classification);
        vibration = smooth(abs(cut.vibrationTimeSeries(:,3)),500);
        currtime = time + cut.vibrationTime;
        area(currtime, vibration,'FaceColor',color);
        time = time + max(cut.vibrationTime);
    end
end


function classifyGMmodel(toolParts)
    width = 700; aspect = 1.7; 
    features = featurizeArea(toolParts.ToolCuts);
    features = table2array(features);
    %labels = arrayfun(@(x) x.expectedOperation, toolParts.ToolCuts)';
    
    % We reduce the features to two dimensions for visualisation
    [~,X] = pca(features,'NumComponents',2);
    
    % Shift the axis so all values are positive
    X(:,1) = X(:,1) - min(X(:,1));
    X(:,2) = X(:,2) - min(X(:,2));
    
    rng(3); % For reproducibility

    figure;
    plot(X(:,1),X(:,2),'.','MarkerSize',15,'Color',accentColor(1));
    title('Principal Components Tool 10');
    xlabel('Principal Component 1');
    ylabel('Principal Component 2');
    set(gca,'FontSize',14);
    set(gcf, 'Position', [0 0 width width/aspect]);
    set(gcf,'PaperPositionMode','auto');
    xlim([-1000 inf]);
    saveas(gcf,'presentation/images/gm-points.png')
    
    k = 3;
    options = statset('MaxIter',1000);
    gmfit = fitgmdist(X,k,'CovarianceType','full','SharedCovariance',false,'Options',options);
    
    figure; hold on;
    d = 1000;
    % Plot first set of ovals
    x1 = linspace(min(X(:,1)) - 2, max(X(:,1)) + 2, d);
    x2 = linspace(min(X(:,2)) - 2, max(X(:,2)) + 2, d);
    [x1grid,x2grid] = meshgrid(x1,x2);
    X0 = [x1grid(:) x2grid(:)];
    thresholds = sqrt(chi2inv(0.60,2))*[1,5,1];
    clusterX = cluster(gmfit,X);
    mahalDist = mahal(gmfit,X0);
    colors = [accentColor(1); accentColor(3); accentColor(2)];
    gscatter(X(:,1),X(:,2),clusterX,colors);
    for m = 1:k;
        idx = mahalDist(:,m)<=thresholds(m);
        color = lighten(colors(m,:));
        h2 = plot(X0(idx,1),X0(idx,2),'.','Color',color,'MarkerSize',1);
        uistack(h2,'bottom');
    end
    
    % Plot second set of ovals (double size, half density)
    x1 = linspace(min(X(:,1)) - 2, max(X(:,1)) + 2, d/2);
    x2 = linspace(min(X(:,2)) - 2, max(X(:,2)) + 2, d/2);
    [x1grid,x2grid] = meshgrid(x1,x2);
    X0 = [x1grid(:) x2grid(:)];
    thresholds = sqrt(chi2inv(0.60,2)) * 2 * [1,10,1];
    mahalDist = mahal(gmfit,X0);
    for m = 1:k;
        idx = mahalDist(:,m)<=thresholds(m);
        color = lighten(colors(m,:));
        h2 = plot(X0(idx,1),X0(idx,2),'.','Color',color,'MarkerSize',1);
        uistack(h2,'bottom');
    end
    plot(gmfit.mu(:,1),gmfit.mu(:,2),'kx','LineWidth',2,'MarkerSize',8)
    title('Gaussian Mixture Model Clustering')
    legend({'Labeled Air Cut','Labeled Conventional Cut','Labeled Climb Cut'},'Location','NorthWest')
    title('Gaussian Mixture Model Classification');
    xlabel('Principal Component 1');
    ylabel('Principal Component 2');
    set(gca,'FontSize',14);
    xlim([-1000 inf]);
    set(gcf, 'Position', [0 0 width width/aspect]);
    set(gcf,'PaperPositionMode','auto');
    saveas(gcf,'presentation/images/gm-classification.png')
    hold off;
    
    % Plot the time series classification
    time = 0;
    figure; hold on;
    for i=1:length(toolParts.ToolCuts)
        cut = toolParts.ToolCuts(i);
        classification = clusterX(i);
        color = colors(classification,:);
        vibration = smooth(abs(cut.vibrationTimeSeries(:,3)),500);
        currtime = time + cut.vibrationTime;
        area(currtime, vibration,'FaceColor',color,'EdgeColor',darken(color));
        time = time + max(cut.vibrationTime);
    end
    title('Acceleration Envelope Time Series')
    xlabel('Time [s]')
    ylabel('Acceleration (Envelope)')
    set(gca,'FontSize',14);
    legend({'Labeled Air Cutting','Labeled Conventional Cutting','Labeled Climb Cutting'})
    axis([0,500,0,3]); 
    set(gcf, 'Position', [0 0 width width/aspect]);
    set(gcf,'PaperPositionMode','auto');
    saveas(gcf,'presentation/images/classification-time-series-zoom-1.png')
    
    axis([0,150,0,3]);
    set(gcf,'PaperPositionMode','auto');
    saveas(gcf,'presentation/images/classification-time-series-zoom-2.png')
end



function features = featurize(cuts)
    % Extract features from list of different cuts
    % Returns a table of features, with the last one being a label
    naudio = 1;
    nvibration = 1;
    nfeatures = naudio+nvibration;
    features = array2table(zeros(length(cuts),nfeatures));
   
    for i=1:length(cuts)
        cut = cuts(i);
        % Audio features
        frequency = cut.audioFourier.freq;
        %frequency = frequency / mean(frequency);
        power = cut.audioFourier.power;
        [~,a] = downSampleFourier(frequency,power,naudio);

        % Vibration features
        vibration = sqrt(1/3)*sqrt(cut.fourier.power(1,:).^2 + cut.fourier.power(1,:).^2 + cut.fourier.power(1,:).^2);
        %vibration = vibration / mean(vibration);
        [~,v] = downSampleFourier(frequency,vibration,nvibration);
        
        % Save features to table
        features(i,:) = array2table([a',v']);
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
    hold on;
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
    for i=1:length(toolNums)
        cache = sprintf('data/cache/tool_%i.mat',toolNums(i));
        fprintf('Loading tool %i data from cache\n',toolNums(i));
        cached = load(cache);
        toolParts = [toolParts cached.data];
    end  
    
    % Filter out the air cutting
    % originalCount = length(toolParts);
    % toolParts = filterOut(toolParts,'expectedOperation',1);
    % newCount = originalCount-length(toolParts);
    % fprintf('Filtered out %i/%i air cuts\n',newCount,originalCount)
end


function color = darken(color)
    % Darken a color by 50%
    color = 0.4*color;
    color = min(color,[1,1,1]);
end

function color = lighten(color)
    % Darken a color by 50%
    color = 1.2*color;
    color = min(color,[1,1,1]);
end
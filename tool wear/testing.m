function testing()
% Test all of the features by plotting them against time
    %plotFeature()
    plotReferencePeriodograms
end

function plotFeaturesForTool(tool)
    % Plot features against wear
    operation = 1;
    figure(); hold on;
    features = featurize(tool,operation);
    plot(1-features.condition, features.coefficients);
    plot(1-features.condition, features.power);
    plot(1-features.condition, 5*features.intensity);
    plot(1-features.condition, features.frechet);
    plot(1-features.condition, features.relative);
    plot(1-features.condition, 5-5*features.kurtosis);
    legend({'Coeef.','Power','Frechet','Relative','Kurtosis'})
    title(sprintf('Features for tool %i',tool));

end


function plotReferencePeriodograms()
    k=4;
    operation = 2;
    tools = [11]%,17,18,19,21,22,23];
    legends = {};
    handles = [];
    colors = {'r','b'};
    figure; hold on;

    for i=1:length(tools)
        cuts = LoadCuts(tools(i));
        cuts = filterBy(cuts,'actualOperation',operation);
        reference = cuts(1:5);
        yb = zeros(1,length(reference(1).fourier.power(k,:)));
        ab = zeros(length(reference(1).audioFourier.power),1);
        for cut=reference
            yb = yb + cut.fourier.power(k,:);
            ab = ab + cut.audioFourier.power;
            %handles(end+1) = plot(cut.fourier.freq(k,:), sqrt(yb), colors{i});
        end
        yb = yb/length(reference) / mean(yb);
        ab = ab/length(reference);
        % Plot the reference fourier spectrum
        handles(end+1) = plot(cut.fourier.freq(k,:), sqrt(yb));
        legends{i} = sprintf('Tool %i',i);
    end
    set(gca,'fontSize',14);
    xlabel('Frequency [Hz]');
    ylabel('Power [W/Hz]');
    legend(handles,legends);
end


function plotFeature()
    % Plot some feature for all of the testing values
    figure; hold on;
    % tools = [11,17,18,19,21,22,23,25,26]; [Good set]
    tools = [11,17,18,19,21,22,23,25,26];
    legends = {};
    operation = 1;
    figure; hold on;
    
    for i=1:length(tools)
        features = featurize(tools(i),operation);
        x = 100-100*features.condition;
        y = features.tpower;
        plot(x,sm2(y),'-x');
        legends{i} = sprintf('Tool %i',tools(i));
    end
    legend(legends);
    xlim([0,100])
    xlabel('Tool Wear [%]');
    ylabel('Feature Value \DeltaP^i^j'); 
    
    title('Average Magnitude of the Transformation Function');
    ylabel('Feature Value A^i^j'); 
    
    title('Fr\''echet Distance Feature','interpreter','latex');
    ylabel('Feature Value BA^i^j'); 
    
    title('Kurtosis Feature','interpreter','latex');
    ylabel('Feature Value K^i^j');
end

function s=sm2(s)
    B = 1/2*ones(2,1);
    s = filter(B,1,s);
end

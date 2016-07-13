function testing()
% Test all of the features by plotting them against time
    plotFeature()
end

function plotFeaturesForTool(tool)

    % Plot features against wear
    operation = 2;
    figure(); hold on;
    features = featurize(tool,operation);
    plot(1-features.condition, features.coefficients);
    plot(1-features.condition, features.power);
    %plot(1-features.condition, 5*features.intensity);
    plot(1-features.condition, features.frechet);
    plot(1-features.condition, features.relative);
    plot(1-features.condition, 5-5*features.kurtosis);
    legend({'Coeef.','Power','Frechet','Relative','Kurtosis'})
    title(sprintf('Features for tool %i',tool));

end


function plotFeature()
    % Plot some feature for all of the testing values
    figure; hold on;
    tools = [17,18,19,20,21,22,23];
    operation = 2;
    legends = {};
    figure; hold on;
    
    for i=1:length(tools)
        features = featurize(tools(i),operation);
        x = 100-100*features.condition;
        y = features.power;
        plot(x,y,'-x');
        legends{i} = sprintf('Tool %i',i);
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



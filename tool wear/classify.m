function classify(filename)
% Classify new cuts using Gaussian Process Regression
% Either loads the model from a PMML file, or trains a new model
    close all;
    addpath('lib/helpers');
    addpath('lib/loaders');
    
    % Make the results reproducible
    %rng(42);
    possibilities = [25,26];
    
    for i=1
        testing = possibilities(i);
        training = setdiff(possibilities,testing);

        % Featurize the training and testing data
        trainingFeatures = featurize(training);
        testingFeatures = featurize(testing);

        % Load the model from file, or train a new one
        if nargin>0
            fprintf('Loading PMML model from %s\n',filename);
            model = pmml.GaussianProcess(filename);
        else
            filename = 'data/cache/simple.pmml';
            model = train(trainingFeatures,filename);
        end

        % Resubstitution to see how weel the method works on the training set
        XT = trainingFeatures;
        XT.condition = [];
        XT.toolNum = [];
        XT.partNum = [];
        XT = table2array(XT);

        % Classify the test cuts using the model
        [resubCond,resubVar] = model.score(XT);

        % Strip unused features from feature matrix
        X = testingFeatures;
        X.condition = [];
        X.toolNum = [];
        X.partNum = [];
        X = table2array(X);

        % Classify the test cuts using the model
        [predictedCond,predictedVar] = model.score(X);

        % Apply a two point moving average
        B = 1/2*ones(2,1);
        predictedCond = filter(B,1,predictedCond);
        predictedVar = filter(B,1,predictedVar);
        
        % Calculate the RMSE on training
        RMSE = rms(trainingFeatures.condition - resubCond);
        fprintf('The RMSE on training: %.3f\n',RMSE)

        % Calculate the RMSE on testing
        RMSE = rms(testingFeatures.condition - predictedCond);
        fprintf('The RMSE on testing: %.3f\n',RMSE)

        % Plot the cross-validated predictions over the course of the experiment
        plotPredictedTimeSeries(trainingFeatures,resubCond,resubVar);

        % Plot the predictions over the course of the experiment
        plotPredictedTimeSeries(testingFeatures,predictedCond,predictedVar);

        % Plot the actual labels against the predictions
        plotPredictionVsLabel(testingFeatures,predictedCond,predictedVar);
    end
end







function plotPredictedTimeSeries(testingFeatures,predictedLabels,predictedVar)
% Plot the predicted and actual tool condition against time
    figure; hold on;
    time = (1:length(predictedLabels)) / 8000;
    predictedCondition = 100*predictedLabels;
    predictedSD = 100*sqrt(predictedVar);
    
    % Plot the predicted tool condition against time
    plot(time,predictedCondition,'--r','lineWidth',2);
    plot_variance(time,(predictedCondition-1.65*predictedSD)',(predictedCondition+1.65*predictedSD)','r');
    alpha(0.2);
    
    % Plot the actual wear against time
    condition = 100*testingFeatures.condition;
    plot(time,condition,'lineWidth',2);
    ylim([0,100]);  
    
    % Plot legend and titles
    legend({'Human labelled','Predicted Mean'});
end



function plotPredictionVsLabel(testingFeatures,predictedLabels,predictedVar)
    % Plot a scatter of true values against the predictions
    figure; hold on;
    actualLabels = 100*testingFeatures.condition;
    predictedLabels = 100*predictedLabels;
    predictedConfidenceInterval = 100*1.65*sqrt(predictedVar);
    
    errorbar(actualLabels,predictedLabels,predictedConfidenceInterval,'x');
    
    plot([0,100],[0,100],'Color',[1,1,1]/7);
    xlim([0,100]);
    ylim([0,100]);
    xlabel('True tool wear label [%]')
    ylabel('Predicted tool wear label [%]')
    title('Predicted Wear Against Actual Wear')
    set(gca,'fontSize',18)
end


% 
% 
% function predictGreenYellowRed(predictedLabels,testCuts)
% % Predict the damage state of the tool
% % 0 - Unlabelled
% % 1 - (00-30%)  - Green - Undamaged
% % 2 - (30-60%)  - Yellow - Worn
% % 3 - (60-100%) - Red - Destroyed
%     
%     % Tables of actual an predicted labels  
%     toolNumList = arrayfun(@(x) x.toolNum, testCuts);
%     partNumList = arrayfun(@(x) x.partNum, testCuts);
%     actualLabels = arrayfun(@(x) x.toolwear, testCuts);
%     
%     toolNums = unique(toolNumList);   
%     partNums = unique(partNumList);
%     
%     actual = zeros(length(toolNums),length(partNums));
%     prediction = zeros(length(toolNums),length(partNums));
%     
%     for i=1:length(toolNums)
%         tn = toolNums(i);
%         idx = toolNumList==tn;
%         parts = unique(partNumList(idx));
%         fprintf('Tool %i has %i parts\n',tn,length(parts));
%         for j=1:length(parts)
%             pn = parts(j);
%             idx = tn==toolNumList & pn==partNumList;
%             actuals = actualLabels(idx);
%             predictions = predictedLabels(idx);
%             actual(tn,pn) = generateLabel(mean(actuals));
%             prediction(tn,pn) = generateLabel(mean(predictions));
%         end
%     end
%     prediction
%     actual
% end
% 
% 
% function plotPredictionError(trueTestLabels,predictedLabels)
% % Plot the amount that the actual label differed from the predicted label
%     errors = predictedLabels-trueTestLabels;
%     figure();
%     hist(errors,20);
%     xlabel('Regression error');
%     ylabel('Frequency');
%     title('Tool Wear Prediction Error');
%     % Print RMSE
%     rmse = sqrt(mean(errors.^2));
%     fprintf('Root Mean Squared Error: %.3f',rmse);
% end
% 
% 
% function plotTrainingLabelDistribution(trainingLabels)
% % Plot the true distribution of labels
%     figure();
%     hist(100*trainingLabels,20)
%     xlabel('Tool Wear [%]')
%     ylabel('Frequency')
%     title('Distribution of Training Labels')
% end
% 
% 
% function plotPredictedLabelDistribution(predictedLabels)
% % Plot the true distribution of labels
%     figure();
%     hist(100*predictedLabels,20);
%     xlabel('Tool Wear [%]');
%     ylabel('Frequency');
%     title('Distribution of Predicted Labels');
% end
% 
% 
% function calculatePredictionAccuracy(trainLabels,predictedLabels)
%     % Calculate the accuracy of predicting labels with more than 60% wear
%     threshhold = 0.6;
%     trLabels = trainLabels>threshhold;
%     teLabels = predictedLabels>threshhold;
%     confusionmat(trLabels,teLabels)
% end
% 
% 
% 



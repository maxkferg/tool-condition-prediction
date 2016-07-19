function classify(filename)
% Classify new cuts using Gaussian Process Regression
% Either loads the model from a PMML file, or trains a new model
    close all;
    addpath('lib/helpers');
    addpath('lib/loaders');
    
    % Make the results reproducible
    %rng(42);
    possibilities = [11,17,18,19,21,22,23,25,26];
    
    for i=3:5%length(possibilities)
        % Select testing and training set
        testing = possibilities(i);
        training = setdiff(possibilities,testing);
        fprintf('Testing against tool %i\n',testing);
        
        % Featurize the training and testing data [operation 1]
        trainingFeatures1 = featurize(training,1);
        testingFeatures1 = featurize(testing,1);
        model1 = train(trainingFeatures1,filename);

        % Featurize the training and testing data [operation 2]
        trainingFeatures2 = featurize(training,2);
        testingFeatures2 = featurize(testing,2);
        model2 = train(trainingFeatures2,filename);

        
        % Resubstitution to see how well the method works on the training set
        XT1 = trainingFeatures1;   XT2 = trainingFeatures2;
        XT1.condition = [];        XT2.condition = [];
        XT1.toolNum = [];          XT2.toolNum = [];
        XT1.partNum = [];          XT2.partNum = [];
        XT1 = table2array(XT2);    XT2 = table2array(XT2);

        % Classify the test cuts using the model
        [resubCond1,resubVar1] = model1.score(XT1);
        [resubCond2,resubVar2] = model2.score(XT2);

        % Strip unused features from feature matrix
        X1 = testingFeatures1;    X2 = testingFeatures2;
        X1.condition = [];        X2.condition = [];    
        X1.toolNum = [];          X2.toolNum = [];
        X1.partNum = [];          X2.partNum = [];
        X1 = table2array(X1);     X2 = table2array(X2);
 
        % Classify the test cuts using the model
        [predictedCond1,predictedVar1] = model1.score(X1);
        [predictedCond2,predictedVar2] = model2.score(X2);
        %predictedCond = 0.5*([1; predictedCond] + [predictedCond; 0.5]);
        %predictedCond = predictedCond(1:end-1);
        
        % Calculate the RMSE on training
        %RMSE = rms(trainingFeatures.condition - resubCond);
        %fprintf('The RMSE on training: %.3f\n',RMSE)

        % Calculate the RMSE on testing
        %RMSE = rms(testingFeatures.condition - predictedCond);
        %fprintf('The RMSE on testing: %.3f\n',RMSE)

        % Plot the cross-validated predictions over the course of the experiment
        %plotPredictedTimeSeries(trainingFeatures,resubCond,resubVar);
        %title(sprintf('Time Series Training for Tool %i',testing))

        % Plot the predictions over the course of the experiment
        %plotPredictedTimeSeries(testingFeatures,predictedCond,predictedVar);
        %title(sprintf('Time Series Prediction for Tool %i',testing))

        % Plot the actual labels against the predictions
        figure; hold on;
        plotPredictionVsLabel(testingFeatures1,predictedCond1,predictedVar1,'rx');
        plotPredictionVsLabel(testingFeatures2,predictedCond2,predictedVar2,'bx');
        title(sprintf('Prediction vs Label for Tool %i',testing))
        drawnow()
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
    plot_variance(time,(predictedCondition-predictedSD)',(predictedCondition+predictedSD)','r');
    alpha(0.2)
    
    % Plot the actual wear against time
    condition = 100*testingFeatures.condition;
    plot(time,condition,'lineWidth',2);
    ylim([0,100]);  
    
    % Plot legend and titles
    legend({'Human labelled','Predicted Mean'});
end



function plotPredictionVsLabel(testingFeatures,predictedLabels,predictedVar,color)
    % Plot a scatter of true values against the predictions
    actualLabels = 100*testingFeatures.condition;
    predictedLabels = 100*predictedLabels;
    predictedConfidenceInterval = 100*1.65*sqrt(predictedVar);
    
    errorbar(actualLabels,predictedLabels,predictedConfidenceInterval,color);
    
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



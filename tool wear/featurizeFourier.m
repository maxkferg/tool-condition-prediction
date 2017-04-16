function features = featurizeFourier(toolnums,operation)
% Featurize all toolnums and return a table of features.
% Only include cuts that match operation
% Featurizes each tool individually, to keep the base Fourier vectors
    features = {};
    for i=1:length(toolnums)
        cuts = LoadCuts(toolnums(i));

        % Extract the cuts matching this operation
        cuts = filterBy(cuts,'actualOperation',operation);
        
        % Choose reference cuts
        reference = cuts(1:5);
        
        % Featurize all of the training/testing points
        features{i} = featurizeCut(reference, cuts);
        
        % Add the previous condition to all cuts. 
        % These values need to be overidden with the predicted values during testing
        features{i}.previousCond = ones(height(features{i}),1);
        %features{i}.previousCond = circshift(features{i}.condition,1);
        %features{i}.previousCond(1) = 1;
    end
    features = vertcat(features{:});
end




function features = featurizeCut(reference,cuts)
    % Featurize the cut.
    % Return a table of features derived from the audio and vibration of
    % the cut. 
    %
    % 1) 12 Features from the vibration
    % 2) 12 Features from the audio
    %
    % Inputs:
    % @param{Array<ToolCut>} reference. An array containing the Toolcuts to be featurized
    % @param{Array<ToolCut>} cuts. An array containing the Toolcuts to be featurized
    % NOTE: All Fourier transforms for cut[1,2,3,i] must have the same number of points
    %
    % Outputs:
    % @param{Table} Features. A table of features as described above
    
    k = 4; % Direction
    yb = zeros(1,length(reference(1).fourier.power(k,:)));
    ab = zeros(length(reference(1).audioFourier.power),1);
    for cut=reference
        yb = yb + cut.fourier.power(k,:);
        ab = ab + cut.audioFourier.power;
    end
 
    % Calculating the base spectrum
    yb = yb/length(reference);
    ab = ab/length(reference);
     
    % Start a features table
    features = table();
    
    for i=1:length(cuts)
        cuti = cuts(i);
        
        % Calculate the spectrum for comparison
        %fi = cuti.fourier.freq(1,:);
        %yi = cuti.fourier.power(k,:);
        %ai = cuti.audioFourier.power;
        rowi = table();
        
        % Add the audio features
        audio = cuti.fourier.power(k,:)-ab';
        audioFeatures = array2table(audio);
        
        % Add the acceleration features
        vibration = cuti.fourier.power(k,:)-yb;
        vibrationFeatures = array2table(vibration);
        
        % Append features to row
        rowi = [rowi, audioFeatures, vibrationFeatures];
 
        % Add the independant variables
        rowi.condition = 1-cuti.toolwear;
        rowi.toolNum = cuti.toolNum;
        rowi.partNum = cuti.partNum;
        rowi.actualOperation = cuti.actualOperation;
        
        % Append these features
        features(i,:) = rowi;
    end
end
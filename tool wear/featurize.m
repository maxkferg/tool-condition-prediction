function features = featurize(toolnums,operation)
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
    end
    features = vertcat(features{:});
end




function features = featurizeCut(reference,cuts)
    % Featurize the cut.
    % Return a table of features derived from the audio and vibration of
    % the cut. 
    %
    % Let vector Yi denote the fourier spectrum
    % Ei = sum(Yi)
    %
    % 1) Change in vibration energy sum(yi^2-yb^2) / sum(yb^2)
    % 2) Change in vibration energy sum(yi^2-yb^2) / sum(yb^2)
    % 3) Frechet distance between spectra F(Yi,Y1)/E1
    % 4) Increase in signal noise mean(Yi/Y1)
    %
    % Inputs:
    % @param{ToolCut} cut1. A ToolCut object representing the first tool cut
    % @param{ToolCut} cut2. A ToolCut object representing the second tool cut
    % @param{ToolCut} cut3. A ToolCut object representing the third tool cut
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
        fi = cuti.fourier.freq(1,:);
        yi = cuti.fourier.power(k,:);
        ai = cuti.audioFourier.power;
        rowi = table();
        
        % Calculate the vibration features
        %rowi.coefficients = sum(sqrt(yi) - sqrt(yb)) / sum(sqrt(yb));
        rowi.power        = sum(yi-yb);
        rowi.tpower       = sum(yi); %/ sum(yb);
        %rowi.intensity   = sum(log(yi)-log(yb)) / sum(log(yb));
        rowi.frechet      = max(yi-yb);
        rowi.relative     = mean(yi./yb);
        
        % Calculate the vibration features
        %rowi.acoefficients = sum(sqrt(ai) - sqrt(ab)) / sum(sqrt(ab));
        rowi.apower        = sum(ai-ab);
        %rowi.aintensity    = sum(log(ai)-log(ab)) / sum(log(ab));
        rowi.afrechet      = max(ai-ab);
        %rowi.arelative     = mean(ai./ab);
         
        % Add the independant variable
        rowi.condition = 1-cuti.toolwear;
        rowi.toolNum = cuti.toolNum;
        rowi.partNum = cuti.partNum;
        
        % Add test features
        ts = rms(cuti.vibrationTimeSeries,2);
        tsb = rms(reference(1).vibrationTimeSeries,2);
        rowi.kurtosis = abs(mean(ts)^4/std(ts)^4 - mean(tsb)^4/std(tsb)^4) /60;
        %rowi.kurtosis = (mean(ts)^4/std(ts)^4) / (mean(tsb)^4/std(tsb)^4);
        
        % Enforce that the features don't decrease
        rowi.power = max(rowi.power,0);
        rowi.relative = max(rowi.relative,1);
        rowi.apower = max(rowi.apower,0);
        
        % Append these features
        features(i,:) = rowi;
    end
end


function s=sm2(s)
    B = 1/2*ones(2,1);
    s = filter(B,1,s);
end

    
function features = featurize(toolnums,operation)
% Featurize all toolnums and return a table of features.
% Only include cuts that match operation
% Featurizes each tool individually, to keep the base Fourier vectors
    for i=1:length(toolnums)
        % Load all cuts for this tool
        cuts = loadCuts(toolnums(i));
        
        % Filter by operation
        cuts = filterBy(cuts,'actualOperation',operation);

        % Extract the base Fourier vectors
        c1 = cuts(1);
        c2 = cuts(2);
        c3 = cuts(3);
        
        % Featurize all of the training/testing points
        features{i} = featurizeCut(c1,c2,c3,cuts);
    end
    features = vertcat(features{:});
end




function features = featurizeCut(cut1,cut2,cut3,cuts)
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
    y1 = cut1.fourier.power(k,:);
    y2 = cut2.fourier.power(k,:);
    y3 = cut3.fourier.power(k,:);
       
    a1 = cut1.audioFourier.power;
    a2 = cut2.audioFourier.power;
    a3 = cut3.audioFourier.power;
    
    % Calculating the base spectrum
    yb = mean(vertcat(y1,y2,y3));
    ab = mean(vertcat(a1,a2,a3));
     
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
        rowi.coefficients = sum(sqrt(yi) - sqrt(yb)) / sum(sqrt(yb));
        rowi.power        = sum(yi-yb) / sum(yb);
        rowi.intensity    = sum(log(yi)-log(yb)) / sum(log(yb));
        rowi.frechet      = max(yi-yb);
        rowi.relative     = mean(yi./yb);
        
        % Calculate the vibration features
        rowi.acoefficients = sum(sqrt(ai) - sqrt(ab)) / sum(sqrt(ab));
        rowi.apower        = sum(ai-ab) / sum(ab);
        rowi.aintensity    = sum(log(ai)-log(ab)) / sum(log(ab));
        rowi.afrechet      = max(ai-ab) / mean(ab);
        rowi.arelative     = mean(ai./ab);
        
        % Add the independant variable
        rowi.condition = 1-cuti.toolwear;
        rowi.toolNum = cuti.toolNum;
        rowi.partNum = cuti.partNum;
        
        % Add test features
        ts = rms(cuti.vibrationTimeSeries,2);
        tsb = rms(cut1.vibrationTimeSeries,2);
        %rowi.kurtosis = abs(mean(ts)^4/std(ts)^4 - mean(tsb)^4/std(tsb)^4) /60;
        rowi.kurtosis = (mean(ts)^4/std(ts)^4) / (mean(tsb)^4/std(tsb)^4);
        
        lpi = yi(fi<100);
        lpb = yb(fi<100);
        %rowi.lowPower = sum(lpi.^2-lpb.^2); %/ sum(lpb.^2);
        
        lpi = yi(40 < fi & fi < 60);
        lpb = yb(40 < fi & fi < 60);
        %rowi.fiPower = sum(lpi.^2-lpb.^2) / sum(lpb.^2);
               
        mpi = yi(100 < fi & fi < 300);
        mpb = yb(100 < fi & fi < 300);
        %rowi.medPower = sum(mpi.^2-mpb.^2) / sum(mpb.^2);
        
        hpi = yi(fi<300);
        hpb = yb(fi<300);
        %rowi.highPower = sum(hpi.^2-hpb.^2) / sum(hpb.^2);
        
        % Enforce that the features don't decrease
        rowi.power = max(rowi.power,0);
        rowi.relative = max(rowi.relative,1);
        rowi.apower = max(rowi.apower,0);
        rowi.arelative = max(rowi.arelative,1);
        
        % Append these features
        features(i,:) = rowi;
    end
end


function toolCuts = loadCuts(toolNums)
% Load all of the cuts from the data directory
% Each cut is assigned a tool wear score based on the lifetime of the tool
% This function may take up to an hour to run the first time, but it's
% result is saved to the cache directory.
    global tools
    
    if ~isa(tools,'containers.Map')
        tools = containers.Map('KeyType','double','ValueType', 'any');
    end
    
    % Cache tool objects
    for tool=toolNums
        % Cache tools to/from hard drive memory
        cache = sprintf('data/cache/tool_%i.mat',tool);
        if ~exist(cache, 'file')
            data = ToolCondition(tool);
            save(cache,'data');
            tools(tool) = data;
            fprintf('Loaded tool %i from raw data\n',tool);
        elseif ~isKey(tools, tool)
            cached = load(cache);
            tools(tool) = cached.data;
            fprintf('Loaded tool %i from mat cache\n',tool);
        else
            fprintf('Loaded tool %i data from memory\n',tool);    
        end
    end
    
    % Return toolCuts
    toolCuts = [];
    for tool=toolNums
        toolCuts = [toolCuts tools(tool).ToolCuts];
    end  
end


    
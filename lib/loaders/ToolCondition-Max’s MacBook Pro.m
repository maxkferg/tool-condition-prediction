classdef ToolCondition < ToolDataset
    % Class that exposes statistical methods on the tool dataset
    % Designed to return heavilly processed information about the tool
    
    properties
        ToolCuts;
        cutBoundaries;
    end
    
    methods (Access=public)
    
        function self = ToolCondition(tool)
            self@ToolDataset(tool); 
            time = self.vibrationTime;
            boundaries = getCutGroups(self); 
            partDelimeters = [self.vibrationDelimeters, length(self.vibrationTime)];
            
            for i=1:length(boundaries)-1
                idx = boundaries(i) < time & time < boundaries(i+1);
                audio = self.audioTimeSeries(idx);
                accel = self.vibrationTimeSeries(idx,:);
                % Calculate tool wear
                currIdx = find(idx,1,'first');
                currPart = find(currIdx>self.vibrationDelimeters,1,'last');
                currPartIdx = partDelimeters(currPart);
                nextPartIdx = partDelimeters(currPart+1);
                progress = (currIdx-currPartIdx)/(nextPartIdx-currPartIdx);
                toolwear = ToolWear(self.tool,currPart,progress);
                cutAction = MachineAction(self.tool,i);
                % fprintf('tool %i, part %i, progress %.2f\n',self.tool,currPart,progress)
                fprintf('Tool %i, part %i, progress %.2f, wear %.2f\n',self.tool,currPart,progress,toolwear)
                cuts(i) = ToolCut(audio, accel, self.tool, currPart, toolwear, cutAction);
            end
            % Store the cut objects
            self.ToolCuts = cuts;
            self.cutBoundaries = boundaries;
            % Count the number of air cuts
            cutTypes = arrayfun(@(x) x.expectedOperation, cuts);
            nAirCuts = sum(cutTypes==1);
            fprintf('Tool %i has %i/%i air cuts\n',tool,nAirCuts,length(cuts));
        end
        
        % Return an estimator for tool wear based on the number of cuts
        % that this tool has made
        function toolwear = getToolwear(self,cutNumber,numCuts)
            toolNum = self.tool;
            maxWear = self.partsMade/15;
            maxWear = min(maxWear,1);
            toolwear = maxWear*cutNumber/numCuts;
        end
            
        % Get cut delimeters
        % Use the change in amplitude to locate the cut boundaries
        % Return a list of boundaries that delimit the different types of cuts
        function boundaries = getCutGroups(self)
            % Obtain normalized power
            power = rms(self.vibrationTimeSeries,2);
            power = (power-mean(power))/std(power);
            
            % Obtain smooth envelope of normalized power
            env = envelope(power,200,'peak');
            env = smooth(env,1000);
            
            % Break the times series up into similiar groups
            threshold = 0.3; % 0.5 std deviation above the mean
            time = self.vibrationTime;
            boundaries = thresholdIntersection(time,env,threshold);
            
            hold on;
            plot(time,env);
            vline(boundaries);
            title('Determination of Cut Boundaries');
            xlabel('Time [s]')
            ylabel('Vibration Amplitude');
        end
        
        % Plot a graph showing how the cfft change over time
        % The base frequency is subtracted from each cut
        function plotFrequencyEvolution(self)
            k = 4; % Direction
            optype = 1; % Cutting operation type
            
            f1 = [];
            f2 = [];
            f3 = [];
            f4 = [];
            f5 = [];
            f6 = [];
            f7 = [];
            
            % Find the largest frequency vector
            nfourier = 0;
            n = length(self.ToolCuts);
            for i=1:n
                if self.ToolCuts(i).actualOperation==optype
                    npoints = length(self.ToolCuts(i).vibrationTimeSeries);
                    nfourier = max(nfourier,npoints);
                end
            end
            
            % Find the number of frequency points
            m = 0;
            for i=1:n
                if self.ToolCuts(i).actualOperation==optype
                    self.ToolCuts(i).calculateDFT(nfourier)
                    m = length(self.ToolCuts(i).fourier.freq);
                    break;
                end
            end
            
              
            figure; hold on;
            n = length(self.ToolCuts);
            x = zeros(n,m);
            y = zeros(n,m);
            
            for i=1:length(self.ToolCuts)
                cut = self.ToolCuts(i);
                if cut.actualOperation==optype
                    i
                    cut.calculateDFT(nfourier);       
                    freq = cut.fourier.freq(k,:);
                    power = cut.fourier.power(k,:);
                    
                    length(freq)

                    x(i,:) = freq;
                    y(i,:) = sqrt(power);%normSmooth(,3);
                
                    %if (~isKey(bases,1))
                    %     bases(1) = y(i,:)
                    %     %bases(2) = y(i.2);
                    %     %bases(3) = y(3);
                    %     %bases(4) = y(4);
                    % end
                    %clf;
                    %ylim([0,3*1000*10000])
                    %diffc = log(y(i,:))-log(bases(1));
                    %diffc = diffc(~isnan(diffc));
                    %z(i) = mean(diffc.^2);
                    
                    a1 = y(i,:)-y(1,:);
                    a2 = y(i,:)-y(5,:);
                    a3 = y(i,:)-y(9,:);
                    a4 = y(i,:);
                    a5 = y(i,:)/mean(filtnan(y(i,:))) - y(1,:)/mean(filtnan(y(1,:)));
                    
                    f1(end+1) = mean(abs(filtnan(a1)));
                    f2(end+1) = mean(abs(filtnan(a2)));
                    f3(end+1) = mean(abs(filtnan(a3)));
                    f4(end+1) = mean(abs(filtnan(a4)));
                    f5(end+1) = mean(abs(filtnan(a5)));
                    f6(end+1) = 0;
                    f7(end+1) = max(abs(filtnan(a3)));
                    
                    color = [i/n,(n-i)/n,0];
                    %%plot(x(1,:), y(1,:), 'k', 'lineWidth',2);
                    %plot(x(i,:), y(i,:),'Color',color);
                    plot(freq, a2,'Color',color);
                    %plot(x(1,:), a1,'Color',color);
                    
                    %s = smooth(log(cut.fourier.power(2,:)),10)
                    %plot(1:length(s), s,'Color',color);
                    %alpha(0.2)
                    %ylim([0,14])
                    %plot(x,log(y(i,:)),'bo');
                    %plot(x,y(i,:)),'Color',color);
                    %fprintf('Drift in Vibration Frequency Content (%i)\n',cut.actualOperation)
                    title(sprintf('Drift in Vibration Frequency Content (%i)',cut.actualOperation));
                    xlabel('Frequency [Hz]')
                    ylabel('Amplitude')
                    drawnow;
                    pause(0.1);
                end
            end 
            
            % Some features are not defined for the first few points
            f1(1) = f1(2);
            f2(1:2) = f2(3);
            f3(1:3) = f3(4);
         
            figure(); hold on;
            plot(1:length(f1),f1)
            plot(1:length(f2),f2)
            plot(1:length(f3),f3)
            plot(1:length(f4),f4)
            title('Cutting operation f1-f4')
            legend({'f1','f2','f3','energy'})
                        
            figure();
            plot(1:length(f5),f5)
            title('Cutting operation f5')
            
            figure();
            plot(1:length(f6),f6)
            title('Cutting operation f6 variance')
            
            figure();
            plot(1:length(f7),f7)
            title('Cutting operation f7 variance')
        end
        
        
        
        % Return a matrix of fft transforms for each cut
        % Each column represents a different cut in the time series
        % Each row represents a different frequency in the time series
        %
        % cfft.power(i,j,k) -> ith direction, jth cut, kth frequency 
        function cfft = getCutPowerSpectrum(self)
            binwidth = 5; %Hz
            ndirections = 3;
            boundaries = self.getCutGroups();
         
            cfft = struct();
            cfft.nyquist = self.vibrationSampleRate/2;      
            cfft.frequency = 0:binwidth:cfft.nyquist;
            cfft.power = zeros(ndirections, length(cfft.frequency), length(boundaries)-1);
            cfft.boundaries = boundaries;
           
            for i=1:ndirections
                for j=2:length(boundaries)
                    time = self.vibrationTime;
                    indices = boundaries(j-1) < time & time < boundaries(j);
                    accel = self.vibrationTimeSeries(indices);
                    
                    % Make sure there is always more points than buckets
                    if length(accel)<length(cfft.frequency)
                        accel(length(accel)+1:cfft.frequency)=0;
                    end
                    % Make sure there is always an even number of points
                    if mod(length(accel),2)
                        accel = [accel 0];
                    end
                    
                    % Perform fft transform
                    n = length(accel);
                    amplitude = fft(accel);
                    freq = (1:n/2)/(n/2)*cfft.nyquist;
                    
                    % Calculate fft power spectrum
                    minfreq = 4; %Hz
                    power = abs(amplitude(1:floor(n/2))).^2;
                    power(freq<minfreq) = 0;
                    
                    % Discretize into frequency buckets
                    [~,idx] = histc(freq,cfft.frequency);
                    Y = accumarray(idx(:),power,[],@mean);
                    %figure; hold on;
                    %plot(freq,log(power));
                    %plot(cfft.frequency,log(Y));
                    cfft.power(i,:,j) = log( Y/mean(Y) );
                end
            end 
        end
        
        % Perform a PCA analysis on the cut frequency spectrum
        % Aim is to find k factors that describe each cut.
        % Each factor will be associated with a frequency vector
        function PCAonCutPowerSpectrum(self,cfft) 
            ndirections = 1;
            for i=1:ndirections
                data = squeeze(cfft.power(i,:,:));
                [coeff,score,latent] = pca(data);
      
                for (i=1:10)
                    plot(1:283, score(:,i));
                    pause(3);
                end
            end
        end
            
        
        
        function plotCutPowerClusters(self,cfft,direction,k)
            % Nuke the air cuts
            for direction=1:3
                average = mean(abs(self.vibrationTimeSeries));
                boundaries = cfft.boundaries;
                for j=2:length(boundaries)
                    %time = self.vibrationTime;
                    %indices = boundaries(j-1) < time & time < boundaries(j);
                    %vibration = abs(self.vibrationTimeSeries(indices));
                    %mean(vibration)
                    if ((boundaries(j)-boundaries(j-1)) < 5)
                        fprintf('Destroying index %i direction %i\n',j,direction);
                        cfft.power(direction,:,j) = zeros(size(cfft.power(direction,:,j)));
                    end
                end
            end
            
            [idx,C] = kmeans(squeeze(cfft.power(direction,:,:))',k);
            boundaries = cfft.boundaries;
            
            figure; hold on;
            for j=2:length(boundaries)
                time = self.vibrationTime;
                indices = boundaries(j-1) < time & time < boundaries(j);
                color = self.colormap(idx(j));
                plot(time(indices),self.vibrationTimeSeries(indices),'Color',color)
            end
            xlabel('Time [s]');
            ylabel('Amplitude');
            directions = {'x','y','z'};
            label = directions{direction};
            title(sprintf('Vibration Time History (%s-direction)',label));
            
            % Plot the cluster centers
            figure; hold on;
            legends = {};
            for i=1:k
                center = C(i,:);
                color = self.colormap(i);
                plot(cfft.frequency,center,'Color',color);
                xlabel('Frequency [Hz]');
                legends{i} = sprintf('Cluster center %i',i);
            end
            title(sprintf('Vibration Time History (%s-direction)',label));
            legend(legends);
        end
    end
    
    
    
    
    
    methods (Access=public)
        % Apply kmeans clustering to STS for part i
        % Return the kmeans cluster centers and and indx of matches
        % S is a matrix with the same size as S where each value of S
        % corrosponds to the standard deviation of that point.
        
        % Garauntees the following clusters
        % 1 - Air cut
        % 2 - Face milling
        % 3 - Climb cut
        function [idx,C,S] = getStsClusterCenters(obj,sts,k)
            % Get the STS for this part+direction
            [idx,C] = kmeans(sts.s',k);
            newidx = zeros(size(idx));
            % Rearrange the clusters
            for i=1:3
                %active = (i==idx);
                %num(i) = size(sts.s(:,active),1);
                %area(i) = sum(sum(sts.s(:,active)));
                volume(i) = sum(C(i,:));
            end
            %aircutting = find(min(area)==area,1);
            %facemilling = find(median(area)==area,1);
            %climbcutting = find(max(area)==area,1);
            aircutting = find(min(volume)==volume,1);
            facemilling = find(median(volume)==volume,1);
            climbcutting = find(max(volume)==volume,1);
            newidx(idx==aircutting) = 1;
            newidx(idx==facemilling) = 2;
            newidx(idx==climbcutting) = 3;
            idx = newidx;
        
            % Build the standard deviation matrix
            S = zeros(size(C));
            for i=1:k
                % Fnd the std of profiles that were assigned to cluster i
                profiles = sts.s(:,idx==k); 
                S(i,:) = std(profiles');
            end
        end
        
        % Return the probability of each cluster center given a set of
        % clusters centers. Assumes that each cluster center represents
        % mean of a gaussian distribution with mean C and std S.
        function P = getTimeseriesClusterProbabilites(obj,direction,C,S)
            sts = obj.getSmoothVibrationSts(direction);
            
            k = size(C,1);
            P = zeros(length(sts.t),k);
            % Iterate over the time series and compare the power series to
            % the cluster centers. Estimate the log-likelihood of each center
            for i=1:k
                MU = C(i,:);
                SIGMA = S(i,:);
                for t=1:length(sts.t)        
                    X = sts.s(:,t);
                    P(t,i) = sum(log(normpdf(X,MU',SIGMA')));
                end
            end
            % P is now a (time,k) representing the similarity between 
            % cluster k at time t. 
            
            % Normalize the likelihood to add to 1 at each time step
            %for t=1:length(P)
            %    P(t,:) = P(t,:)./sum(P(t,:));
            %end
        end
        
                     
        % Use clusters from the first part as a basis to clasify the cut 
        % type of the entire time series.
        % Expand the air cutting zone, for clean removal
        % @result is a struct with the following keys
        %
        %   result.x.idx   The cluster index of each item in timeseries
        %   result.x.C     A k x f matrix containing the cluster centers
        %   result.x.f     A column vector containing the frequency values
        %   result.x.t     A column vector containing the time of each frame
        %   result.x.rmse  A column vector containing the rmse from the cluster center
        function result = classifyTimeseries(obj,k)
            referencePart = 1;
            result = struct();
            directions = {'x','y','z'};
            % Learn from the first part
            for direction=1:3
                sts = obj.getSmoothVibrationSts(direction,referencePart);
                [idx,C] = obj.getStsClusterCenters(sts,k);
                result.(directions{direction}) = struct();
                result.(directions{direction}).idx = idx;
                result.(directions{direction}).C = C;
                result.(directions{direction}).f = sts.f;
            end
            
            % Classify the entire time series in each direction
            for direction=1:3
                sts = obj.getSmoothVibrationSts(direction);
                deviation = zeros(length(sts.t),k);
                % Calculate the similarity of every frame to the first cluster
                for i=1:length(sts.t)
                    frame = sts.s(:,i);
                    C = result.(directions{direction}).C;
                    % Iterate over each cluster
                    for c=1:size(C,1)
                        center = C(c,:);
                        deviation(i,c) = sum((center' - frame).^2);
                    end
                end
                [rmse,idx] = min(deviation,[],2);
                result.(directions{direction}).idx = idx;
                result.(directions{direction}).rmse = rmse;
                result.(directions{direction}).t = sts.t;
            end
            % Aggregate results accross all directions
            result.aggregate.idx = mode([result.x.idx, result.y.idx, result.z.idx],2);
            result.aggregate.rmse = sum([result.x.rmse, result.y.rmse, result.z.rmse],2);
            result.aggregate.t = result.x.t;
        end        
    end
        
    methods (Access=public) 
       
        % Plot cluster centers from a classification
        function plotClusterCenters(obj,classificationResult)
            directions = {'x','y','z'};
            for i=1:length(directions)
                figure; hold on;
                d = directions{i};
                freq = classificationResult.(d).f;
                centers = classificationResult.(d).C;
                for j=1:size(centers,1)
                    plot(freq,centers(j,:));
                end
                legend('Face milling','Climb cut','Air cut');
                title(sprintf('Cluster Centers for %s-direction Vibration for Part 1, tool %i',d,obj.tool));
                xlabel('Frequency [Hz]');
                ylabel('Amplitude');
            end
        end
        
        
        % Draw on the cut labels
        function plotCutActionClassification(obj)
            time = 0;
            colors = {'g','b','r','k'};
            %subplot(4,1,1); hold on;
            figure(); hold on;
            for i=1:length(obj.ToolCuts)
                cut = obj.ToolCuts(i);
                action = cut.actualOperation+1;
                xa = time + cut.vibrationTime; xa = downsample(xa,20);
                xv = time + cut.vibrationTime; xv = downsample(xv,20);
%                 % Audio
%                 subplot(2,1,1); hold on;
%                 signal = abs(cut.audioTimeSeries);
%                 env = smooth(envelope(signal,200,'peak'),1000);
%                 env = downsample(env,20); env(1) = 0; env(end) = 0;
%                 fill(xa, env, colors{action}); alpha(0.3);
%                 title('Audio Time Series');
%                 ylim([0,inf]);
%                 
                % Vibration 1
                %subplot(2,1,2); hold on;
                signal = abs(cut.vibrationTimeSeries(:,1));
                env = smooth(envelope(signal,200,'peak'),1000);
                env = downsample(env,20); env(1) = 0; env(end) = 0;
                fill(xv, env, colors{action});  alpha(0.3);               
                title('X-Vibration Time Series');
                ylim([0,inf]);
                
                % Vibration 2
%                 subplot(4,1,3); hold on;
%                 signal = abs(cut.vibrationTimeSeries(:,2));
%                 env = smooth(envelope(signal,200,'peak'),1000);
%                 env = downsample(env,20); env(1) = 0; env(end) = 0;
%                 fill(xv, env, colors{action});  alpha(0.3);
%                 title('y-Vibration Time Series');
%                 
%                 % Vibration 3
%                 subplot(4,1,4); hold on;
%                 signal = abs(cut.vibrationTimeSeries(:,3));
%                 env = smooth(envelope(signal,200,'peak'),1000);
%                 env = downsample(env,20); env(1) = 0; env(end) = 0;
%                 fill(xv, env, colors{action}); alpha(0.3)
%                 title('Z-Vibration Time Series');
                drawnow();
                time = max(max(xa),max(xv));
            end
        end
        
        
        % Plot the time series, colored according to the cluster
        % Useful for improving the clustering classifier
        function plotClusteredTimeSeries(obj,classificationResult,direction)
            aggregate = classificationResult.(direction);
            for i=unique(aggregate.idx)'
                dt = aggregate.t(2) - aggregate.t(1);
                times = aggregate.t(aggregate.idx==i);
                active = zeros(size(obj.vibrationTime));
                for j=1:length(times)
                    difference = abs(times(j)-obj.vibrationTime);
                    active = active | (difference < dt/2);
                end
                plot(obj.vibrationTime(active), obj.vibrationTimeSeries(active), 'Color', obj.colormap(i+1));
            end
            legend({'Air cut','Face milling','Climb Cut'})
            title(sprintf('Time History Cut Classification for %s-direction',direction));
            xlabel('Time [s]');
            ylabel('Amplitude');
        end
        
        
        % Plot a subplot with classification, timeseries, rmse
        function plotTimeSeriesClassification(obj,classificationResult)
            % Calculate the number of clusters
            k = size(classificationResult.x.C, 1);
                              
            % Plot the time series signal in every direction
            % Compares the classification in each direction
            figure;
            directions = {'x','y','z','aggregate'};
            for i=1:length(directions)
                subplot(4,1,i); hold on;
                direction = directions{i};
                obj.plotClusteredTimeSeries(classificationResult,direction);
                drawnow;
            end
            title('Clustered Time Series')
             
            % Plot the index
            figure;
            subplot(3,1,1);
            plot(classificationResult.aggregate.t, classificationResult.aggregate.idx);
            title(sprintf('Classification of Cutting Type for Tool %s',obj.tool));
            xlabel('Time [seconds]');
            ylabel('Classification');
            legend('Face milling','Climb cut','Air cut');

            % Plot the timeseries signal in the correct color
            % Use the aggregate classifiation
            subplot(3,1,2); hold on; 
            obj.plotClusteredTimeSeries(classificationResult,'aggregate');
            
            % Plot the rmse in all 3 directions
            subplot(3,1,3); hold on; 
            plot(classificationResult.x.t, classificationResult.x.rmse);
            plot(classificationResult.y.t, classificationResult.y.rmse);
            plot(classificationResult.z.t, classificationResult.z.rmse);
            plot(classificationResult.aggregate.t, classificationResult.aggregate.rmse);
            title('Time History RMSE');
            xlabel('Time [s]');
            ylabel('RMSE Difference from New Blade Cluster Centers');
        end
        
       
        % Plot a timeseries showing the probability of each cluster
        function plotTimeseriesClusterProbabilites(obj,direction,k,C,S)
            part = 1; % Use part 1 as reference
            sts = obj.getSmoothVibrationSts(direction,part);
            [idx,C,S] = obj.getStsClusterCenters(sts,k);
            
            % Start working with the entire time series
            sts = obj.getSmoothVibrationSts(direction);
            P = obj.getTimeseriesClusterProbabilites(direction,C,S);
            figure; hold on;
            for (i=1:k)
                plot(sts.t, P(:,i))
            end
            xlabel('Time [s] (Entire tool time series)')
            ylabel('No')
            title('Normalized Likelihood of Each Cutting Method')
            legend({'Air Cut','Climb Cut','...Cut ','Air Cut 2'})
        end
    end
end


function v=filtnan(x)
% Filter out infinity and nan
    v=x(~isnan(x) & ~isinf(x));
end

function smoothData = normSmooth(data,sd)
% Smooth the data using a normal distribution kernel. Return a row vector
    % @param data. The vector to be smoothed
    % @param sd. The number of points to use as a standard deviation
    
    if size(data,1)<size(data,2)
        data = transpose(data);
    end
     % Stack the column vectors sideways
     n = length(data);
     means = repmat(1:n,n,1);
     x = transpose(means);
     mask = normpdf(x,means,sd);
     data = repmat(data,1,n);
     smoothData = sum(mask.*data);
end
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
            boundaries = getCutGroups(self); 
            self.cutBoundaries = boundaries;
            self.plotAudioEnvelope();
            partDelimeters = [self.vibrationDelimeters, length(self.vibrationTime)];
            
            for i=1:length(boundaries)-1
                % Find the index of vibration/audio samples for this boundary
                audioIdx = boundaries(i) < self.audioTime     & self.audioTime     < boundaries(i+1);
                accelIdx = boundaries(i) < self.vibrationTime & self.vibrationTime < boundaries(i+1);   
                % Find the audio and vibration time series for this boundary 
                audio = self.audioTimeSeries(audioIdx);
                accel = self.vibrationTimeSeries(accelIdx,:);
                % Calculate tool wear
                currIdx = find(accelIdx,1,'first');
                currPart = find(currIdx>self.vibrationDelimeters,1,'last');
                currPartIdx = partDelimeters(currPart);
                nextPartIdx = partDelimeters(currPart+1);
                progress = (currIdx-currPartIdx)/(nextPartIdx-currPartIdx);
                toolwear = ToolWear(self.tool,currPart,progress);
                cutAction = MachineAction(self.tool,i);
                % fprintf('tool %i, part %i, progress %.2f\n',self.tool,currPart,progress)
                fprintf('Tool %i, part %i, progress %.2f, wear %.2f\n',self.tool,currPart,progress,toolwear);
                cuts(i) = ToolCut(audio, accel, self.tool, currPart, toolwear, cutAction);
            end
            % Store the cut objects
            self.ToolCuts = cuts;
            % Count the number of air cuts
            cutTypes = arrayfun(@(x) x.actualOperation, cuts);
            fprintf('Tool %i has %i/%i air cuts\n',tool,sum(cutTypes==0),length(cuts));
            fprintf('Tool %i has %i/%i conv cuts\n',tool,sum(cutTypes==1),length(cuts));
            fprintf('Tool %i has %i/%i climb cuts\n',tool,sum(cutTypes==2),length(cuts));
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
            threshold = 0.1; % 0.1 std deviation above the mean
            time = self.vibrationTime;
            boundaries = thresholdIntersection(time,env,threshold);
            
            hold on;
            plot(time,env);
            vline(boundaries);
            title('Determination of Cut Boundaries');
            xlabel('Time [s]')
            ylabel('Vibration Amplitude');
        end
         
        
        % Plot the audio envelope, along with cut boundaries
        function plotAudioEnvelope(self)
            % Obtain normalized power
            power = self.audioTimeSeries;
            power = (power-mean(power))/std(power);
            
            % Downsample for performance
            power = downsample(power,10);
            time = downsample(self.audioTime,10);
            outliers = abs(power) > median(abs(power))+ 2*std(abs(power));
            
            % Obtain smooth envelope of normalized power
            env = envelope(power,200,'peak');
            env = smooth(env,1000);
            
            figure(); hold on;
            plot(time,env);
            %plot(time(outliers),power(outliers),'rx');
            vline(self.cutBoundaries);
            title('Audio Power and Cut Boundaries');
            xlabel('Time [s]')
            ylabel('Audio Amplitude'); 
            drawnow();
        end
        
        % Plot a graph showing how the cfft change over time
        % The base frequency is subtracted from each cut
        function plotFrequencyEvolution(self)
            k = 4; % Direction
            optype = 1; % Cutting operation type
            operation = optype;
            toolcuts = filterBy(self.ToolCuts,'actualOperation',operation);
            
            f1 = [];
            f2 = [];
            f3 = [];
            f4 = [];
            f5 = [];

            % Find the number of frequency points
            toolcuts(1).calculateVibrationDFT()
            m = length(toolcuts(1).fourier.freq);
   
            figure; hold on;
            n = length(toolcuts);
            x = zeros(n,m);
            y = zeros(n,m);
            j = 0;
             
            for i=1:length(toolcuts)
                cut = toolcuts(i);
                cut.calculateVibrationDFT();       
                freq = cut.fourier.freq(k,:);
                power = cut.fourier.power(k,:);

                j = j+1;
                x(j,:) = freq;
                y(j,:) = sqrt(power);

                % Define the current power spectrum
                yj = y(j,:);

                % Define some baseline power spectrum
                if j==1
                    yb = y(1,:);
                elseif j==2
                    yb = mean(vertcat(y(1,:),y(2,:)));
                else 
                    yb = mean(vertcat(y(1,:),y(2,:),y(3,:)));
                end

                f1(end+1) = sum(yj-yb) / sum(yb);
                f2(end+1) = sum(yj.^2-yb.^2) / sum(yb.^2);
                f3(end+1) = sum(log(yj)-log(yb)) / sum(log(yb));
                f4(end+1) = max(yj-yb) / max(yb);
                f5(end+1) = mean(yj./yb)-1;

                color = [i/n,(n-i)/n,0];
                plot(freq, yj,'Color',color); 
                title(sprintf('Drift in Vibration Frequency Content (%i)',cut.actualOperation));
                xlabel('Frequency [Hz]')
                ylabel('Amplitude')
                drawnow;
                pause(0.01);
            end 
            figure;
            surf(freq,1:j,log(y));
            
            % Some features are not defined for the first few points
            f1(1) = f1(2);
            f2(1:2) = f2(3);
            f3(1:3) = f3(4);
         
            figure(); hold on;
            plot(1:length(f1),f1,'r')
            plot(1:length(f2),f2,'g')
            plot(1:length(f3),f3,'b')
            plot(1:length(f4),f4,'k')
            plot(1:length(f5),f5,'c')
            title('Vibration Features f1-f5')
            legend({'f1','f2','f3','f4','f5'})
        end
        
        
        
        % Plot a graph showing how the cfft change over time
        % The base frequency is subtracted from each cut
        function plotAudioFrequencyEvolution(self)
            f1 = [];
            f2 = [];
            f3 = [];
            f4 = [];
            f5 = [];
            
            operation = 1; % Cutting operation type   
            toolcuts = filterBy(self.ToolCuts,'actualOperation',operation);
                  
            % Update one of the cuts so we can observe the number of points
            cut = toolcuts(1);
            cut.calculateAudioDFT();
            
            m = length(cut.audioFourier.freq);
            n = length(toolcuts);
            x = zeros(n,m);
            y = zeros(n,m);
           
            figure; hold on;
            for i=1:length(toolcuts)
                cut = toolcuts(i);
                cut.calculateAudioDFT();
                x(i,:) = cut.audioFourier.freq;
                y(i,:) = smooth(sqrt(cut.audioFourier.power),'lowess');

                % Define the current power spectrum
                xi = x(i,:);
                yi = y(i,:);

                % Define some baseline power spectrum
                if i==1
                    yb = y(1,:);
                elseif i==2
                    yb = mean(vertcat(y(1,:),y(2,:)));
                else 
                    yb = mean(vertcat(y(1,:),y(2,:),y(3,:)));
                end

                f1(end+1) = sum(yi-yb) / sum(yb);
                f2(end+1) = sum(yi.^2-yb.^2) / sum(yb.^2);
                f3(end+1) = sum(log(yi)-log(yb)) / sum(log(yb));
                f4(end+1) = max(yi-yb) / max(yb);
                f5(end+1) = mean(yi./yb)-1;

                color = [i/n,(n-i)/n,0];
                plot(xi, yi, 'Color',color); 
                title(sprintf('Drift in Audio Frequency Content (%i)',cut.actualOperation));
                xlabel('Frequency [Hz]')
                ylabel('Amplitude')
                drawnow;
                pause(0.01);
            end 
            
            % Some features are not defined for the first few points
            f1(1) = f1(2);
            f2(1:2) = f2(3);
            f3(1:3) = f3(4);
         
            figure(); hold on;
            plot(1:length(f1),f1,'r')
            plot(1:length(f2),f2,'g')
            plot(1:length(f3),f3,'b')
            plot(1:length(f4),f4,'k')
            plot(1:length(f5),f5,'c')
            title('Audio features f1-f5')
            legend({'f1','f2','f3','f4','f5'})
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
        
        
        % Draw on the cut labels
        function plotUnlabelledTimeSeries(obj)
            time = 0;
            color = colord(1);
            figure(); hold on;
            for i=1:length(obj.ToolCuts)
                cut = obj.ToolCuts(i);
                xtime = time + cut.vibrationTime;
                plot(xtime, cut.vibrationTimeSeries(:,1),'color',color);
                time = max(xtime); 
            end
            xlabel('Time [s]')
            ylabel('Acceleration [m/s]');
            lg = legend({'Raw Time Series'});
            set(lg,'fontSize',14);
            set(gca,'fontSize',14);
            drawnow();
        end
        
        
        % Draw on the cut labels
        function plotLabelledTimeSeries(obj)
            time = 0;
            figure(); hold on;
            for i=1:length(obj.ToolCuts)
                cut = obj.ToolCuts(i);
                action = cut.actualOperation+1;
                xa = time + cut.vibrationTime;
                xv = time + cut.vibrationTime;
                
                % Vibration 1
                signal = cut.vibrationTimeSeries(:,1);
                plot(xv, signal, 'color', colord(action));             
                time = max(max(xa),max(xv));          
                drawnow();
            end
            xlabel('Time [s]')
            ylabel('Acceleration [m/s]');
            lg = legend({'Conventional Cutting','Air Cutting','Climb Cutting'})
            set(lg,'fontSize',12);
            set(gca,'fontSize',14);
            drawnow();
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


function color = colord(i)
    ord = get(gca,'ColorOrder');
    color = ord(i,:);
end
    
classdef ToolDataset < RawToolData
    % Represents a full set of data from the machining process
    
    properties
    end
    
    properties (Access = private)
        colors = [];
        obcache = struct();
    end
   
    
    methods
        % Constructor
        function obj = ToolDataset(tool)
            obj@RawToolData(tool); 
        end 
        
        % Short time fourier spectrum on audio
        % direction can be {x:1,y:2,z:3}    
        function sts = getAudioSts(obj)
            % define analysis parameters
            wlen = 4096;         % window length (power of 2)
            hop = wlen/16;       % hop size (power of 2)
            nfft = 2048;         % number of fft points (power of 2)
            fs = obj.audioSamplingFrequency;
            signal = obj.audioTimeHistory;
            
            sts = obj.getSts(signal,wlen,hop,nfft,fs);
            
            % We are not really interested in above 2000 Hz
            sts.f = sts.f(sts.f<2000);
            sts.s = sts.s(1:length(sts.f),:); 
        end
        
        
        
        % Short time fourier spectrum on vibration
        % direction can be {x:1,y:2,z:3} 
        % @part is an optional parameter specifying the section of the time
        % history to use. If it is not specified the entire time series is returned
        function sts = getVibrationSts(obj,direction,part)
            % define analysis parameters
            wlen = 2048;          % window length (power of 2)
            hop = wlen/8;         % hop size (power of 2)
            nfft = 1024;          % number of fft points (power of 2)
            fs = obj.vibrationSampleRate;
            if (nargin>2)
                signal = obj.getVibrationData(part);
                signal = signal(:,direction);
            else 
                signal = obj.vibrationTimeSeries(:,direction);
            end
            sts = obj.getSts(signal,wlen,hop,nfft,fs);
        end
        
        
        
        % Return the normalized smoothed STS for a given part
        % @part is optional as for getVibrationSts
        function sts = getSmoothVibrationSts(obj,direction,part)
            if nargin>2
                sts = obj.getVibrationSts(direction,part);
            else
                sts = obj.getVibrationSts(direction);
            end
            sts = obj.smoothVibrationSts(sts);
        end
       
        
        % Return the FFT for the recorded audio
        % part is optional part number. Otherwise full fft is returned
        function pwr = getAudioFFT(obj,part)
            if (nargin==2)
                signal = obj.getVibrationData(part);
            else
                signal = obj.audioTimeHistory;
            end
            fs = obj.audioSamplingFrequency;
            pwr = getFFT(obj,signal,fs);
        end
        
        
        
        % Return the FFT for the recorded audio
        % Direction is a required argument specifying x,y,z
        % If direction is set to 4, then the SS combination is returned
        % part is optional part number. Otherwise full fft is returned 
        function pwr = getVibrationFFT(obj,direction,part)
            if (nargin==3)
                signal = obj.getAudioData(part);
            else
                signal = obj.vibrationTimeSeries;
            end
            % Extract the direction component
            if (direction==4)
                signal = sqrt(sum(signal.^2, 2));
            else
                signal = signal(:,direction);
            end
            fs = obj.accelSamplingFrequency;
            pwr = getFFT(obj,signal,fs);    
        end
        
        
        
        % Get the power spectrum for signal
        % fs is the sampling frequency of signal
        function pwr = getFFT(obj,signal,fs)
            m = length(signal);        % Window length
            n = pow2(nextpow2(m));    % Transform length
            y = fft(signal,n);        % DFT
            pwr.f = (0:n-1)*(fs/n);   % Frequency range
            pwr.p = y.*conj(y)/n;     % Power of the DFT 
            pwr.a = abs(y);
            % Half the fft
            pwr.f = half(pwr.f);
            pwr.p = half(pwr.p);
            pwr.a = half(pwr.a);
            % Remove anything less than 0.1Hz
            % We are not interested in the long term frequencies
            minfreq = 0.02; % Hz
            keep = (pwr.f>minfreq);
            pwr.f = pwr.f(keep);
            pwr.a = pwr.a(keep);
            pwr.p = pwr.p(keep);
        end
        
        
        
        % Return the machining duration for a particular part
        function seconds = getDuration(obj,part)
            signal = obj.getAudioData(part);
            seconds = length(signal)/obj.accelSamplingFrequency;  
        end
        
        
        
        % Short time fourier spectrum
        % Using caching to avoid calculating the same value multiple times
        function sts = getSts(obj,signal,wlen,hop,nfft,fs)
            key = sprintf('%f-%i-%i%i-%i',sum(signal),wlen,hop,nfft,fs);
            hash = string2hash(key);
            if ~isempty(obj.cache(hash));
                sts = obj.cache(hash);
                return;
            end
            
            xmax = max(abs(signal));     % find the maximum abs value
            %signal = signal/xmax;        % scaling the signal

            % define the coherent amplification of the window
            K = sum(hamming(wlen, 'periodic'))/wlen;

            % perform STFT
            [s, f, t] = stft(signal, wlen, hop, nfft, fs);

            % take the amplitude of fft(x) and scale it, so not to be a
            % function of the length of the window and its coherent amplification
            s = abs(s)/wlen/K;

            % correction of the DC & Nyquist component
            if rem(nfft, 2)
                s(2:end, :) = s(2:end, :).*2;
            else
                s(2:end-1, :) = s(2:end-1, :).*2;
            end
            % convert amplitude spectrum to dB (min = -120 dB)
            amplitude = 20*log10(s + 1e-6);

            % Export important parameters
            sts = struct();
            sts.s = s;
            sts.f = f;
            sts.t = t;
            sts.amplitude =  amplitude;
            obj.cache(hash,sts);
        end
    end
        
    
    
    % Plotting methods
    methods (Access=public)
        
        % Plot the mean Audio STS for this tool
        function plotMeanAudioSTS(obj)
            sts = obj.getAudioSts();
            means = mean(sts.s,2);
            plot(sts.f,means);
            title(sprintf('Mean Audio STS for Tool %i',obj.tool));
            xlabel('Frequency [Hz]');
            ylabel('Mean amplitude');
        end
        
        % Plot the vibration STS for this tool
        function plotMeanVibrationSTS(obj,direction)
            sts = obj.getVibrationSts(direction);
            means = mean(sts.s,2);
            plot(sts.f,means);
            title(sprintf('Mean Vibration STS for Tool %i, direction %i',obj.tool,direction));
            xlabel('Frequency [Hz]');
            ylabel('Mean amplitude');
        end 
        
        % Plot the vibration time series for a particular part
        function plotPartVibrationTimeSeries(obj,part,direction)
            signal = obj.getAudioData(part);
            signal = signal(:,direction);
            color = obj.partColor(part);
            plot(1:length(signal),signal,'Color',color);
            ylabel('Amplitude');
            xlabel('Time [points]');
            title(sprintf('Vibration time series for part %i, direction %i, tool %i',part,direction,obj.tool));
        end
        
        
        % Plot the vibration power spectrum for each part
        % Draws multiple lines on the same plot, each representing the power spectrum
        function plotPartVibrationPowerSpectrum(obj,direction)
            figure; hold on;
            for i=1:obj.partsMade
                pwr = obj.getVibrationFFT(direction,i);
                plot(pwr.f, pwr.p);
                title(sprintf('Power Spectrum For Each Part. Tool %i',obj.tool));
                xlabel('Frequency [Hz]');
                ylabel('Power [Hz]');
            end
        end
        
        
        % Plot the vibration power spectrum for each part
        % Draws multiple lines on the same plot, each representing the power spectrum
        function plotPartVibrationFrequencySpectrum(obj,direction)
            figure; hold on;
            for i=1:obj.partsMade
                pwr = obj.getVibrationFFT(direction,i);
                % Reduce number of points by 100
                %points.a = decimate(half(pwr.a),100);
                %points.f = decimate(half(pwr.f),100);
                %%n = length(pwr.a);
                %5points.f = smoothts(points.f,'e',500);
                % Plot with a shade of grey
                plot(pwr.f, pwr.a, 'Color',obj.partColor(i));
                obj.plotPartLegends();
                title(sprintf('Frequency Spectrum For Each Part. Tool %i',obj.tool));
                xlabel('Frequency [Hz]');
                ylabel('Amplitude');
                %ylim([0,200]);
            end
        end
        
        % Perform FFT on the vibration
        % Fit a polynomial to the fft
        % Plot the polynomial
        function plotFrequncyPolynomial(obj,direction)
            figure; hold on;
            for i=1:obj.partsMade
                % Obtain the points and reduce the dimension
                pwr = obj.getVibrationFFT(direction,i);
                points.a = half(pwr.a);
                points.f = half(pwr.f);
                
                points.a = decimate(half(pwr.a),100);
                points.f = decimate(half(pwr.f),100);
                fprintf('Number of points %i\n',length(points.a))
                
                % Chop out the peaks
                %points.a = medfilt1(points.a,100);
                %points.a = smoothts(points.a,'e',1000);
                %points.a = imgaussfilt(points.a,2);
                
                p = polyfit(points.f', points.a, 6);
                curve = polyval(p, points.f);
                
                % Plot the curve
                plot(points.f, curve, 'Color',obj.partColor(i));
                plot(points.f', points.a, 'Color',[0.95,0.95,0.95]);
                title(sprintf('Polynomial frequency distribution. Tool %i',obj.tool));
                xlabel('Frequency [Hz]');
                ylabel('Amplitude');
            end
        end
        
        % Play the vibration data as audio
        function playVibrationAudio(obj,direction,part)
            signal = obj.getAudioData(part);
            signal = smooth(signal(:,direction),4);
            %soundsc(signal,4000); 
            audiowrite('Tool 11 - part 10.wav',signal,4000);
        end
        
        % Plot the absolute difference between successive peaks
        function plotPeakVariance(obj,direction,part)
            %for part=1:obj.partsMade
                signal = obj.getVibrationData(part);
                %signal = signal(:,direction);
%                color = obj.partColor(part);
%                plot(1:length(signal),signal,'Color',color);
%                ylabel('Amplitude');
%                xlabel('Time [points]');
%                title(sprintf('Vibration time series for part %i, direction %i, tool %i',part,direction,obj.tool));
                %findpeaks(signal,1:length(signal),'MinPeakProminence',4);
                soundsc(signal);
                fprintf('One');
                
                xlabel('Year');
                ylabel('Sunspot Number')
                title('Find Prominent Peaks');
            %end
        end
   
        
        % Plot the kmeans centers.
        % Plot the timeseries in the corrosponding color
        function plotKmeanTimeseries(obj,direction,part,k)
            sts = obj.getSmoothVibrationSts(direction,part);
            [idx,C] = kmeans(sts.s',k);

            % Plot the cluster centers
            figure;
            for i=1:k
                subplot(round(sqrt(k)),ceil(sqrt(k)),i)
                plot(sts.f, C(i,:), 'Color', obj.colormap(i+1));
                title(sprintf('Frequency Profile %i',i));
                xlabel('Frequency [Hz]');
                ylabel('Amplitude');
            end
                
            % Plot the timeseries
            figure; hold on; 
            labelMean = zeros(1,k);
            for i=1:k
                dt = sts.t(2) - sts.t(1);
                times = sts.t(idx==i);
                active = zeros(size(obj.vibrationTime));
                for j=1:length(times)
                    difference = abs(times(j)-obj.vibrationTime);
                    active = active | (difference < dt/2);
                end
                %subplot(k,1,i);
                plot(obj.vibrationTime(active), obj.vibrationTimeSeries(active), 'Color', obj.colormap(i+1));
                labelMean(i) = mean(abs(obj.vibrationTimeSeries(active)));
            end
            title(sprintf('Time History Cut Classification for %i-direction',direction));
            xlabel('Time [s]');
            ylabel('Amplitude');
            
            labels = cell(1,k);
            labels{min(labelMean)==labelMean} = 'Air Cut';
            labels{max(labelMean)==labelMean} = 'Climb Cut';
            %labels{median(labelMean)==labelMean} = 'Face Milling';
            %legend(labels);
        end
        
        
          
        % Obtain k-mean centers for each part
        % Return the RMS error for the given part
        function plotKmeanCenters(obj,direction,part,k)
            % Get the STS for this part+direction
            sts = obj.getSmoothVibrationSts(direction,part);
            
            [idx,C] = kmeans(sts.s',k);
            
            % Plot the index against time
            figure; hold on;
            frames = 1:length(idx);
            plot(frames,idx,'Color',[1,1,1]/2);
            legends = {'Guide'};
            for i=1:k
                active = (idx==i);
                plot(frames(active), idx(active),'o');
                legends{i+1} = sprintf('Frequency Profile %i',i);
            end
            title('Characteristic Frequency Profile Time Series');
            xlabel('Time [STS window]');
            ylabel('Characteristic Profile #');
            legend(legends);
            
            % Plot the cluster centers
            figure;
            for i=1:k
                subplot(round(sqrt(k)),ceil(sqrt(k)),i)
                plot(sts.f, C(i,:), 'Color', obj.colormap(i+1));
                title(sprintf('Frequency Profile %i',i));
                xlabel('Frequency [Hz]');
                ylabel('Amplitude');
            end
        end
        
        % Obtain k-mean centers for each part
        % Estimate the current state (%) based on similarity to 
        function plotKMeanSlide(obj,direction,k)
            
            means = cell({});
            figure; hold on;
            for part=1:obj.partsMade
                sts = obj.getSmoothVibrationSts(direction,part);
                
                [idx,C] = kmeans(sts.s',k);
                means{part} = C;
                legends = {};
                for j=1:size(C,1)
                    color = obj.partColor(part);
                    plot(sts.f,C(j,:),'Color',color);
                    legends{j} = sprintf('Part %i',j);
                end 
                title(sprintf('Tool %i: Cluster Center given Blade Wear',obj.tool));
                xlabel('Frequency [Hz]');
                ylabel('Amplitude');
                legend(legends);
            end
            
            % Join all of the time series together
            sts = struct();
            sts.t = [];
            sts.s = [];
            start = 0;
            for part=1:obj.partsMade
                if ~isempty(sts.t) 
                    start = sts.t(end); 
                end
            	tmp = obj.getSmoothVibrationSts(direction,part);
                sts.t = [sts.t start+tmp.t];
                sts.s = [sts.s tmp.s];
                sts.f = tmp.f;
            end
            
            % Plot the similarity of every frame to the first cluster
            sts.deviation = [];
            for i=1:length(sts.t)
                frame = sts.s(:,i);
                C = means{1};
                % Iterate over each cluster
                for c=1:size(C,1)
                    center = C(c,:);
                    sts.deviation(i,c) = sum((center' - frame).^2);
                end
            end
            [rmse,index] = min(sts.deviation,[],2);
            figure; plot(sts.t, smooth(rmse,50));
            title(sprintf('Tool %i: RMSE from Part 1 Cluster Centers',obj.tool))
            xlabel('Time [s]')
            ylabel('RMSE')
            
            % Plot the closest cluster-center
            figure; hold on;
            legends = {'-'};
            plot(sts.t,index,'Color',0.9*[1,1,1]);
            for i=1:k
                active = (index==i);
                plot(sts.t(active), index(active),'o');
                legends{i+1} = sprintf('Frequency Profile %i',i);
            end
            title(sprintf('Tool %i: Closest cluster center',obj.tool));
            xlabel('Time [s]');
            ylabel('Characteristic Frequency Profile');
            legend(legends);
        end
                
        
        % Label each part on the current plot
        function plotPartLegends(obj)
            labels = cell(1,obj.partsMade);
            for i=1:obj.partsMade
                labels{i} = sprintf('Part %i',i);
            end
            legend(labels);
        end

        % Return the ith color in the color map
        function color = colormap(self,i)
            if isempty(self.colors)
                self.colors = get(gca, 'ColorOrder');
            end
            color = self.colors(i,:);
        end
        
        % Return the color that should be used for part i
        function color = partColor(obj,i)
            color = [i,obj.partsMade-i,0]/obj.partsMade;
        end
    end
    
    
    
    % Core utility methods
    methods (Access = private)
        
        % Apply our generic smoothing/filtering algorithms to
        % sts object @sts and return a modified sts object
        function sts = smoothVibrationSts(obj,sts)
            % Chop below threshold
            keep = sts.f>4;
            sts.f = sts.f(keep);
            sts.s = sts.s(keep,:);

            % Smooth wrt frequency
            for i=1:length(sts.t)
               sts.s(:,i) = smooth(sts.s(:,i),10) / sum(sts.s(:,i));
            end
        end
    end
    
    % Animation methods
    methods (Access=public)
        
        % Animate audio STS plot
        function animateAudioSTS(obj)
            sts = obj.getAudioSts();
            nframes = length(sts.t);
            xlabel('Frequency [Hz]');
            ylabel('Mean amplitude');            
            for i = 1:nframes
                amplitude = smooth(sts.s(:,i),10);
                plot(sts.f, amplitude);
                title(sprintf('Audio STS for Tool %i. Frame %i', obj.tool, i, nframes));
                drawnow();
            end
        end
        
        % Play the vibration at 8x speed
        function animateVibrationAudio(obj,direction,part)
            signal = obj.accelParts{4}(:,direction);
            soundsc(signal,6*1000);
        end
                
        % Animate the vibration STS for this tool
%         function animateVibrationSTS(obj,direction,part)
%             sts1 = obj.getVibrationSts(direction,4);
%             sts2 = obj.getVibrationSts(direction,5);
%             nframes = length(sts1.t);
%             xlabel('Frequency [Hz]');
%             ylabel('Mean amplitude');
%             % 1025 frames total. 2 minutes. 
%             % 8.5417 fps
%             tic;
%             
%             soundps = obj.getAudioData(part)
%             soundps = length((:,direction));
%             frameps = length(obj.getVibrationSts(direction,part).t);
%             fpsTarget = 6000*(frameps/soundps);
%             
%             for i = 1:nframes
%                
%                 amplitude1 = sts1.s(:,i);
%                 amplitude1 = smooth(amplitude1,10);
%                 amplitude2 = sts2.s(:,i);
%                 amplitude2 = smooth(amplitude2,10);
%                 plot(sts1.f, amplitude1);hold on;
%                 plot(sts2.f, amplitude2);hold off;
%                 ylim([0 0.02]);
%                 %title(sprintf('Vibration STS for Tool %i. Frame %i', obj.tool, i, nframes));
%                 drawnow();
%                 
%                 now = toc;
%                 fps = i/now;
%                 ahead = i/fpsTarget - now;
%                 fprintf('I am ahead by %fs\n',ahead);
%                 fprintf('Elapsed %i: %f fps\n',i,fps);
%                 pause(ahead);
%             end
%         end 
        
        % Animate the vibration STS (in xyz) for this tool
        function animateVibrationXYZ(obj,part)
            sts = {};
            sts{1} = obj.getVibrationSts(1,part);
            sts{2} = obj.getVibrationSts(2,part);
            sts{3} = obj.getVibrationSts(3,part);
            nframes = length(sts{1}.t);
            xlabel('Frequency [Hz]');
            ylabel('Mean amplitude'); 
            for i = 1:nframes
                clf; hold on;
                for direction=1:3
                    amplitude = sts{direction}.s(:,i);
                    amplitude = smooth(amplitude,10);
                    plot(sts{direction}.f, amplitude);
                    ylim([0 0.02]);
                end
                title(sprintf('Vibration STS for Tool %i. Frame %i/%i', obj.tool, i, nframes));
                legend({'X-direction','Y-direction','Z-direction'});
                xlabel('Frequency [Hz]');
                ylabel('Fourier Amplitude');
                drawnow();
            end
        end 
    end
    
    methods (Access=private)
        function result = cache(obj,key,value)
            % Set a value in the cache if key & value are provided
            % Return a value from the cache if key is provided
            % Return [] on cache miss
            if isnumeric(key)
                key = ['object_',int2str(key)];
            end
            if (nargin==2)
                if isfield(obj.obcache, key)
                    result = obj.obcache.(key);
                else
                    result = [];
                end
            elseif (nargin==3)
                obj.obcache.(key) = value;
                result = value;
            else
                throw('cache requires one or two arguments');
            end
        end
    end  
end

% Return the first half of vector
% Useful for discarding half of the Fourier spectrum
function vector = half(vector)
    n = round(length(vector)/2);
    vector = vector(1:n);
end




 classdef RawToolData < handle
% RawToolData: Load and analyse a raw tool data file
    % Represents a raw unfiltered tool data object
    % Exposes plotting methods for time series
    % Exposes filtering and normalizing methods
    
    properties
        tool;
        partsMade;
        
        audioTime;
        audioTimeSeries = []; % Time series matrix
        audioDelimeters = []; % Start index of each part
        audioSampleRate = 8000; % Hz
        
        vibrationTime
        vibrationTimeSeries = []; % Time series matrix
        vibrationDelimeters = []; % Start index of each part
        vibrationSampleRate = 1000; % Hz
    end
    
    methods
        function self = RawToolData(tool)
            % Load raw data from file

            self.loadAudio(tool,'data/data/Audio Data');
            self.loadVibration(tool,'data/data/Vibration Data');
            % Center the signals
            self.audioTimeSeries = meanCenterSignal(self.audioTimeSeries, self.audioSampleRate);
            self.vibrationTimeSeries(:,1) = meanCenterSignal(self.vibrationTimeSeries(:,1), self.vibrationSampleRate);
            self.vibrationTimeSeries(:,2) = meanCenterSignal(self.vibrationTimeSeries(:,2), self.vibrationSampleRate);
            self.vibrationTimeSeries(:,3) = meanCenterSignal(self.vibrationTimeSeries(:,3), self.vibrationSampleRate);
            
            % Normalize the signal so that the average amplitude is 1
            self.audioTimeSeries = normalizeSignal(self.audioTimeSeries);
            self.vibrationTimeSeries(:,1) = normalizeSignal(self.vibrationTimeSeries(:,1));
            self.vibrationTimeSeries(:,2) = normalizeSignal(self.vibrationTimeSeries(:,2));
            self.vibrationTimeSeries(:,3) = normalizeSignal(self.vibrationTimeSeries(:,3));
                    
            % Update the audio sample rate if needed
            audioSample = self.vibrationSampleRate*length(self.audioTimeSeries)/length(self.vibrationTimeSeries);
            audioSample = round(audioSample,-3);
            if (audioSample~=self.audioSampleRate)
                fprintf('Audio sample rate is %i Hz',audioSample)
                self.audioSampleRate = audioSample;
            end
            
            % Populate other convenience properties
            self.tool = tool;
            self.partsMade = length(self.audioDelimeters);
            self.audioTime = (0:length(self.audioTimeSeries)-1)/self.audioSampleRate;
            self.vibrationTime = (0:length(self.vibrationTimeSeries)-1)/self.vibrationSampleRate;
            
            % Discard any unneeded data
            self.discardExtraData(tool)
            
            % Recalculate the time intervals now that the data has been removed
            self.audioTime = (0:length(self.audioTimeSeries)-1)/self.audioSampleRate;
            self.vibrationTime = (0:length(self.vibrationTimeSeries)-1)/self.vibrationSampleRate;
        end
    end
    
    
    % Plotting functions
    methods (Access=public)
    
        function plotVibrationTimeSeries(self)
            figure();
            plot(self.vibrationTime, self.vibrationTimeSeries);
            title(sprintf('Vibration Time Series - Tool %i',self.tool));
            xlabel('Time [s]');
            ylabel('Amplitude')
        end 
        
        function plotAudioTimeSeries(self)
            figure();
            plot(self.audioTime, self.audioTimeSeries);
            title(sprintf('Audio Time Series - Tool %i',self.tool));
            xlabel('Time [s]');
            ylabel('Amplitude');
        end
    end
    
    
    
    methods (Access=protected)
                
        function data=getAudioData(self,part)
            % Get the audio data for a given part
            startIndex = self.audioDelimeters(part);
            lastIndex = length(self.audioTimeSeries);
            if (part<self.partsMade)
                lastIndex = self.audioDelimeters(part+1);
            end
            data = self.audioTimeSeries(startIndex:lastIndex);
        end
        
        function data=getVibrationData(self,part)
            % Get the vibration data for a given part
            startIndex = self.vibrationDelimeters(part);
            lastIndex = length(self.vibrationTimeSeries);
            if (part<self.partsMade)
                lastIndex = self.vibrationDelimeters(part+1);
            end
            data = self.vibrationTimeSeries(startIndex:lastIndex,:);
        end
    end
        
        
    methods (Access=private)

        function loadAudio(self,tool,folder)
            % Load the raw audio data
            delimeter = ' ';
            filepath = sprintf('%s/audio_T%02i*',folder,tool);
            files = dir(filepath);
            if isempty(files)
                error('No audio files matching %s found',filepath)
            end
            for i = 1:length(files)
                filepath = sprintf('%s/%s', folder, files(i).name);
                fprintf('Reading audio file %s\n',filepath);
                self.convertUTF8(filepath);
                % Parse the data ignoring time stamps
                fid = fopen(filepath);
                parsed = textscan(fid,'%f','commentStyle','2016-','delimiter',delimeter,'headerLines',1);
                partdata = parsed{1};
                fclose(fid);
                % Store the start point, and part data
                self.audioDelimeters(end+1) = length(self.audioTimeSeries)+1;
                self.audioTimeSeries = vertcat(self.audioTimeSeries, partdata);
            end
        end
              
        function loadVibration(self,tool,folder)
            % Load the vibration data
            delimeter = ' ';
            filepath = sprintf('%s/accel_T%02i*',folder,tool);
            files = dir(filepath);
            if isempty(files)
                error('No acceleration files matching %s found',filepath)
            end
            for i = 1:length(files)
                filepath = sprintf('%s/%s', folder, files(i).name);
                fprintf('Reading vibration file %s\n',filepath);
                self.convertUTF8(filepath);
                % Parse the data ignoring time stamps
                fid = fopen(filepath);
                parsed = textscan(fid,'%f %f %f','commentStyle','2016-','delimiter',delimeter,'headerLines', 1);           
                partdata = [parsed{1} parsed{2} parsed{3}];
                fclose(fid);
                % Store the start point, and part data
                self.vibrationDelimeters(end+1) = length(self.vibrationTimeSeries)+1;
                self.vibrationTimeSeries = vertcat(self.vibrationTimeSeries, partdata);
            end
        end
        
        function discardExtraData(self,tool)
        % Discard data points at any point in the recording
        % The discard Map contains the alternating start point/end times to discard
        % When selecting times to discard, comment out the relevant discard(*) line
        
            discard = containers.Map(0,[0 0]);
            discard(11) = [398,401,   600,604,   1800,1804];
            discard(12) = [206,225,   572,576];
            discard(17) = [582,584];
            discard(18) = [542,546,679,690];
            discard(19) = [1116,1120];
            discard(21) = [190,192,  295,300,  420,421,  437,442,  896,900   1394,1395];
            
            discard(25) = [0,6,   302,326,  622,640];
            discard(26) = [0,20,  400,426,  801,810,  1170,1185];
            
            % Add more items here
            
            if (discard.isKey(tool))
                times = discard(tool);
                removeAudio = zeros(size(self.audioTime));
                removeAccel = zeros(size(self.vibrationTime));
                for i=1:2:length(times)
                    start = times(i);
                    endpoint = times(i+1);
                    removeAudio = removeAudio | (start < self.audioTime     & self.audioTime     < endpoint);
                    removeAccel = removeAccel | (start < self.vibrationTime & self.vibrationTime < endpoint);
                end
                self.audioTimeSeries(removeAudio) = [];
                self.vibrationTimeSeries(removeAccel,:) = [];
            end
        end
        
        function convertUTF8(self,filename)
            % Convert file to a readable format
            if (isunicode(filename))
                fprintf('Converting file to UTF8 %s\n',filename);
                unicode2ascii(filename,filename);
            end
        end
    end
end


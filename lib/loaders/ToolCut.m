classdef ToolCut < handle
    % Holds information about a specific cut that was performed by the 
    % milling machine. Intialized with time series of the cut.
    %
    % Can return:
    %   DFT for the cut
    %   DWT for the cut 
    %   expectedOperation: The estimated operation being performed
    %   actualOperation: The actual operation being performed
    
	properties
        partNum; % Number of the part that included this cut
        toolNum; % Number of the tool that made this cut
        toolwear; % Value from 0 to 1 representing tool wear
        
        fourier;
        wavelet;
        expectedOperation; % [0,1,2] -> [Air,Face,Climb]
        actualOperation;   % [0,1,2] -> [Air,Conventional,Climb]

        audioTime;
        audioTimeSeries = []; % Time series matrix
        audioSampleRate = 8000; % Hz
        audioFourier = [];
        
        vibrationTime
        vibrationTimeSeries = []; % Time series matrix
        vibrationSampleRate = 1000; % Hz
    end
    
    methods
        % Constructor
        function self = ToolCut(audioTimeSeries,vibrationTimeSeries,toolNum,partNum,toolWear,cutAction)
            self.audioTimeSeries = audioTimeSeries;
            self.vibrationTimeSeries = vibrationTimeSeries;
            self.partNum = partNum;
            self.toolNum = toolNum;
            self.toolwear = toolWear;
            self.actualOperation = cutAction;
            self.updateAudioSampleRate();
            self.audioTime = (0:length(self.audioTimeSeries)-1)/self.audioSampleRate;
            self.vibrationTime = (0:length(self.vibrationTimeSeries)-1)/self.vibrationSampleRate;
            self.calculateVibrationDFT();
            self.calculateAudioDFT();
        end

        % Normalize a fft for graphing
        function power = normalize(self,power)
           %size = length(power);
           %n = round(size / min(100,size/5));
           %power = smooth(power,n)/mean(power);   
        end
        
        % Plot the Audio Fourier Spectrum on the current figure
        % Direction should be one of [1,2,3]
        % Color should be a rgb three tuple eg [0,0,0]
        function plotAudioFourierSpectrum(self,color)
           freq = self.audioFourier.freq;
           power = self.audioFourier.power;
           plot(freq, self.normalize(sqrt(power)),'Color',color);
           xlabel('Frequency [Hz]');
           ylabel('Fourier Amplitude)')
           title('Fourier Spectrum for Audio Signal');
        end
        
        % Plot the Fourier Spectrum on the current figure
        % Direction should be one of [1,2,3]
        % Color should be a rgb three tuple eg [0,0,0]
        function plotFourierSpectrum(self,direction,color)
           freq = self.fourier.freq(direction,:);
           power = self.fourier.power(direction,:);
           plot(freq, self.normalize(sqrt(power)),'Color',color);
           xlabel('Frequency [Hz]');
           ylabel('Fourier Amplitude)')
           title('Fourier Spectrum');
        end
              
        % Plot the Fourier Power Spectrum on the current figure
        % Direction should be one of [1,2,3]
        % Color should be a rgb three tuple eg [0,0,0]
        function plotFourierPowerSpectrum(self,direction,color)
           freq = self.fourier.freq(direction,:);
           power = self.fourier.power(direction,:);
           plot(freq, self.normalize(power),'Color',color);
           xlabel('Frequency [Hz]');
           ylabel('Power')
           title('Fourier Power Spectrum');
        end
                        
        % Plot the LOG Fourier Power spectrum on the current figure
        % Direction should be one of [1,2,3]
        % Color should be a rgb three tuple eg [0,0,0]
        function plotLogPowerSpectrum(self,direction,color)
           freq = self.fourier.freq(direction,:);
           power = self.fourier.power(direction,:);
           plot(freq, log(power),'Color',color);
           xlabel('Frequency [Hz]');
           ylabel('Log(Power)')
           title('Log Power spectrum');
        end
        
        % Plot the DWT on the current figure
        % Direction should be one of [1,2,3]
        % Color should be a rgb three tuple eg [0,0,0]
        function plotWavelet(self,direction,color)
           freq = self.wavelet.freq(direction,:);
           power = self.wavelet.power(direction,:);
           plot(freq, power,'Color',color);
           xlabel('Frequency [Hz]');
           ylabel('Power')
           title('Wavelet spectrum');
        end
        
        % Calculate the DFT for vibration
        % Welch's Method is used to reduce variance in the spectra
        % The frequency domain is discretized into [0,500] Hz frequency intervals
        function calculateVibrationDFT(self)       
            minfreq = 4; %Hz
            ndirections = 3;
            timeSeries = self.vibrationTimeSeries;
            
            % Precalculate frequencies to allocate
            n = 2^9;
            npoints = n/2+1;
           
            % Pad the signal with zeros and scale appropriately
            originalLength = length(timeSeries);
            timeSeries = zeroPad(timeSeries,n);
            scaleFactor = length(timeSeries)/originalLength;
            
            % Preallocate dft matrices
            self.fourier.freq = zeros(ndirections,npoints);
            self.fourier.power = zeros(ndirections,npoints);

            for i=1:ndirections
                accel = timeSeries(:,i);

                % Perform fft transform
                [energy,freq] = pwelch(accel,hanning(n),[],[],self.vibrationSampleRate);
                
                % Discard the low frequency values
                energy(freq<minfreq) = mean(energy(freq>minfreq));
                
                % Compute the power by dividing the energy by the total unpadded duration
                power = scaleFactor*energy; % Watts
                
                self.fourier.freq(i,:) = freq;
                self.fourier.power(i,:) = power;
            end
            % Calculate the fourth direction
            self.fourier.freq(4,:) = sum(self.fourier.freq)/3;
            self.fourier.power(4,:) = sum(self.fourier.power)/3;
            fprintf('%i points in vibration spectra',length(self.fourier.power(4,:)));
        end
        
        
        
        % Calculate the DFT for vibration
        % If n is specified then the time series will be zero padded to n points
        function calculateAudioDFT(self)       
            % Precalculate frequencies to allocate
            n = 2^9;
            minfreq = 10; %Hz
            
            % Precalculate frequencies to allocate
            audio = self.audioTimeSeries;

            % Pad the signal with zeros and scale appropriately
            originalLength = length(audio);
            audio = zeroPad(audio,n);
            scaleFactor = length(audio)/originalLength;
            
            % Perform fft transform
            [energy,freq] = pwelch(audio,hanning(n),[],[],self.audioSampleRate);

            % Discard the low frequency values
            energy(freq<minfreq) = mean(energy(freq>minfreq));

             % Compute the power by dividing the energy by the total unpadded duration
            power = scaleFactor*energy; % Watts
            
            % Compute the power by dividing the energy by the total unpadded duration
            %time = length(self.audioTimeSeries)/self.audioSampleRate; % seconds
            %power = energy/time; % Watts
           
            self.audioFourier.freq = freq;
            self.audioFourier.power = power;
        end
        
        function updateAudioSampleRate(self)
            % Update the audio sample rate if needed
            audioSample = self.vibrationSampleRate*length(self.audioTimeSeries)/length(self.vibrationTimeSeries);
            audioSample = round(audioSample,-3);
            if (audioSample~=self.audioSampleRate)
                fprintf('Audio sample rate is %iHz ',audioSample)
                self.audioSampleRate = audioSample;
            end
        end
    end
end

    
% Zero pad the timeseries to length n 
function ts = zeroPad(ts,n)
    if size(ts,1)>size(ts,2)
        nzeros = n-size(ts,1);
        zero = zeros(nzeros,size(ts,2));
        ts = vertcat(ts,zero);
    else
        nzeros = n-size(ts,2);
        zero = zeros(size(ts,1),nzeros);
        ts = horzcat(ts,zero);
    end
end
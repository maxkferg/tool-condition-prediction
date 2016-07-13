function plotRawData()
    % Plot the raw timeseries acceleration and vibration
    toolNums = [21];%,2,3];
    tools = loadTools(toolNums);
    for tool = tools
        %plotRawVibrationTimeSeries(tool);
        %plotRawAudioTimeSeries(tool);
        %plotNormalizedVibrationTimeSeries(tool);
        %plotCutBoundaries(tool);
        plotEnvelope(tool);
        %plotFrequencyDomain(tool);
        %plotFourierRotationSpeed(tool);
        %plotFourierAudio(tool);
        %plotFourierEnergy(tool);
        %plotGaussianConvolution(tool);
    end
end


function plotFourierVibration(tool)
    cut = tool.ToolCuts(57);
    figure; hold on;
    width = 700; aspect = 1.7;
    plot(cut.fourier.freq(1,:), smooth(log(cut.fourier.power(1,:)),30), 'Color',accentColor(1), 'linewidth',1);
    set(gcf, 'Position', [0 0 width width/aspect]);
    title('Vibration Power Spectrum');
    xlabel('Frequency [Hz]');
    ylabel('Log(Power)');
    set(gca,'FontSize',14);
    legend({'x'});
    ylim([5,12]);
end



function plotFrequencyDomain(tool)
% Choose a random cut and plot the Fourier transform
% Plot three vibration and one audio on the same plot
    cut = tool.ToolCuts(57);
    figure; hold on;
    width = 700; aspect = 1.7;
    plot(cut.fourier.freq(1,:), smooth(log(cut.fourier.power(1,:)),100), 'Color',accentColor(1), 'linewidth',1);
    set(gcf, 'Position', [0 0 width width/aspect]);
    title('Vibration Power Spectrum');
    xlabel('Frequency [Hz]');
    ylabel('Log(Power)');
    set(gca,'FontSize',14);
    legend({'x'});
    ylim([5,12]);
    
    figure; hold on;
    plot(cut.fourier.freq(1,:), smooth(log(cut.fourier.power(1,:)),100), 'Color',accentColor(1), 'linewidth',1);
    plot(cut.fourier.freq(2,:), smooth(log(cut.fourier.power(2,:)),100), 'Color',accentColor(2), 'linewidth',1);
    set(gcf, 'Position', [0 0 width width/aspect]);
    title('Vibration Power Spectra');
    xlabel('Frequency [Hz]');
    ylabel('Log(Power)');
    set(gca,'FontSize',14);
    legend({'x','y'});
    ylim([5,12]);
    
    figure; hold on;
    plot(cut.fourier.freq(1,:), smooth(log(cut.fourier.power(1,:)),100), 'Color',accentColor(1), 'linewidth',1);
    plot(cut.fourier.freq(2,:), smooth(log(cut.fourier.power(2,:)),100), 'Color',accentColor(2), 'linewidth',1);
    plot(cut.fourier.freq(3,:), smooth(log(cut.fourier.power(3,:)),100), 'Color',accentColor(3), 'linewidth',1);
    set(gcf, 'Position', [0 0 width width/aspect]);
    title('Vibration Power Spectra');
    xlabel('Frequency [Hz]');
    ylabel('Log(Power)');
    set(gca,'FontSize',14);
    legend({'x','y','z'});
    ylim([5,12]);
end


function plotFourierRotationSpeed(tool)
% Plot the fourier transform and mark the rotation speed
    figure; hold on;
    width = 700; aspect = 1.7;
    cut = tool.ToolCuts(57);
    plot(cut.fourier.freq(1,:), smooth(log(cut.fourier.power(1,:)),100), 'Color', 1.15*accentColor(1), 'linewidth',1);
    plot(cut.fourier.freq(2,:), smooth(log(cut.fourier.power(2,:)),100), 'Color', 1.15*accentColor(2), 'linewidth',1);
    plot(cut.fourier.freq(3,:), smooth(log(cut.fourier.power(3,:)),100), 'Color', 1.00*accentColor(3), 'linewidth',1);
    set(gcf, 'Position', [0 0 width width/aspect]);
    title('Milling Machine Rotation Speed');
    xlabel('Frequency [Hz]');
    ylabel('Log(Power)');
    legend({'x','y','z'});
    ylim([5,12]);
    vline(3080/60*4,'--');
    set(gca,'FontSize',14);
end


function plotFourierAudio(tool)
% Plot the fourier transform and mark the rotation speed
    figure; hold on;
    width = 700; aspect = 1.7;
    cut = tool.ToolCuts(57);
    plot(cut.audioFourier.freq, smooth(log(cut.audioFourier.power),20), 'Color', accentColor(1), 'linewidth',1);   
    set(gcf, 'Position', [0 0 width width/aspect]);
    title('Audio Power Spectra');
    xlabel('Frequency [Hz]');
    ylabel('Log(Power)');
    legend({'audio'});
    ylim([7,10]);
    set(gca,'FontSize',14);
end


function plotGaussianConvolution(tool)
% Plot the Gaussian curves on top of the audio and vibration spectra
    n = 3;
    plotFourierAudio(tool);
    handle = gca();
    ylim([0,14]);
    xlim([10,handle.XLim(2)]);
    
    frequency = linspace(handle.XLim(1),handle.XLim(2),1000);
    mus = linspace(1,max(frequency),n);
    sd = 0.7*(max(frequency)-min(frequency))/n;
    filters = {'Low-Pass Audio Filter','Band-Pass Audio Filter','High-Pass Audio Filter'};
    
    for i=1:length(mus)
        mu = mus(i);
        y = normpdf(frequency,mu,sd);
        y = 0.9 * y/max(y) * 12;
        %plotFourierAudio(tool);
        plot(frequency,y,'--');
        ylim([0,14]);
        xlim([10,handle.XLim(2)]);
        legend({'Power Spectrum','Bandwidth Filter'})
        %title(filters{i})
        set(gcf,'PaperPositionMode', 'auto');
        %saveas(gcf,sprintf('presentation/images/feature-%s.png',filters{i}))
        hold on;
    end
    %set(gca,'xscale','log');
    %saveas(gcf,sprintf('presentation/images/feature-%s-log.png',filters{i}))
end


function plotFourierEnergy(tool)
    % Plot the fourier spectrum energy in four colors
    cut = tool.ToolCuts(57);
    directions = {'X','Y','Z'};
    for i=1:3
        figure; hold on;
        %width = 700; aspect = 1.7;
        direction = directions{i};
        cut = tool.ToolCuts(57);
        f = cut.fourier.freq(i,:);
        p = cut.fourier.power(i,:);
        c = accentColor(i);
        area(f, smooth(log(p),80), 'Facecolor', c);
        plot(f, smooth(log(p),80),'Color',c,'lineWidth',2);
        alpha(0.05)
        %set(gcf, 'Position', [0 0 width width/aspect]);
        title(sprintf('Vibration Energy in %s Direction',direction));
        xlabel('Frequency [Hz]');
        ylabel('Power Spectral Density');
        ylim([6,12]);
        set(gca,'FontSize',22);
        saveas(gcf,sprintf('presentation/images/Power %s.png',direction));
    end
    figure; hold on;
    f = cut.audioFourier.freq;
    p = cut.audioFourier.power;
    c = accentColor(4);
    area(f, smooth(log(p),80), 'Facecolor', c);
    plot(f, smooth(log(p),80),'Color',c,'lineWidth',2);
    alpha(0.05)
    %set(gcf, 'Position', [0 0 width width/aspect]);
    title('Audio Energy');
    xlabel('Frequency [Hz]');
    ylabel('Log(Power)');
    ylim([6,12]);
    set(gca,'FontSize',22);
    saveas(gcf,sprintf('presentation/images/Power %s.png','audio'));
end



function plotNormalizedVibrationTimeSeries(tool)
    % Plot the raw vibration time series
    figure; hold on;
    plot(tool.vibrationTime, tool.vibrationTimeSeries(:,1), 'Color',accentColor(1));
    plot(tool.vibrationTime, tool.vibrationTimeSeries(:,2), 'Color',accentColor(2));
    plot(tool.vibrationTime, tool.vibrationTimeSeries(:,3), 'Color',accentColor(3));
    title('Normalized Vibration Time Series');
    xlabel('Time [s]');
    ylabel('Acceleration [g]');
    set(gca,'FontSize',14);
    legend({'x','y','z'});
end


function plotRawVibrationTimeSeries(tool)
    % Plot the raw vibration time series
    figure; hold on;
    d = [0.2229   -0.3956    0.9616]; 
    s = [0.0373    0.0190    0.0080];
    plot(tool.vibrationTime, d(1)+s(1)*tool.vibrationTimeSeries(:,1), 'Color',accentColor(1));
    plot(tool.vibrationTime, d(2)+s(2)*tool.vibrationTimeSeries(:,2), 'Color',accentColor(2));
    plot(tool.vibrationTime, d(3)+s(3)*tool.vibrationTimeSeries(:,3), 'Color',accentColor(3));
    title('Raw Vibration Time Series');
    xlabel('Time [s]');
    ylabel('Acceleration [g]');
    set(gca,'FontSize',14);
    legend({'x','y','z'});
end


function plotRawAudioTimeSeries(tool)
    % Plot the raw vibration time series
    figure; hold on;
    s = 241430; 
    plot(tool.audioTime, s*tool.audioTimeSeries, 'color',accentColor(1));
    title('Raw Audio Time Series');
    xlabel('Time [s]');
    ylabel('Intensity');
    set(gca,'FontSize',14);
end


function plotCutBoundaries(tool)
% Plot the boundaries between each cut
    power = rms(tool.vibrationTimeSeries,2);
    env = envelope(power,100,'peak');
    env = smooth(env,500);

    figure; hold on;
    plot(tool.vibrationTime, env, 'color',accentColor(1));
    vline(tool.cutBoundaries,'color',accentColor(8));
    title('Assigning Cut Boundaries');
    xlabel('Time [s]')
    ylabel('Vibration Amplitude');
    set(gca,'FontSize',14);
    set(gca,'YTick', 0:5);
end


function plotEnvelope(tool)
% Plot the boundaries between each cut
    power = rms(tool.vibrationTimeSeries,2);
    env = envelope(power,100,'peak');
    env = smooth(env,500);

    figure; hold on;
    plot(tool.vibrationTime, env, 'color',accentColor(1));
    title('Vibration Amplitude Envelope');
    xlabel('Time [s]')
    ylabel('Vibration Amplitude');
    xlim([0,200]);
    ylim([0,4]);
    set(gca,'FontSize',14);
    set(gca,'YTick', 0:5);
   
    
    % Plot the raw data
    figure; hold on;
    plot(tool.vibrationTime, power, 'color',accentColor(1));
    title('RMS Vibration Amplitude');
    xlabel('Time [s]')
    ylabel('Vibration Amplitude');
    xlim([0,200]);
    ylim([0,4]);
    set(gca,'FontSize',14);
    set(gca,'YTick', 0:5);
end






function tools = loadTools(toolNums)
% Load all of the cuts from the data directory    
    % Cache tool objects
    for tool=toolNums
        cache = sprintf('data/cache/tool_%i.mat',tool);
        if ~exist(cache, 'file')
            data = ToolCondition(tool);
            save(cache,'data');
        end
    end
    
    % Load tool object
    tools = [];
    for i=1:length(toolNums)
        cache = sprintf('data/cache/tool_%i.mat',toolNums(i));
        fprintf('Loading tool %i data from cache\n',toolNums(i));
        cached = load(cache);
        tools = [tools cached.data];
    end  
end


function values = log2space(min,max,n)
    % Create a list of n numbers that increase exponentially
    % log2space(2,16,4) -> [2,4,8,16]
    base = 2;
    bases = base*ones(1,n);
    minExp = log(min)/log(base);
    maxExp = log(max)/log(base);
    powers = linspace(minExp,maxExp,n);
    values = bases.^powers;
end


function color = darken(color)
    % Darken a color by 50%
    color = 0.4*color;
    color = min(color,[1,1,1]);
end
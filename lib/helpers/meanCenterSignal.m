function signal = meanCenterSignal(signal,sampleRate)
% Make the moving-average of the signal zero
% Signal must be a column or row vector
    if size(signal,1)~=1
        signal = transpose(signal);
    end

    if (size(signal,1)~=1)
        error('Signal must be a vector');
    end

    start = 1;
    last = sampleRate;
    while (start<length(signal))
        if (last>length(signal))
            last = length(signal);
        end
        signal(start:last) = signal(start:last) - mean(signal(start:last));
        start = start+sampleRate;
        last = last+sampleRate;
    end
end
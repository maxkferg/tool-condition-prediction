function signal = normalizeSignal(signal)
% Make the mean absolute amplitude of the signal equal to one
    factor = mean(abs(signal));
    signal = signal/factor;
end
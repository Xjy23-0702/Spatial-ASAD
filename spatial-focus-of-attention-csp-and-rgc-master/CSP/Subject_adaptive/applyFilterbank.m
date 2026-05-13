function eegFiltered = applyFilterbank(eeg, bands, fs)
% APPLYFILTERBANK Apply bandpass filterbank to EEG data
%
% Input:
%   eeg: channel x time x trial (or channel x time x band x trial for existing)
%   bands: 2 x nBands matrix of frequency bounds
%   fs: sampling rate
%
% Output:
%   eegFiltered: channel x band x time x trial

if ndims(eeg) == 3
    [nCh, nTime, nTrials] = size(eeg);
    nBands = size(bands, 2);
    eegFiltered = zeros(nCh, nBands, nTime, nTrials);
    
    for band = 1:nBands
        d = designfilt('bandpassiir', 'FilterOrder', 8, ...
            'HalfPowerFrequency1', bands(1, band), ...
            'HalfPowerFrequency2', bands(2, band), ...
            'SampleRate', fs);
        
        % Apply filter to each channel and trial
        for ch = 1:nCh
            for tr = 1:nTrials
                eegFiltered(ch, band, :, tr) = filtfilt(d, squeeze(eeg(ch, :, tr)));
            end
        end
    end
else
    eegFiltered = eeg; % Already filtered
end
end
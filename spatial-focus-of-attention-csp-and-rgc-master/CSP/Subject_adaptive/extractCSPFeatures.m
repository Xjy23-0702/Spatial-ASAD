function [feat, windowedLabels] = extractCSPFeatures(Y, labels, windowSamples, trialLength)
% EXTRACTCSPFEATURES Extract log-energy features from CSP-filtered data
%
% Input:
%   Y: nFilters x band x time x trial
%   labels: original trial labels (nTrials x 1)
%   windowSamples: number of samples per decision window
%   trialLength: length of each trial in samples
%
% Output:
%   feat: feature matrix (nWindows x (nFilters*nBands))
%   windowedLabels: labels for each window

[nFilters, nBands, nTime, nTrials] = size(Y);

% Reshape to combine filters and bands
Y_reshaped = reshape(Y, nFilters * nBands, nTime, nTrials);

% Segment into windows
nWindowsPerTrial = floor(nTime / windowSamples);
nTotalWindows = nTrials * nWindowsPerTrial;

feat = zeros(nTotalWindows, nFilters * nBands);
windowedLabels = zeros(nTotalWindows, 1);

winIdx = 1;
for tr = 1:nTrials
    for win = 1:nWindowsPerTrial
        startIdx = (win-1)*windowSamples + 1;
        endIdx = win*windowSamples;
        
        % Extract window
        Y_window = Y_reshaped(:, startIdx:endIdx, tr);
        
        % Compute log-energy
        feat(winIdx, :) = log(sum(Y_window.^2, 2))';
        
        % Assign label (same as trial label)
        windowedLabels(winIdx) = labels(tr);
        
        winIdx = winIdx + 1;
    end
end
end
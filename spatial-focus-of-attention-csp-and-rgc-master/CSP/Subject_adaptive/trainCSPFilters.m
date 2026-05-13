function [CSP, Y] = trainCSPFilters(eeg, labels, params)
% TRAINCSPFILTERS Train CSP filters on given data
%
% Input:
%   eeg: channel x band x time x trial
%   labels: trial labels (vector)
%   params: parameter structure
%
% Output:
%   CSP: structure with W, score, traceratio
%   Y: filtered data (channel_after_CSP x band x time x trial)

nBands = size(params.filterbank.bands, 2);
CSP.W = [];
CSP.score = [];
CSP.traceratio = [];

Y = [];

for band = 1:nBands
    % Extract data for this band
    X_band = squeeze(eeg(:, band, :, :)); % channel x time x trial
    
    % Train CSP
    [W, score, traceratio] = trainCSP(X_band, labels, ...
        params.csp.npat, params.csp.optmode, ...
        params.csp.heuristicPatSel, params.cov.method);
    
    CSP.W = cat(3, CSP.W, W);
    CSP.score = cat(3, CSP.score, score);
    CSP.traceratio = cat(3, CSP.traceratio, traceratio);
    
    % Filter data
    Y_band = tmprod(X_band, W', 1); % nFilters x time x trial
    Y = cat(4, Y, Y_band);
end

% Reorder: nFilters x band x time x trial
Y = permute(Y, [1, 4, 2, 3]);
end
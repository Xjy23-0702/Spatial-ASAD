function Y = applyCSPFilters(eeg, CSP)
% APPLYCSPFILTERS Apply pre-trained CSP filters to new data
%
% Input:
%   eeg: channel x band x time x trial
%   CSP: structure with W field (channel x nFilters x band)
%
% Output:
%   Y: nFilters x band x time x trial

nBands = size(eeg, 2);
Y = [];

for band = 1:nBands
    X_band = squeeze(eeg(:, band, :, :)); % channel x time x trial
    W_band = CSP.W(:, :, band); % channel x nFilters
    
    Y_band = tmprod(X_band, W_band', 1); % nFilters x time x trial
    Y = cat(4, Y, Y_band);
end

Y = permute(Y, [1, 4, 2, 3]);
end
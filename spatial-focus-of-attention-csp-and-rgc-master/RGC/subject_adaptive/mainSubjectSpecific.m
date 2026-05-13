%% MAIN SUBJECT-ADAPTIVE RIEMANNIAN CLASSIFIER
% Subject-adaptive RGC with pretrain-finetune paradigm
% - Pretrain: Use M-1 subjects to compute class-specific Riemannian means
% - Finetune: Update class means using target subject's 20% data
% - Test: Evaluate on remaining 80% of target subject data
%
% Reference: Adapted from Geirnaert et al. ICASSP 2021

clear; close all;

% Set random seed for reproducibility
rng(42);

%% Setup: parameters
params.dataset = 'das-2016';
params.subjects = 1:16;
params.windowLengths = [60,30,20,10,5,2,1];
params.save = true;
params.saveName = 'RGC-64ch-beta-subject-adaptive';

% Subject-adaptive parameters
params.subjectAdaptive.finetuneRatio = 0.2; % 20% for finetuning
params.subjectAdaptive.biasUpdateOnly = true; % true: only update Riemannian means

% preprocessing
params.preprocessing.normalization = false;
params.preprocessing.rereference = 'none';
params.preprocessing.eegChanSel = [];

% bandpass filter
params.filterbands = [12;30]; % beta band

% covariance construction
params.cov.method = 'lwcov';

% riemannian mean parameters
params.riem.method = 'log-euclidean'; % 'riemannian' or 'log-euclidean'
params.riem.epsilon = 1e-12;

% classification parameters
params.riem.class.method = 'mdrm'; % 'mdrm' or 'tsm'
params.class.method = 'svm'; % only used for 'tsm'
params.class.kernel = 'linear';
params.class.optimized = false;
params.class.arg = {'Prior','uniform', 'KernelFunction', 'linear', 'Verbose', 0, 'Standardize', false};

%% Results storage
results = struct;
results.testacc = zeros(length(params.subjects), length(params.windowLengths));
results.finetuneacc = zeros(length(params.subjects), length(params.windowLengths));

%% Load all subjects data first
fprintf('Loading all subjects data...\n');
allData = cell(length(params.subjects), 1);
allLabels = cell(length(params.subjects), 1);
allFs = [];
allTrialLength = [];

for subjIdx = 1:length(params.subjects)
    subj = params.subjects(subjIdx);
    [eeg, attendedEar, fs, trialLength] = loadData(params.dataset, subj, params.preprocessing);
    
    % Apply bandpass filter
    d = designfilt('bandpassiir', 'FilterOrder', 8, ...
        'HalfPowerFrequency1', params.filterbands(1), ...
        'HalfPowerFrequency2', params.filterbands(2), ...
        'SampleRate', fs);
    eeg = permute(filtfilt(d, permute(eeg, [2,1,3])), [2,1,3]);
    
    allData{subjIdx} = eeg;
    allLabels{subjIdx} = attendedEar;
    allFs = fs;
    allTrialLength = trialLength;
    
    fprintf('Subject %d: %d trials, labels: ', subj, length(attendedEar));
    disp(unique(attendedEar)');
end

%% Loop over test subjects
for testSubj = 1:length(params.subjects)
    fprintf('\n%s\n*** Processing subject %d (adaptive mode) ***\n%s\n', ...
        repmat('-',1,50), params.subjects(testSubj), repmat('-',1,50));
    
    % Split target subject data into finetune and test sets
    targetData = allData{testSubj};
    targetLabels = allLabels{testSubj};
    
    % Get actual class values (should be 1 and 2)
    unique_labels = unique(targetLabels);
    class1_val = unique_labels(1);
    class2_val = unique_labels(2);
    
    nTrials = size(targetData, 3);
    
    idxClass1 = find(targetLabels == class1_val);
    idxClass2 = find(targetLabels == class2_val);
    
    fprintf('Original distribution: class%d=%d, class%d=%d\n', ...
        class1_val, length(idxClass1), class2_val, length(idxClass2));
    
    % Ensure both classes have samples
    if length(idxClass1) == 0 || length(idxClass2) == 0
        warning('Subject %d has only one class! Skipping...', params.subjects(testSubj));
        continue;
    end
    
    nFinetuneClass1 = max(1, round(length(idxClass1) * params.subjectAdaptive.finetuneRatio));
    nFinetuneClass2 = max(1, round(length(idxClass2) * params.subjectAdaptive.finetuneRatio));
    
    nFinetuneClass1 = min(nFinetuneClass1, length(idxClass1));
    nFinetuneClass2 = min(nFinetuneClass2, length(idxClass2));
    
    finetuneIdxClass1 = idxClass1(randperm(length(idxClass1), nFinetuneClass1));
    finetuneIdxClass2 = idxClass2(randperm(length(idxClass2), nFinetuneClass2));
    finetuneIdx = [finetuneIdxClass1; finetuneIdxClass2];
    testIdx = setdiff(1:nTrials, finetuneIdx);
    
    finetuneData = targetData(:, :, finetuneIdx);
    finetuneLabels = targetLabels(finetuneIdx);
    testData = targetData(:, :, testIdx);
    testLabels = targetLabels(testIdx);
    
    fprintf('Finetune set: %d trials (class%d=%d, class%d=%d)\n', ...
        length(finetuneLabels), class1_val, sum(finetuneLabels==class1_val), ...
        class2_val, sum(finetuneLabels==class2_val));
    fprintf('Test set: %d trials (class%d=%d, class%d=%d)\n', ...
        length(testLabels), class1_val, sum(testLabels==class1_val), ...
        class2_val, sum(testLabels==class2_val));
    
    %% PRETRAIN: Train subject-independent model using all other subjects
    fprintf('\n--- Pretraining using other subjects ---\n');
    
    pretrainIndices = setdiff(1:length(params.subjects), testSubj);
    
    % Combine data from all pretrain subjects
    pretrainData = [];
    pretrainLabels = [];
    
    for pretrainIdx = 1:length(pretrainIndices)
        subjIdx = pretrainIndices(pretrainIdx);
        subjData = allData{subjIdx};
        subjLabels = allLabels{subjIdx};
        
        if isempty(pretrainData)
            pretrainData = subjData;
            pretrainLabels = subjLabels;
        else
            pretrainData = cat(3, pretrainData, subjData);
            pretrainLabels = [pretrainLabels; subjLabels];
        end
    end
    
    fprintf('Pretrain dataset: %d trials (class%d=%d, class%d=%d)\n', ...
        length(pretrainLabels), class1_val, sum(pretrainLabels==class1_val), ...
        class2_val, sum(pretrainLabels==class2_val));
    
    %% For each decision window length
    for w = 1:length(params.windowLengths)
        windowSamples = params.windowLengths(w) * allFs;
        nWindowsPerTrial = floor(allTrialLength / windowSamples);
        
        if nWindowsPerTrial == 0
            fprintf('Window length %ds: Not enough data, skipping\n', params.windowLengths(w));
            continue;
        end
        
        % Segment pretrain data
        [P_pretrain, labels_pretrain_win] = segmentAndCov(pretrainData, pretrainLabels, windowSamples, nWindowsPerTrial, params);
        
        % Segment finetune data
        [P_finetune, labels_finetune_win] = segmentAndCov(finetuneData, finetuneLabels, windowSamples, nWindowsPerTrial, params);
        
        % Segment test data
        [P_test, labels_test_win] = segmentAndCov(testData, testLabels, windowSamples, nWindowsPerTrial, params);
        
        % Skip if no valid covariance matrices
        if isempty(P_pretrain) || isempty(P_finetune) || isempty(P_test)
            fprintf('Window length %ds: No valid covariance matrices, skipping\n', params.windowLengths(w));
            continue;
        end
        
        %% Train subject-independent model (class means)
        % Compute Riemannian mean per class on pretrain data
        classLabels = unique(labels_pretrain_win);
        Pm_pretrain = zeros(size(P_pretrain,1), size(P_pretrain,2), length(classLabels));
        
        for c = 1:length(classLabels)
            idx = (labels_pretrain_win == classLabels(c));
            Pm_pretrain(:,:,c) = computeRiemannianMean(P_pretrain(:,:,idx), params.riem);
        end
        
        %% FINETUNE: Update class means using target subject's finetune data
        if params.subjectAdaptive.biasUpdateOnly
            % Update each class mean by combining pretrain mean with finetune data
            Pm_finetuned = zeros(size(Pm_pretrain));
            
            for c = 1:length(classLabels)
                idx = (labels_finetune_win == classLabels(c));
                if sum(idx) >= 1
                    % Compute mean of finetune covariances for this class
                    Pm_finetune_class = computeRiemannianMean(P_finetune(:,:,idx), params.riem);
                    
                    % Combine pretrain mean and finetune mean (weighted average)
                    % More weight to pretrain if few finetune samples
                    nPretrain = sum(labels_pretrain_win == classLabels(c));
                    nFinetune = sum(idx);
                    alpha = nFinetune / (nPretrain + nFinetune);
                    
                    % Riemannian interpolation (using log-Euclidean for simplicity)
                    if strcmp(params.riem.method, 'log-euclidean')
                        % Log-Euclidean interpolation
                        logP_pretrain = logm(Pm_pretrain(:,:,c));
                        logP_finetune = logm(Pm_finetune_class);
                        logP_combined = (1-alpha) * logP_pretrain + alpha * logP_finetune;
                        Pm_finetuned(:,:,c) = expm(logP_combined);
                    else
                        % For affine-invariant, use geodesic interpolation
                        % Simplified: just use finetune mean if enough samples
                        if nFinetune >= 3
                            Pm_finetuned(:,:,c) = Pm_finetune_class;
                        else
                            Pm_finetuned(:,:,c) = Pm_pretrain(:,:,c);
                        end
                    end
                else
                    Pm_finetuned(:,:,c) = Pm_pretrain(:,:,c);
                end
            end
        else
            % Alternative: Compute means only from finetune data
            Pm_finetuned = zeros(size(P_finetune,1), size(P_finetune,2), length(classLabels));
            for c = 1:length(classLabels)
                idx = (labels_finetune_win == classLabels(c));
                if sum(idx) >= 1
                    Pm_finetuned(:,:,c) = computeRiemannianMean(P_finetune(:,:,idx), params.riem);
                else
                    Pm_finetuned(:,:,c) = Pm_pretrain(:,:,c);
                end
            end
        end
        
        %% Test on target subject's test data
        predicted_test = zeros(size(P_test,3), 1);
        
        for tr = 1:size(P_test,3)
            dist = zeros(size(Pm_finetuned,3), 1);
            for c = 1:size(Pm_finetuned,3)
                dist(c) = riemannianDist(P_test(:,:,tr), Pm_finetuned(:,:,c));
            end
            [~, predicted_test(tr)] = min(dist);
            % Convert back to original label values
            predicted_test(tr) = classLabels(predicted_test(tr));
        end
        
        % Also evaluate on finetune set for monitoring
        predicted_finetune = zeros(size(P_finetune,3), 1);
        for tr = 1:size(P_finetune,3)
            dist = zeros(size(Pm_finetuned,3), 1);
            for c = 1:size(Pm_finetuned,3)
                dist(c) = riemannianDist(P_finetune(:,:,tr), Pm_finetuned(:,:,c));
            end
            [~, predicted_finetune(tr)] = min(dist);
            predicted_finetune(tr) = classLabels(predicted_finetune(tr));
        end
        
        test_acc = mean(labels_test_win == predicted_test);
        finetune_acc = mean(labels_finetune_win == predicted_finetune);
        
        results.testacc(testSubj, w) = test_acc;
        results.finetuneacc(testSubj, w) = finetune_acc;
        
        fprintf('Window %ds: Finetune Acc = %.2f%%, Test Acc = %.2f%%\n', ...
            params.windowLengths(w), finetune_acc*100, test_acc*100);
    end
    
    % Save intermediate results
    if params.save
        save(['results-',params.dataset,'-',params.saveName], 'results');
    end
end

%% Results aggregation
acc_test_mean = mean(results.testacc, 1);
acc_test_std = std(results.testacc, 0, 1);
acc_finetune_mean = mean(results.finetuneacc, 1);

fprintf('\n%s\n*** FINAL RESULTS ***\n%s\n', repmat('-',1,50), repmat('-',1,50));
fprintf('%-10s | %-15s | %-15s\n', 'Window(s)', 'Finetune Acc', 'Test Acc');
fprintf('%s\n', repmat('-',1,45));
for w = 1:length(params.windowLengths)
    fprintf('%-10d | %-14.2f%% | %-14.2f%% ± %.2f%%\n', ...
        params.windowLengths(w), ...
        acc_finetune_mean(w)*100, ...
        acc_test_mean(w)*100, ...
        acc_test_std(w)*100);
end

results.params = params;
if params.save
    save(['results-',params.dataset,'-',params.saveName], 'results', ...
        'acc_test_mean', 'acc_test_std', 'acc_finetune_mean');
end

%% Helper function: segment data and construct covariance matrices
function [P, windowedLabels] = segmentAndCov(eeg, trialLabels, windowSamples, nWindowsPerTrial, params)
    % eeg: channel x time x trial
    % trialLabels: original trial labels
    
    [nCh, nTime, nTrials] = size(eeg);
    nTotalWindows = nTrials * nWindowsPerTrial;
    
    % Pre-allocate cell array for windowed data
    windowedData = cell(nTotalWindows, 1);
    windowedLabels = zeros(nTotalWindows, 1);
    
    idx = 1;
    for tr = 1:nTrials
        for win = 1:nWindowsPerTrial
            startIdx = (win-1)*windowSamples + 1;
            endIdx = win*windowSamples;
            windowedData{idx} = eeg(:, startIdx:endIdx, tr);
            windowedLabels(idx) = trialLabels(tr);
            idx = idx + 1;
        end
    end
    
    % Construct covariance matrices
    P = zeros(nCh, nCh, nTotalWindows);
    
    if strcmp(params.cov.method, 'lwcov')
        for tr = 1:nTotalWindows
            P(:,:,tr) = lwcov(windowedData{tr}');
            P(:,:,tr) = (P(:,:,tr) + P(:,:,tr)') / 2;
        end
    else
        for tr = 1:nTotalWindows
            P(:,:,tr) = cov(windowedData{tr}');
            P(:,:,tr) = (P(:,:,tr) + P(:,:,tr)') / 2;
        end
    end
end

%% Helper function: Riemannian distance
function d = riemannianDist(P1, P2)
    % Compute Riemannian distance between two covariance matrices
    % d = sqrt(trace(log(P1^(-1/2) * P2 * P1^(-1/2))^2))
    
    P1sqrt = sqrtm(P1);
    P1invsqrt = inv(P1sqrt);
    M = P1invsqrt * P2 * P1invsqrt;
    d = sqrt(trace(logm(M)^2));
    d = real(d); % Ensure real-valued due to numerical errors
end
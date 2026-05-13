%% MAIN SUBJECT-ADAPTIVE FILTERBANK CSP FILTER
% Subject-adaptive FB-CSP filtering with pretrain-finetune paradigm
% - Pretrain: Use M-1 subjects to train general CSP filters
% - Finetune: Train classifier on target subject's 20% data (supervised)
% - Test: Evaluate on remaining 80% of target subject data

clear; close all;

% Set random seed for reproducibility
rng(42);

%% Setup: parameters
params.dataset = 'das-2016';
params.subjects = 1:16;
params.windowLengths = [60,30,20,10,5,2,1];
params.save = true;
params.saveName = 'FB-64ch-beta-subject-adaptive';

% Subject-adaptive parameters
params.subjectAdaptive.finetuneRatio = 0.2; % 20% for finetuning

% preprocessing
params.preprocessing.normalization = true;
params.preprocessing.rereference = 'none';
params.preprocessing.eegChanSel = [];

% filterbank setup
params.filterbank.bands = [12;30]; % beta band

% covariance estimation
params.cov.method = 'lwcov';

% CSP filters
params.csp.npat = 6;
params.csp.optmode = 'ratiotrace';
params.csp.heuristicPatSel = true;

% classification parameters
params.class.method = 'lda';
params.class.optimized = false;
params.class.arg = {'Prior','uniform'};

%% Results storage
results = struct;
results.testacc = zeros(length(params.subjects), length(params.windowLengths));
results.finetuneacc = zeros(length(params.subjects), length(params.windowLengths));

%% Load all subjects data first
fprintf('Loading all subjects data...\n');
allData = cell(length(params.subjects), 1);
allLabels = cell(length(params.subjects), 1);
fs = [];
trialLength = [];

for subjIdx = 1:length(params.subjects)
    subj = params.subjects(subjIdx);
    [eeg, attendedEar, fs, trialLen] = loadData(params.dataset, subj, params.preprocessing);
    
    % 标签已经是 1 和 2，保持不变
    fprintf('Subject %d - label values: ', subj);
    disp(unique(attendedEar));
    
    % Apply filterbank
    [nCh, nTime, nTrials] = size(eeg);
    eegFiltered = zeros(nCh, 1, nTime, nTrials);
    d = designfilt('bandpassiir', 'FilterOrder', 8, ...
        'HalfPowerFrequency1', params.filterbank.bands(1), ...
        'HalfPowerFrequency2', params.filterbank.bands(2), ...
        'SampleRate', fs);
    
    for ch = 1:nCh
        for tr = 1:nTrials
            eegFiltered(ch, 1, :, tr) = filtfilt(d, squeeze(eeg(ch, :, tr)));
        end
    end
    
    allData{subjIdx} = eegFiltered;
    allLabels{subjIdx} = attendedEar;
    trialLength = trialLen;
end

%% Loop over test subjects
for testSubj = 1:length(params.subjects)
    fprintf('\n%s\n*** Processing subject %d (adaptive mode) ***\n%s\n', ...
        repmat('-',1,50), params.subjects(testSubj), repmat('-',1,50));
    
    % Split target subject data into finetune and test sets
    targetData = allData{testSubj};
    targetLabels = allLabels{testSubj};
    
    % 获取实际的标签值（应该是 1 和 2）
    unique_labels = unique(targetLabels);
    class1_val = unique_labels(1);  % 应该是 1
    class2_val = unique_labels(2);  % 应该是 2
    
    nTrials = size(targetData, 4);
    
    idxClass1 = find(targetLabels == class1_val);
    idxClass2 = find(targetLabels == class2_val);
    
    fprintf('Original distribution: class%d=%d, class%d=%d\n', ...
        class1_val, length(idxClass1), class2_val, length(idxClass2));
    
    % 确保两类都有样本
    if length(idxClass1) == 0 || length(idxClass2) == 0
        warning('Subject %d has only one class! Skipping...', params.subjects(testSubj));
        continue;
    end
    
    nFinetuneClass1 = max(1, round(length(idxClass1) * params.subjectAdaptive.finetuneRatio));
    nFinetuneClass2 = max(1, round(length(idxClass2) * params.subjectAdaptive.finetuneRatio));
    
    % 确保不超过总数
    nFinetuneClass1 = min(nFinetuneClass1, length(idxClass1));
    nFinetuneClass2 = min(nFinetuneClass2, length(idxClass2));
    
    finetuneIdxClass1 = idxClass1(randperm(length(idxClass1), nFinetuneClass1));
    finetuneIdxClass2 = idxClass2(randperm(length(idxClass2), nFinetuneClass2));
    finetuneIdx = [finetuneIdxClass1; finetuneIdxClass2];
    testIdx = setdiff(1:nTrials, finetuneIdx);
    
    finetuneData = targetData(:, :, :, finetuneIdx);
    finetuneLabels = targetLabels(finetuneIdx);
    testData = targetData(:, :, :, testIdx);
    testLabels = targetLabels(testIdx);
    
    fprintf('Finetune set: %d trials (class%d=%d, class%d=%d)\n', ...
        length(finetuneLabels), class1_val, sum(finetuneLabels==class1_val), ...
        class2_val, sum(finetuneLabels==class2_val));
    fprintf('Test set: %d trials (class%d=%d, class%d=%d)\n', ...
        length(testLabels), class1_val, sum(testLabels==class1_val), ...
        class2_val, sum(testLabels==class2_val));
    
    %% PRETRAIN: Train CSP filters using all other subjects
    fprintf('\n--- Pretraining CSP filters using other subjects ---\n');
    
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
            pretrainData = cat(4, pretrainData, subjData);
            pretrainLabels = [pretrainLabels; subjLabels];
        end
    end
    
    fprintf('Pretrain dataset: %d trials (class%d=%d, class%d=%d)\n', ...
        length(pretrainLabels), class1_val, sum(pretrainLabels==class1_val), ...
        class2_val, sum(pretrainLabels==class2_val));
    
    % Train subject-independent CSP filters
    X_pretrain = squeeze(pretrainData(:,1,:,:));  % channel x time x trial
    [W, ~, ~] = trainCSP(X_pretrain, pretrainLabels, params.csp.npat, ...
        params.csp.optmode, params.csp.heuristicPatSel, params.cov.method);
    
    % Apply CSP filters to finetune and test data
    Y_finetune = tmprod(squeeze(finetuneData(:,1,:,:)), W', 1);
    Y_test = tmprod(squeeze(testData(:,1,:,:)), W', 1);
    
    %% For each decision window length
    for w = 1:length(params.windowLengths)
        windowSamples = params.windowLengths(w) * fs;
        nWindowsPerTrial = floor(trialLength / windowSamples);
        
        if nWindowsPerTrial == 0
            fprintf('Window length %ds: Not enough data, skipping\n', params.windowLengths(w));
            continue;
        end
        
        % Extract features
        [feat_finetune, labels_finetune_win] = extractFeatures(Y_finetune, finetuneLabels, windowSamples, nWindowsPerTrial);
        [feat_test, labels_test_win] = extractFeatures(Y_test, testLabels, windowSamples, nWindowsPerTrial);
        
        % Train classifier on finetune data (labels are 1 and 2)
        if strcmp(params.class.method, 'lda')
            model = fitcdiscr(feat_finetune, labels_finetune_win, params.class.arg{:});
        else
            model = fitcsvm(feat_finetune, labels_finetune_win, params.class.arg{:});
        end
        
        % Test on target subject's test data
        predicted_test = predict(model, feat_test);
        predicted_finetune = predict(model, feat_finetune);
        
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

%% Helper function: extract features
function [feat, windowedLabels] = extractFeatures(Y, trialLabels, windowSamples, nWindowsPerTrial)
    % Y: nFilters x time x trial
    % trialLabels: original trial labels (1 or 2)
    
    [nFilters, nTime, nTrials] = size(Y);
    nTotalWindows = nTrials * nWindowsPerTrial;
    
    feat = zeros(nTotalWindows, nFilters);
    windowedLabels = zeros(nTotalWindows, 1);
    
    idx = 1;
    for tr = 1:nTrials
        for win = 1:nWindowsPerTrial
            startIdx = (win-1)*windowSamples + 1;
            endIdx = win*windowSamples;
            
            Y_window = Y(:, startIdx:endIdx, tr);
            feat(idx, :) = log(sum(Y_window.^2, 2))';
            windowedLabels(idx) = trialLabels(tr);
            
            idx = idx + 1;
        end
    end
end
%% Section 1: Construct Dataset with 300 Segments
% Load ECG dataset
if exist('ECGData.mat', 'file')
    load('ECGData.mat'); % Ensure ECGData.mat is in the working directory
else
    error('ECGData.mat not found. Please place the file in the working directory or add its path.');
end

fs = 128; % Sampling frequency (Hz)
segment_length = fs * 10; % 10 seconds in samples

% Initialize variables
labels = {'NSR', 'CHF', 'ARR'};
segments_per_class = 2000;
subjects_per_class = 30;

% Initialize arrays (build these dynamically)
NSR_segments = [];
CHF_segments = [];
ARR_segments = [];

% Loop through each label and extract segments
for l = 1:length(labels)
    label = labels{l};
    indices = find(strcmp(ECGData.Labels, label)); % Get indices for the label
    % Use available subjects if fewer than desired
    nSubjects = min(subjects_per_class, length(indices));
    selected_subjects = indices(randperm(length(indices), nSubjects)); % Randomly choose subjects
    
    all_segments = [];
    for j = 1:length(selected_subjects)
        data = ECGData.Data(selected_subjects(j), :);
        nSegPerSubject = ceil(segments_per_class / nSubjects);
        maxStart = length(data) - segment_length + 1;
        if maxStart < 1
            continue; % Skip if insufficient data length
        end
        start_points = randi([1, maxStart], 1, nSegPerSubject);
        for k = 1:length(start_points)
            segment = data(start_points(k):(start_points(k) + segment_length - 1));
            all_segments = [all_segments; segment]; %#ok<AGROW>
            if size(all_segments, 1) >= segments_per_class
                break;
            end
        end
        if size(all_segments, 1) >= segments_per_class
            break;
        end
    end
    
    % Truncate to the desired number of segments
    if size(all_segments,1) > segments_per_class
        all_segments = all_segments(1:segments_per_class, :);
    end
    
    % Assign segments to corresponding variable
    switch label
        case 'NSR'
            NSR_segments = all_segments;
        case 'CHF'
            CHF_segments = all_segments;
        case 'ARR'
            ARR_segments = all_segments;
    end
end

% Save the constructed dataset
save('Selected_ECG_Segments.mat', 'NSR_segments', 'CHF_segments', 'ARR_segments');

%% Section 2 (a): Preprocessing and Visualization
% Load the selected segments
load('Selected_ECG_Segments.mat');

% Baseline wander removal using high-pass filtering
fc = 0.5; % Cutoff frequency for baseline wander
[b, a] = butter(6, fc / (fs / 2), 'high'); % 6th-order Butterworth filter

% Preallocate filtered matrices
NSR_filtered = zeros(size(NSR_segments));
CHF_filtered = zeros(size(CHF_segments));
ARR_filtered = zeros(size(ARR_segments));

% Apply zero-phase filtering
for i = 1:size(NSR_segments, 1)
    NSR_filtered(i, :) = filtfilt(b, a, NSR_segments(i, :)');
end
for i = 1:size(CHF_segments, 1)
    CHF_filtered(i, :) = filtfilt(b, a, CHF_segments(i, :)');
end
for i = 1:size(ARR_segments, 1)
    ARR_filtered(i, :) = filtfilt(b, a, ARR_segments(i, :)');
end

% Plot unfiltered signals (choose a segment index that exists)
chosen_segment = min(50, size(NSR_segments, 1));
figure;
subplot(3, 1, 1);
plot(NSR_segments(chosen_segment, :));
title('NSR - Original ECG');
xlabel('Time (samples)'); ylabel('Amplitude'); grid on;
subplot(3, 1, 2);
plot(CHF_segments(chosen_segment, :));
title('CHF - Original ECG');
xlabel('Time (samples)'); ylabel('Amplitude'); grid on;
subplot(3, 1, 3);
plot(ARR_segments(chosen_segment, :));
title('ARR - Original ECG');
xlabel('Time (samples)'); ylabel('Amplitude'); grid on;
sgtitle('Unfiltered ECG Signals');

% Plot filtered signals
figure;
subplot(3, 1, 1);
plot(NSR_filtered(chosen_segment, :));
title('NSR - Filtered ECG');
xlabel('Time (samples)'); ylabel('Amplitude'); grid on;
subplot(3, 1, 2);
plot(CHF_filtered(chosen_segment, :));
title('CHF - Filtered ECG');
xlabel('Time (samples)'); ylabel('Amplitude'); grid on;
subplot(3, 1, 3);
plot(ARR_filtered(chosen_segment, :));
title('ARR - Filtered ECG');
xlabel('Time (samples)'); ylabel('Amplitude'); grid on;
sgtitle('Filtered ECG Signals');

%% Section 2 (b): Time and Frequency Features

% Define function for Hjorth parameters
hjorth_params = @(signal) [...
    var(diff(signal)) / var(signal), ...  % Mobility
    var(diff(diff(signal))) / var(diff(signal)) ...  % Complexity
];

% Helper function to compute PSD entropy
psd_entropy = @(psd) -sum((psd / max(sum(psd), eps)) .* log2(psd / max(sum(psd), eps) + eps));

% Helper function to count negative peaks below a threshold
count_negative_peaks = @(signal, threshold) sum(findpeaks(-signal) < -threshold);

% Initialize feature matrices
NSR_features = [];
CHF_features = [];
ARR_features = [];

numSegments = size(NSR_filtered, 1);
for i = 1:numSegments
    % Calculate Hjorth parameters
    NSR_hjorth = hjorth_params(NSR_filtered(i, :));
    CHF_hjorth = hjorth_params(CHF_filtered(i, :));
    ARR_hjorth = hjorth_params(ARR_filtered(i, :));
    
    % Calculate PSD and its entropy for each segment
    NSR_psd = abs(fft(NSR_filtered(i, :))).^2;
    CHF_psd = abs(fft(CHF_filtered(i, :))).^2;
    ARR_psd = abs(fft(ARR_filtered(i, :))).^2;
    
    % Normalize the PSDs
    NSR_psd = (NSR_psd - mean(NSR_psd)) ./ max(NSR_psd);
    CHF_psd = (CHF_psd - mean(CHF_psd)) ./ max(CHF_psd);
    ARR_psd = (ARR_psd - mean(ARR_psd)) ./ max(ARR_psd);
    
    NSR_entropy = psd_entropy(NSR_psd);
    CHF_entropy = psd_entropy(CHF_psd);
    ARR_entropy = psd_entropy(ARR_psd);
    
    % Compute LF/HF ratio
    freq = (0:segment_length-1) * (fs / segment_length);
    LF_band = (freq >= 0.1 & freq <= 10);
    HF_band = (freq >= 50 & freq <= 64);
    NSR_LFHF = sum(NSR_psd(LF_band)) / sum(NSR_psd(HF_band));
    CHF_LFHF = sum(CHF_psd(LF_band)) / sum(CHF_psd(HF_band));
    ARR_LFHF = sum(ARR_psd(LF_band)) / sum(ARR_psd(HF_band));
    
    % Count negative peaks (for NSR and ARR)
    threshold = -0.5;
    NSR_neg_peaks = count_negative_peaks(NSR_filtered(i, :), threshold);
    ARR_neg_peaks = count_negative_peaks(ARR_filtered(i, :), threshold);
    
    % Concatenate features:
    % For NSR: use 5 features (Hjorth Mobility, Complexity, LF/HF, Entropy, and Negative Peaks)
    % For CHF: use 4 features (Hjorth Mobility, Complexity, LF/HF, Entropy)
    % For ARR: use 4 features (Hjorth Mobility, LF/HF, Entropy, Negative Peaks)
    NSR_features = [NSR_features; [NSR_hjorth(1), NSR_hjorth(2), NSR_LFHF, NSR_entropy, NSR_neg_peaks]];
    CHF_features = [CHF_features; [CHF_hjorth(1), CHF_hjorth(2), CHF_LFHF, CHF_entropy]];
    ARR_features = [ARR_features; [ARR_hjorth(1), ARR_LFHF, ARR_entropy, ARR_neg_peaks]];
end

% For classification, we use the first 4 features from each group.
% (You may experiment with including the extra NSR feature if desired.)
feature_names_chf = {'Hjorth Mobility', 'Hjorth Complexity', 'LF/HF Ratio', 'PSD Entropy'};
feature_names_arr = {'Hjorth Mobility', 'LF/HF Ratio', 'PSD Entropy', 'Negative Peak Count'};

% Truncate to the minimum number of segments across classes
min_rows = min([size(NSR_features, 1), size(CHF_features, 1), size(ARR_features, 1)]);
NSR_features = NSR_features(1:min_rows, :);
CHF_features = CHF_features(1:min_rows, :);
ARR_features = ARR_features(1:min_rows, :);

% Combine features for classification:
% NSR vs CHF: use first 4 features from NSR and CHF groups
features_chf = [NSR_features(:, 1:4); CHF_features(:, 1:4)];
labels_chf = [zeros(min_rows, 1); ones(min_rows, 1)]; % 0 = NSR, 1 = CHF

% NSR vs ARR: use first 4 features from NSR and ARR groups
features_arr = [NSR_features(:, 1:4); ARR_features(:, 1:4)];
labels_arr = [zeros(min_rows, 1); ones(min_rows, 1)]; % 0 = NSR, 1 = ARR

%% Section 2 (c): SVM Model with Hyperparameter Optimization
% Normalize features with z-score and force them to be real
X_chf = real(zscore(features_chf));
X_arr = real(zscore(features_arr));
y_chf = labels_chf;
y_arr = labels_arr;

% --- Hyperparameter Optimization for NSR vs CHF ---
% Use RBF kernel and optimize BoxConstraint and KernelScale
tempModel_chf = fitcsvm(X_chf, y_chf, 'KernelFunction', 'rbf', 'Standardize', true, ...
    'OptimizeHyperparameters', {'BoxConstraint','KernelScale'}, ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',false,'Verbose',0));
bestBoxConstraint_chf = tempModel_chf.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint;
bestKernelScale_chf = tempModel_chf.HyperparameterOptimizationResults.XAtMinObjective.KernelScale;

% --- Hyperparameter Optimization for NSR vs ARR ---
tempModel_arr = fitcsvm(X_arr, y_arr, 'KernelFunction', 'rbf', 'Standardize', true, ...
    'OptimizeHyperparameters', {'BoxConstraint','KernelScale'}, ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',false,'Verbose',0));
bestBoxConstraint_arr = tempModel_arr.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint;
bestKernelScale_arr = tempModel_arr.HyperparameterOptimizationResults.XAtMinObjective.KernelScale;

% Display optimized hyperparameters
fprintf('Optimized NSR vs CHF parameters: BoxConstraint = %.4f, KernelScale = %.4f\n', bestBoxConstraint_chf, bestKernelScale_chf);
fprintf('Optimized NSR vs ARR parameters: BoxConstraint = %.4f, KernelScale = %.4f\n', bestBoxConstraint_arr, bestKernelScale_arr);

% Define number of folds for cross-validation
k = 5;

% --- NSR vs CHF Cross-Validation ---
N_chf = size(X_chf, 1);
fold_size_chf = floor(N_chf / k);
Acc_chf = zeros(1, k);
TPR_chf = zeros(1, k);
FPR_chf = zeros(1, k);
Precision_chf = zeros(1, k);

for j = 1:k
    test_idx = ((j-1)*fold_size_chf + 1):(j*fold_size_chf);
    train_idx = setdiff(1:N_chf, test_idx);
    
    SVM_Model = fitcsvm(X_chf(train_idx, :), y_chf(train_idx), 'KernelFunction', 'rbf', 'Standardize', true, ...
        'BoxConstraint', bestBoxConstraint_chf, 'KernelScale', bestKernelScale_chf);
    predictions = predict(SVM_Model, X_chf(test_idx, :));
    
    tp = sum(predictions == 1 & y_chf(test_idx) == 1);
    tn = sum(predictions == 0 & y_chf(test_idx) == 0);
    fp = sum(predictions == 1 & y_chf(test_idx) == 0);
    fn = sum(predictions == 0 & y_chf(test_idx) == 1);
    
    Acc_chf(j) = (tp + tn) / (tp + tn + fp + fn);
    TPR_chf(j) = tp / max(tp + fn, eps);
    FPR_chf(j) = fp / max(fp + tn, eps);
    Precision_chf(j) = tp / max(tp + fp, eps);
end

avg_Acc_chf = mean(Acc_chf);
avg_TPR_chf = mean(TPR_chf);
avg_FPR_chf = mean(FPR_chf);
avg_Precision_chf = mean(Precision_chf);

% --- NSR vs ARR Cross-Validation ---
N_arr = size(X_arr, 1);
fold_size_arr = floor(N_arr / k);
Acc_arr = zeros(1, k);
TPR_arr = zeros(1, k);
FPR_arr = zeros(1, k);
Precision_arr = zeros(1, k);

for j = 1:k
    test_idx = ((j-1)*fold_size_arr + 1):(j*fold_size_arr);
    train_idx = setdiff(1:N_arr, test_idx);
    
    SVM_Model = fitcsvm(X_arr(train_idx, :), y_arr(train_idx), 'KernelFunction', 'rbf', 'Standardize', true, ...
        'BoxConstraint', bestBoxConstraint_arr, 'KernelScale', bestKernelScale_arr);
    predictions = predict(SVM_Model, X_arr(test_idx, :));
    
    tp = sum(predictions == 1 & y_arr(test_idx) == 1);
    tn = sum(predictions == 0 & y_arr(test_idx) == 0);
    fp = sum(predictions == 1 & y_arr(test_idx) == 0);
    fn = sum(predictions == 0 & y_arr(test_idx) == 1);
    
    Acc_arr(j) = (tp + tn) / (tp + tn + fp + fn);
    TPR_arr(j) = tp / max(tp + fn, eps);
    FPR_arr(j) = fp / max(fp + tn, eps);
    Precision_arr(j) = tp / max(tp + fp, eps);
end

avg_Acc_arr = mean(Acc_arr);
avg_TPR_arr = mean(TPR_arr);
avg_FPR_arr = mean(FPR_arr);
avg_Precision_arr = mean(Precision_arr);

% Display performance metrics in a table
comparison = table({'NSR vs CHF'; 'NSR vs ARR'}, ...
    [avg_Acc_chf*100; avg_Acc_arr*100], ...
    [avg_TPR_chf*100; avg_TPR_arr*100], ...
    [avg_FPR_chf*100; avg_FPR_arr*100], ...
    [avg_Precision_chf*100; avg_Precision_arr*100], ...
    'VariableNames', {'Comparison', 'Accuracy (%)', 'TPR (%)', 'FPR (%)', 'Precision (%)'});
disp(comparison);

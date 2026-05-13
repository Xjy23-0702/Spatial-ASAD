
clear; clc; close all;
%储存位置
save_dir = './figures_CORAL';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

n_subjects = 16;%被试数量
target_window=1;
color_plot = [0.12, 0.47, 0.71];  % 蓝色

results(1).filepath = '../Improvement/Domain_imp/CORAL/subjective_adaptive/1s/result_subject_adaptive.csv';
results(1).method   = 'CORAL';
results(1).window   = 1;

results(2).filepath = '../Improvement/Domain_imp/CORAL/subjective_adaptive/2s/result_subject_adaptive.csv';
results(2).method   = 'CORAL';
results(2).window   = 2;

results(3).filepath = '../Improvement/Domain_imp/CORAL/subjective_adaptive/5s/result_subject_adaptive.csv';
results(3).method   = 'CORAL';
results(3).window   = 5;

results(4).filepath = '../Improvement/Domain_imp/CORAL/subjective_adaptive/10s/result_subject_adaptive.csv';
results(4).method   = 'CORAL';
results(4).window   = 10;

fprintf('读取数据中...\n');
num_files = length(results);
window_vals = [results.window];
all_acc = cell(num_files, 1);  % 每个 cell 存 16x1

for idx = 1:num_files
    raw = readmatrix(results(idx).filepath);
    if size(raw, 1) ~= n_subjects
        error('文件 %s 的行数不是 %d', results(idx).filepath, n_subjects);
    end
    all_acc{idx} = raw(:, 1);  % 直接取第一列
end

% 计算每个窗口的均值和标准差
mean_acc = zeros(num_files, 1);
std_acc  = zeros(num_files, 1);
for idx = 1:num_files
    mean_acc(idx) = mean(all_acc{idx});
    std_acc(idx)  = std(all_acc{idx});
end

% 提取目标窗口的被试准确率（用于图二）
target_idx = find(window_vals == target_window);
if isempty(target_idx)
    error('找不到窗口为 %d 的文件，请检查 target_window', target_window);
end
sub_acc = all_acc{target_idx};  % 16x1


%图一
figure('Position', [100, 100, 700, 500]);
errorbar(window_vals, mean_acc, std_acc, '-o', ...
    'Color', color_plot, 'LineWidth', 2.5, ...
    'MarkerSize', 8, 'MarkerFaceColor', color_plot);
xlabel('Decision Window (s)', 'FontSize', 12);
ylabel('Accuracy (%)', 'FontSize', 12);
title('Fig1: Mean Accuracy vs Decision Window', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
xlim([min(window_vals)-0.5, max(window_vals)+0.5]);
ylim([0 100]);
set(gca, 'XTick', window_vals);

% 保存
fname1 = 'Fig1_Window_Accuracy';
saveas(gcf, fullfile(save_dir, [fname1, '.png']));
saveas(gcf, fullfile(save_dir, [fname1, '.fig']));
fprintf('Saved: %s\n', fname1);


%图二
figure('Position', [100, 100, 900, 500]);
bar(1:n_subjects, sub_acc, 'FaceColor', color_plot, ...
    'EdgeColor', 'k', 'LineWidth', 0.8);
xlabel('Subject ID', 'FontSize', 12);
ylabel('Accuracy (%)', 'FontSize', 12);
title(sprintf('Fig2: Per-Subject Accuracy (Window = %ds)', target_window), ...
    'FontSize', 14, 'FontWeight', 'bold');
xlim([0.3, n_subjects + 0.7]);
ylim([0 105]);
grid on;
set(gca, 'XTick', 1:n_subjects, 'FontSize', 10);

% 保存
fname2 = sprintf('Fig2_Window%ds', target_window);
saveas(gcf, fullfile(save_dir, [fname2, '.png']));
saveas(gcf, fullfile(save_dir, [fname2, '.fig']));
fprintf('Saved: %s\n', fname2);

fprintf('\n所有图片已保存至 %s\n', save_dir);
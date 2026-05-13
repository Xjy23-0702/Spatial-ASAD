%% ==================== 手动配置区 ====================
% 图类型：
%   1 = 单独折线图（4个窗口）
%   2 = 合并折线图（4个窗口）
%   3 = 单独柱状图（16个被试）
%   4 = 合并柱状图（16个被试）+ 平均线
%   5 = 合并折线图（16个被试）+ 70%虚线  ← 新增
fig_type = 2;   % <--- 修改这里

% 图形标题（同时作为保存的文件名）
title_str = 'time_accuracy_CSP';   % <--- 修改这里

% 输入数据：
%  类型1：4 个元素向量 → 窗口 1,2,5,10 的准确率
%  类型2：n 行 4 列矩阵，每行一个方法
%  类型3：16 个元素向量 → 16 个被试准确率
%  类型4：m 行 16 列矩阵，每行一个方法
%  类型5：m 行 16 列矩阵，每行一个方法
data = [72.2737068965517 73.0926724137931 73.6170977011494 73.5272988505747
        63.5955459770115 63.7715517241379 63.6045258620690 63.2902298850575
        70.5347218513488	71.5185189247131	73.289351940155	73.6076388359069
];  % <--- 修改这里

% 图例名称（仅类型2/4/5需要，长度应与 data 行数一致）
legend_names = ["CSP","RGC", "CNN-Baseline", "SCR-CNN","CE-CNN","STM-CNN","CNN-Mamba"];   % <--- 修改

% 保存目录
save_dir = './pictures';
% ================================================================


%% ==================== 自动画图部分 ====================
% 创建保存目录
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% 固定参数
window_ticks = [1, 2, 5, 10];
subject_ticks = 1:16;
ylim_val = [0, 100];

% 单独图颜色
color_red  = [0.85, 0.33, 0.10];   % 单独折线图红色
color_blue = [0.00, 0.45, 0.74];   % 单独柱状图蓝色

% 合并图固定颜色序列（红、绿、蓝、橙、棕、粉、黄）
merge_colors = [
    1.00 0.00 0.00;   % 红
    0.00 0.60 0.00;   % 绿
    0.00 0.00 1.00;   % 蓝
    1.00 0.65 0.00;   % 橙
    0.60 0.30 0.00;   % 棕
    1.00 0.40 0.70;   % 粉
    1.00 0.90 0.00    % 黄
];

% 数据预处理与校验
data = double(data);
switch fig_type
    case 1
        if numel(data) ~= 4
            error('类型1: data 必须包含 4 个元素');
        end
        data = data(:)';
        n_lines = 1;
    case 2
        if size(data,2) ~= 4
            error('类型2: data 必须是 n×4 矩阵（列数=4 对应窗口数）');
        end
        n_lines = size(data,1);
    case 3
        if numel(data) ~= 16
            error('类型3: data 必须包含 16 个元素');
        end
        data = data(:)';
        n_lines = 1;
    case {4,5}
        if size(data,2) ~= 16
            error('类型4/5: data 必须是 m×16 矩阵（列数=16 对应被试数）');
        end
        n_lines = size(data,1);
    otherwise
        error('fig_type 必须为 1,2,3,4,5');
end

% 图例名称自动补全
if ismember(fig_type, [2,4,5])
    if ~exist('legend_names','var') || length(legend_names) ~= n_lines
        legend_names = "Method-" + (1:n_lines);
    end
end

% 确定颜色
switch fig_type
    case 1
        colors = repmat(color_red, 1, 1);
    case 3
        colors = repmat(color_blue, 1, 1);
    case {2,4,5}
        n_colors = size(merge_colors, 1);
        idx = mod((1:n_lines)-1, n_colors) + 1;
        colors = merge_colors(idx, :);
end

% 绘图
figure('Color','white','Position',[100,100,800,500]);
hold on;

switch fig_type
    case 1   % 单独折线图（4个窗口）
        plot(window_ticks, data, '-o', 'Color', colors(1,:), ...
            'LineWidth',2.5, 'MarkerSize',8, 'MarkerFaceColor',colors(1,:));
        xlabel('Decision Window (s)');
        xlim([0.5, 10.5]);
        set(gca, 'XTick', window_ticks);

    case 2   % 合并折线图（4个窗口）
        for i = 1:n_lines
            plot(window_ticks, data(i,:), '-o', 'Color', colors(i,:), ...
                'LineWidth',2, 'MarkerSize',7, 'MarkerFaceColor',colors(i,:), ...
                'DisplayName', legend_names(i));
        end
        xlabel('Decision Window (s)');
        xlim([0.5, 10.5]);
        set(gca, 'XTick', window_ticks);
        legend('Location', 'southeast');

    case 3   % 单独柱状图（16个被试）
        bar(subject_ticks, data, 'FaceColor', colors(1,:), ...
            'EdgeColor','k', 'LineWidth',0.8);
        xlabel('Subject ID');
        xlim([0.5, 16.5]);
        set(gca, 'XTick', subject_ticks);

    case 4   % 合并柱状图（16个被试）+ 平均线
        bar_width = 0.7 / n_lines;
        for i = 1:n_lines
            x_pos = (1:16) - 0.35 + (i-0.5)*bar_width;
            bar(x_pos, data(i,:), bar_width, ...
                'FaceColor', colors(i,:), 'EdgeColor','k', 'LineWidth',0.5, ...
                'DisplayName', legend_names(i));
        end
        subj_mean = mean(data, 1);
        plot(subject_ticks, subj_mean, '-s', ...
            'Color','k', 'LineWidth',2.5, 'MarkerSize',10, ...
            'MarkerFaceColor','w', 'MarkerEdgeColor','k', ...
            'DisplayName', 'Subject Mean');
        xlabel('Subject ID');
        xlim([0.5, 16.5]);
        set(gca, 'XTick', subject_ticks);
        legend('Location', 'southeast');

    case 5   % 合并折线图（16个被试）+ 70%灰色虚线
        for i = 1:n_lines
            plot(subject_ticks, data(i,:), '-o', 'Color', colors(i,:), ...
                'LineWidth',1.8, 'MarkerSize',6, 'MarkerFaceColor',colors(i,:), ...
                'DisplayName', legend_names(i));
        end
        % 70% 灰色虚线
        yline(70, '--', 'Color', [0.5 0.5 0.5], 'LineWidth',1.5, ...
              'HandleVisibility','off');
        text(16.2, 70.5, '70%', 'FontSize',9, 'Color',[0.5 0.5 0.5]);
        xlabel('Subject ID');
        xlim([0.5, 16.5]);
        set(gca, 'XTick', subject_ticks);
        legend('Location', 'southeast');
end

hold off;
ylabel('Accuracy (%)');
ylim(ylim_val);
%title(title_str, 'FontSize',14, 'FontWeight','bold');
grid on;

% 保存图片
safe_name = regexprep(title_str, '[\\/:*?"<>|]', '_');
save_path_png = fullfile(save_dir, [safe_name, '.png']);
save_path_fig = fullfile(save_dir, [safe_name, '.fig']);
saveas(gcf, save_path_png);
saveas(gcf, save_path_fig);
fprintf('图片已保存至：%s\n', save_path_png);
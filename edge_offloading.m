% 清空环境
clear;
clc;

% 参数设置
I = 3e7;                 % 任务大小（bit）
P = 10;                  % 发射功率（W）
r = [2e7, 1e8, 1.5e8];   % 传输速率（bit/s）
t_av = 4;                % 平均任务到达率（task/slot）
f = 1e9;                 % 设备计算资源（cycles/s）
J = 1.3e9;               % 处理每个任务需要的 cycles 数
beta = 8e-27;            % 能量系数
theta = 30;              % 收益常数
q = 0:16; q_max = 16;    % 队列长度，最大为 16
Action = [0 0; 0 1; 0 2; 1 0; 1 1; 1 2; 2 0; 2 1; 2 2]; % 动作集合
e_greedy = 0.09;         % ε-贪心策略中的 ε
reward_decay = 0.9;      % 折扣因子 γ
slot_max = 30000;        % 最大时隙数

% 构建状态空间
num_q = length(q);
num_r = length(r);
State = zeros(num_q * num_r, 2);

for i = 1:num_q * num_r
    State(i, 1) = q(floor((i - 1) / num_r) + 1); % 队列长度
    State(i, 2) = r(mod((i - 1), num_r) + 1);    % 传输速率
end

% 初始化奖励函数 R
R = zeros(size(State, 1), size(Action, 1));

for i = 1:size(State, 1)
    Q_q = State(i, 1);
    r_r = State(i, 2);
    for j = 1:size(Action, 1)
        a_loc = Action(j, 1);
        a_off = Action(j, 2);
        total_action = a_loc + a_off;
        if total_action == 0
            R(i, j) = -Q_q / t_av;
        else
            energy = beta * f^2 * J * a_loc + P * I * a_off / r_r;
            delay = Q_q / t_av + (J * a_loc / f + I * a_off / r_r) / total_action;
            R(i, j) = theta * log(1 + total_action) - energy - delay;
        end
    end
end

% 初始化 Q 表
Q_table = zeros(size(State, 1), size(Action, 1));

% 初始化队列长度和状态
Q_q = 0; % 初始队列长度
r_r = 1.5e8; % 初始传输速率

% 查找初始状态
epsilon_state = 1e-6; % 浮点数比较容差
r_diffs_init = abs(r - r_r);
r_index_init = find(r_diffs_init < epsilon_state, 1);
if isempty(r_index_init)
    [~, r_index_init] = min(r_diffs_init);
end
state = num_r * Q_q + r_index_init;

% 检查初始 state
if state < 1 || state > size(Q_table, 1)
    error('Invalid initial state index: state=%d, Q_q=%d, r_r=%.2e', state, Q_q, r_r);
end

% 记录奖励
reward = 0;
reward_av = zeros(1, slot_max);
reward_ga_av = zeros(1, slot_max); % 遗传算法的平均奖励
reward_loc_av = zeros(1, slot_max);

% 记录队列长度和能耗
queue_length_qlearning = zeros(1, slot_max); % Q-Learning 队列长度
queue_length_loc = zeros(1, slot_max); % 本地计算策略队列长度
queue_length_ga = zeros(1, slot_max); % 遗传算法队列长度
energy_qlearning = zeros(1, slot_max); % Q-Learning 的能耗
energy_loc = zeros(1, slot_max); % 本地计算策略的能耗
energy_ga = zeros(1, slot_max); % 遗传算法的能耗

% 记录 Q 值更新过程（例如，最大 Q 值）
max_Q_over_time = zeros(1, slot_max);

% ==============================
% 遗传算法参数设置
% ==============================
population_size = 20;       % 种群大小
mutation_rate = 0.1;         % 变异率
num_generations = slot_max;  % 代数与时隙数相同
GA_population = randi([1, size(Action, 1)], population_size, 1); % 初始化种群（动作索引）
best_fitness_over_time = zeros(1, slot_max); % GA的最佳适应度

% 初始化GA的队列长度和能耗
Q_q_ga = 0;
r_r_ga = 1.5e8; % 初始传输速率
queue_length_ga(1) = Q_q_ga;
energy_ga(1) = 0;
reward_ga = 0;

% ==============================
% Q-Learning 策略模拟
% ==============================
for slot = 1:slot_max
    % -----------------
    % Q-Learning部分
    % -----------------
    % ε-贪心策略选择动作
    if rand() > e_greedy
        [max_Q_val, ~] = max(Q_table(state, :));
        action_candidates = find(Q_table(state, :) == max_Q_val);
        if isempty(action_candidates)
            error('No valid actions found at state %d', state);
        end
        action = action_candidates(randi(length(action_candidates)));
    else
        action = randi(size(Action, 1));
    end

    % 检查 action 是否为有效索引
    if action < 1 || action > size(Action, 1)
        error('Invalid action index: action=%d', action);
    end

    % 获取当前动作
    a_loc = Action(action, 1);
    a_off = Action(action, 2);
    total_action = a_loc + a_off;

    % 检查动作是否合法
    if total_action <= Q_q
        % 计算奖励
        if total_action == 0
            U = -Q_q / t_av;
        else
            energy = beta * f^2 * J * a_loc + P * I * a_off / r_r;
            delay = Q_q / t_av + (J * a_loc / f + I * a_off / r_r) / total_action;
            U = theta * log(1 + total_action) - energy - delay;
        end
    else
        action = 1; % 默认选择动作 1（即 [0,0]）
        U = -Q_q / t_av;
    end

    % 累计奖励
    reward = reward + U;
    reward_av(slot) = reward / slot;

    % 记录 Q 值的最大值
    max_Q_over_time(slot) = max(Q_table(:));

    % 更新能耗记录
    if total_action == 0
        energy_qlearning(slot) = 0;
    else
        energy_qlearning(slot) = beta * f^2 * J * a_loc + P * I * a_off / r_r;
    end

    % 更新队列长度记录
    queue_length_qlearning(slot) = Q_q;

    % 模拟任务到达
    c = rand();
    if c <= 0.5
        dao = 0;
    else
        dao = 8;
    end

    % 模拟传输速率变化
    c = rand();
    if c <= 0.25
        r_r = 2e7;
    elseif c <= 0.75
        r_r = 1e8;
    else
        r_r = 1.5e8;
    end

    % 更新队列长度
    Q_q = Q_q + dao - total_action;
    if Q_q > q_max
        Q_q = q_max;
    elseif Q_q < 0
        Q_q = 0;
    end
    Q_q = floor(Q_q); % 确保为整数

    % 获取 r_r 在 r 数组中的索引
    r_diffs = abs(r - r_r);
    r_index = find(r_diffs < epsilon_state, 1);

    % 如果未找到匹配的 r_index，使用最接近的值
    if isempty(r_index)
        [~, r_index] = min(r_diffs);
    end

    % 确保 r_index 为有效索引
    r_index = max(min(r_index, num_r), 1);
    r_index = floor(r_index);

    % 计算 next_state
    next_state = num_r * Q_q + r_index;

    % 检查 next_state 是否为有效索引
    if next_state < 1 || next_state > size(Q_table, 1)
        error('Invalid next_state index: next_state=%d, Q_q=%d, r_index=%d', next_state, Q_q, r_index);
    end

    % 更新 Q 值
    Q_table(state, action) = Q_table(state, action) + 0.9 * ...
        (R(state, action) + reward_decay * max(Q_table(next_state, :)) - Q_table(state, action));

    % 更新状态
    state = next_state;

    % -----------------
    % 遗传算法部分
    % -----------------
    % 1. 计算适应度（基于当前状态和动作的奖励）
    fitness = zeros(population_size, 1);
    for i = 1:population_size
        ga_action = GA_population(i);
        a_loc_ga = Action(ga_action, 1);
        a_off_ga = Action(ga_action, 2);
        total_action_ga = a_loc_ga + a_off_ga;
        if total_action_ga <= Q_q_ga
            if total_action_ga == 0
                U_ga = -Q_q_ga / t_av;
            else
                energy_ga_val = beta * f^2 * J * a_loc_ga + P * I * a_off_ga / r_r_ga;
                delay_ga = Q_q_ga / t_av + (J * a_loc_ga / f + I * a_off_ga / r_r_ga) / total_action_ga;
                U_ga = theta * log(1 + total_action_ga) - energy_ga_val - delay_ga;
            end
        else
            U_ga = -Q_q_ga / t_av;
        end
        fitness(i) = U_ga;
    end

    % 2. 选择（选择适应度最高的一半个体）
    [sorted_fitness, sorted_indices] = sort(fitness, 'descend');
    num_selected = floor(population_size / 2);
    selected = GA_population(sorted_indices(1:num_selected));

    % 3. 记录最佳适应度
    best_fitness_over_time(slot) = sorted_fitness(1);

    % 4. 交叉（随机配对后随机选择父母的基因）
    offspring = zeros(population_size - num_selected, 1);
    for i = 1:population_size - num_selected
        parent1 = selected(randi(num_selected));
        parent2 = selected(randi(num_selected));
        offspring(i) = Action(randi(num_r), 1) + Action(randi(num_r), 2);
        % 确保 offspring 是有效的动作索引
        if offspring(i) < 1 || offspring(i) > size(Action, 1)
            offspring(i) = randi(size(Action, 1));
        end
    end

    % 5. 变异（以一定概率随机改变个体的基因）
    for i = 1:length(offspring)
        if rand() < mutation_rate
            offspring(i) = randi(size(Action, 1));
        end
    end

    % 6. 创建新的种群
    GA_population = [selected; offspring];

    % 7. 选择适应度最高的个体作为当前动作
    best_action = GA_population(sorted_indices(1));
    a_loc_ga = Action(best_action, 1);
    a_off_ga = Action(best_action, 2);
    total_action_ga = a_loc_ga + a_off_ga;

    % 检查动作是否合法
    if total_action_ga <= Q_q_ga
        if total_action_ga == 0
            U_ga = -Q_q_ga / t_av;
        else
            energy_ga_val = beta * f^2 * J * a_loc_ga + P * I * a_off_ga / r_r_ga;
            delay_ga = Q_q_ga / t_av + (J * a_loc_ga / f + I * a_off_ga / r_r_ga) / total_action_ga;
            U_ga = theta * log(1 + total_action_ga) - energy_ga_val - delay_ga;
        end
    else
        best_action = 1; % 默认选择动作 1（即 [0,0]）
        U_ga = -Q_q_ga / t_av;
    end

    % 累计奖励
    reward_ga = reward_ga + U_ga;
    reward_ga_av(slot) = reward_ga / slot;

    % 记录GA的能耗
    if total_action_ga == 0
        energy_ga(slot) = 0;
    else
        energy_ga(slot) = beta * f^2 * J * a_loc_ga + P * I * a_off_ga / r_r_ga;
    end

    % 记录GA的队列长度
    queue_length_ga(slot) = Q_q_ga;

    % 模拟任务到达
    c = rand();
    if c <= 0.5
        dao_ga = 0;
    else
        dao_ga = 8;
    end

    % 模拟传输速率变化
    c = rand();
    if c <= 0.25
        r_r_ga = 2e7;
    elseif c <= 0.75
        r_r_ga = 1e8;
    else
        r_r_ga = 1.5e8;
    end

    % 更新队列长度
    Q_q_ga = Q_q_ga + dao_ga - total_action_ga;
    if Q_q_ga > q_max
        Q_q_ga = q_max;
    elseif Q_q_ga < 0
        Q_q_ga = 0;
    end
    Q_q_ga = floor(Q_q_ga); % 确保为整数

    % 计算 GA 的下一状态（虽然 GA 不使用状态，但为了记录，可以类似处理）
    % 这里假设 GA 使用相同的状态更新方式
    r_diffs_ga = abs(r - r_r_ga);
    r_index_ga = find(r_diffs_ga < epsilon_state, 1);
    if isempty(r_index_ga)
        [~, r_index_ga] = min(r_diffs_ga);
    end
    state_ga = num_r * Q_q_ga + r_index_ga;

    % 更新 GA 的队列长度和能耗（已在上方完成）
    
    % ==============================
    % 更新队列长度和状态
    % ==============================
    % 对于 Q-Learning 已在上方完成
end

% 本地计算策略
Q_q_loc = 0;
state_loc = find(q == Q_q_loc);
reward_loc = 0;

for slot = 1:slot_max
    % 动作选择：在本地计算策略中，我们只考虑本地计算
    action_loc_candidates = find(Action(:, 2) == 0);
    if isempty(action_loc_candidates)
        error('No local compute actions found');
    end
    action_loc = action_loc_candidates(1); % 选择本地计算的第一个动作

    a_loc = Action(action_loc, 1);
    total_action = a_loc;

    if total_action <= Q_q_loc
        if total_action == 0
            U_loc = -Q_q_loc / t_av;
        else
            energy = beta * f^2 * J * a_loc;
            delay = Q_q_loc / t_av + (J * a_loc / f) / total_action;
            U_loc = theta * log(1 + total_action) - energy - delay;
        end
    else
        action_loc = action_loc_candidates(1); % 默认选择 [0, 0]
        U_loc = -Q_q_loc / t_av;
    end

    % 累计奖励
    reward_loc = reward_loc + U_loc;
    reward_loc_av(slot) = reward_loc / slot;

    % 记录本地计算的能耗和队列长度
    if total_action == 0
        energy_loc(slot) = 0;
    else
        energy_loc(slot) = beta * f^2 * J * a_loc;
    end
    queue_length_loc(slot) = Q_q_loc;

    % 模拟任务到达
    c = rand();
    if c <= 0.5
        dao_loc = 0;
    else
        dao_loc = 8;
    end

    % 更新队列长度
    Q_q_loc = Q_q_loc + dao_loc - total_action;
    if Q_q_loc > q_max
        Q_q_loc = q_max;
    elseif Q_q_loc < 0
        Q_q_loc = 0;
    end
    Q_q_loc = floor(Q_q_loc); % 确保为整数

    % 更新状态
    state_loc = find(q == Q_q_loc);
end

% 滑动平均窗口大小（用于平滑曲线）
window_size = 100;

% 对能耗数据进行滑动平均
energy_qlearning_smooth = movmean(energy_qlearning, window_size);
energy_loc_smooth = movmean(energy_loc, window_size);
energy_ga_smooth = movmean(energy_ga, window_size);

% 对奖励数据进行滑动平均
reward_av_smooth = movmean(reward_av, window_size);
reward_loc_av_smooth = movmean(reward_loc_av, window_size);
reward_ga_av_smooth = movmean(reward_ga_av, window_size);

% 对队列长度数据进行滑动平均
queue_length_qlearning_smooth = movmean(queue_length_qlearning, window_size);
queue_length_loc_smooth = movmean(queue_length_loc, window_size);
queue_length_ga_smooth = movmean(queue_length_ga, window_size);

% ==============================
% 绘图部分
% ==============================

% 图1.1：最大 Q 值和遗传算法的最佳适应度
figure;
subplot(1,2,1);
plot(1:slot_max, max_Q_over_time, 'b-', 'LineWidth', 2);
xlabel('Time Slots');
ylabel('Maximum Q Value');
title('Maximum Q Value over Time Slots');
grid on;

% 图1.2：遗传算法的收敛过程（改进）
subplot(1,2,2);
% 只显示前1000个时隙的收敛过程
num_slots_to_display = 1000;
best_fitness_sub = best_fitness_over_time(1:num_slots_to_display);

% 绘制滑动平均曲线，窗口大小为50
smooth_fitness = movmean(best_fitness_sub, 50);

% 绘制原始适应度曲线
plot(1:num_slots_to_display, best_fitness_sub, 'g-', 'LineWidth', 1);
hold on;

% 绘制平滑后的适应度曲线
plot(1:num_slots_to_display, smooth_fitness, 'b-', 'LineWidth', 2);

% 设置坐标轴标签和标题
xlabel('Time Slots (First 1000 slots)');
ylabel('Best Fitness');
title('Genetic Algorithm Convergence (First 1000 Slots)');

% 添加网格和图例
grid on;
legend('Original Best Fitness', 'Smoothed Best Fitness');

% 图3：三种策略的平均奖励、能耗、队列长度比较
figure;
subplot(3, 1, 1);
plot(1:slot_max, reward_av_smooth, 'b-', 'LineWidth', 2); % Q-Learning 平均奖励（平滑）
hold on;
plot(1:slot_max, reward_loc_av_smooth, 'r--', 'LineWidth', 2); % 本地计算（平滑）
xlabel('Time Slots');
ylabel('Average Reward');
legend('Q-Learning', 'Local Computing (loc)');
title('Comparison of Average Reward');
grid on;

subplot(3, 1, 2);
plot(1:slot_max, queue_length_qlearning_smooth, 'b-', 'LineWidth', 2); % Q-Learning 队列长度（平滑）
hold on;
plot(1:slot_max, queue_length_loc_smooth, 'r--', 'LineWidth', 2); % 本地计算（平滑）
xlabel('Time Slots');
ylabel('Queue Length');
legend('Q-Learning', 'Local Computing (loc)');
title('Comparison of Queue Length (Smoothed)');
grid on;

subplot(3,1,3);
plot(1:slot_max, energy_qlearning_smooth, 'b-', 'LineWidth', 2);
hold on;
plot(1:slot_max, energy_loc_smooth, 'r--', 'LineWidth', 2);
xlabel('Time Slots');
ylabel('Energy Consumption (J)');
legend('Q-Learning', 'Local Computing (loc)');
title('Comparison of Energy Consumption');
grid on;

set(gcf, 'Position', [100, 100, 800, 1200]); % 调整图形窗口大小为竖图

% 图4：Q-Learning 和遗传算法的比较
figure;
subplot(3, 1, 1);
plot(1:slot_max, reward_av_smooth, 'b-', 'LineWidth', 2); % Q-Learning 平均奖励（平滑）
hold on;
plot(1:slot_max, reward_ga_av_smooth, 'm--', 'LineWidth', 2); % 遗传算法（平滑）
xlabel('Time Slots');
ylabel('Average Reward');
legend('Q-Learning', 'Genetic Algorithm (GA)');
title('Comparison of Average Reward: Q-Learning vs GA');
grid on;

subplot(3, 1, 2);
plot(1:slot_max, queue_length_qlearning_smooth, 'b-', 'LineWidth', 2); % Q-Learning 队列长度（平滑）
hold on;
plot(1:slot_max, queue_length_ga_smooth, 'm--', 'LineWidth', 2); % 遗传算法（平滑）
xlabel('Time Slots');
ylabel('Queue Length');
legend('Q-Learning', 'Genetic Algorithm (GA)');
title('Comparison of Queue Length: Q-Learning vs GA');
grid on;

subplot(3,1,3);
plot(1:slot_max, energy_qlearning_smooth, 'b-', 'LineWidth', 2);
hold on;
plot(1:slot_max, energy_ga_smooth, 'm--', 'LineWidth', 2);
xlabel('Time Slots');
ylabel('Energy Consumption (J)');
legend('Q-Learning', 'Genetic Algorithm (GA)');
title('Comparison of Energy Consumption: Q-Learning vs GA');
grid on;

% 打印对比数据
fprintf('\nComparison of Strategies:\n');
fprintf('-----------------------------------------------------------\n');
fprintf('Metric                     Q-Learning        Local (loc)      GA\n');
fprintf('-----------------------------------------------------------\n');
fprintf('Average Reward             %.2f              %.2f            %.2f\n', ...
    mean(reward_av), mean(reward_loc_av), mean(reward_ga_av));
fprintf('Cumulative Reward          %.2f              %.2f            %.2f\n', ...
    sum(reward_av), sum(reward_loc_av), sum(reward_ga_av));
fprintf('Average Energy Consumption %.2f J           %.2f J          %.2f J\n', ...
    mean(energy_qlearning), mean(energy_loc), mean(energy_ga));
fprintf('Average Queue Length       %.2f              %.2f            %.2f\n', ...
    mean(queue_length_qlearning), mean(queue_length_loc), mean(queue_length_ga));
fprintf('-----------------------------------------------------------\n');

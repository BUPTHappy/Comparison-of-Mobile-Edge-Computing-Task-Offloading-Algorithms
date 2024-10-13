function [rewards_q, energies_q, delays_q] = q_learning_algorithm(num_tasks, channel_states, actions, max_iter)
    % 初始化 Q 表
    Q = zeros(length(channel_states), length(actions));

    % 学习参数
    alpha = 0.1;   % 学习率
    gamma = 0.9;   % 折扣因子
    epsilon = 0.1; % ε-贪心策略

    % 存储每次迭代的平均奖励、能耗和时延
    rewards_q = zeros(max_iter, 1);
    energies_q = zeros(max_iter, 1);
    delays_q = zeros(max_iter, 1);

    for iter = 1:max_iter
        total_reward = 0;
        total_energy = 0;
        total_delay = 0;
        for task = 1:num_tasks
            % 随机生成信道状态
            state = channel_states(randi(length(channel_states)));

            % 选择动作（ε-贪心策略）
            if rand < epsilon
                action = actions(randi(length(actions)));
            else
                [~, idx] = max(Q(state, :));
                action = actions(idx);
            end

            % 计算奖励和下一个状态
            [reward, next_state, energy, delay] = compute_reward(state, action, channel_states);

            % 更新 Q 值
            best_next_action = max(Q(next_state, :));
            Q(state, action+1) = Q(state, action+1) + ...
                alpha * (reward + gamma * best_next_action - Q(state, action+1));

            total_reward = total_reward + reward;
            total_energy = total_energy + energy;
            total_delay = total_delay + delay;

            % 更新状态
            state = next_state;
        end
        % 记录平均奖励、能耗和时延
        rewards_q(iter) = total_reward / num_tasks;
        energies_q(iter) = total_energy / num_tasks;
        delays_q(iter) = total_delay / num_tasks;
    end
end

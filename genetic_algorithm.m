function [fitness_ga, energies_ga, delays_ga] = genetic_algorithm(num_tasks, channel_states, max_iter)
    % 遗传算法参数
    pop_size = 50;            % 种群规模
    chrom_length = num_tasks; % 染色体长度（任务数量）
    cross_rate = 0.8;         % 交叉概率
    mutate_rate = 0.1;        % 变异概率
    max_gen = max_iter;       % 最大迭代次数

    % 初始化种群（随机生成 0 和 1 的序列）
    population = randi([0,1], pop_size, chrom_length);

    % 存储每一代的平均适应度、能耗和时延
    fitness_ga = zeros(max_gen, 1);
    energies_ga = zeros(max_gen, 1);
    delays_ga = zeros(max_gen, 1);

    for gen = 1:max_gen
        % 计算适应度
        fitness = zeros(pop_size,1);
        total_energy = zeros(pop_size,1);
        total_delay = zeros(pop_size,1);
        for i = 1:pop_size
            [fitness(i), total_energy(i), total_delay(i)] = compute_fitness(population(i,:), channel_states);
        end
        fitness_ga(gen) = mean(fitness);
        energies_ga(gen) = mean(total_energy) / num_tasks;
        delays_ga(gen) = mean(total_delay) / num_tasks;

        % 选择操作（轮盘赌选择）
        idx = randsample(1:pop_size, pop_size, true, fitness - min(fitness));
        population = population(idx, :);

        % 交叉操作
        for i = 1:2:pop_size
            if rand < cross_rate
                cross_point = randi([1, chrom_length-1]);
                temp = population(i, cross_point+1:end);
                population(i, cross_point+1:end) = population(i+1, cross_point+1:end);
                population(i+1, cross_point+1:end) = temp;
            end
        end

        % 变异操作
        for i = 1:pop_size
            for j = 1:chrom_length
                if rand < mutate_rate
                    population(i,j) = 1 - population(i,j);
                end
            end
        end
    end
end

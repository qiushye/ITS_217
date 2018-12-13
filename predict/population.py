import random


class population:
    def __init__(self, RN, time_period, train_rate, size, cp, mp, gen_max):
        # 种群信息
        self.RN = RN
        roads = list(RN.roads.keys())
        links = []
        individual = []
        for r in roads:
            rs = RN.road_info[r]
            UE = list(rs.UE)
            links += UE
            for link in UE:
                w = rs.W[link]
                bin_str = bin(int(w * 1000 / 1024))[2:]
                individual += list(
                    map(int, (10 - len(bin_str)) * '0' + bin_str))
                # print(w)
        self.links = links
        self.time_period = time_period
        self.train_rate = train_rate
        self.individuals = []  # 个体集合
        self.fitness = []  # 个体适应度集合
        self.selector_probability = []  # 个体选择概率集合
        self.new_individuals = []  # 新一代个体集合

        self.elitist = {
            'chromosome': [0] * len(individual),
            'fitness': 0,
            'age': 0
        }  # 最佳个体的信息

        self.size = size  # 种群所包含的个体数
        self.chromosome_size = len(individual)  # 个体的染色体长度
        self.crossover_probability = cp  # 个体之间的交叉概率
        self.mutation_probability = mp  # 个体之间的变异概率

        self.generation_max = gen_max  # 种群进化的最大世代数
        self.age = 0  # 种群当前所处世代

        # 随机产生初始个体集，并将新一代个体、适应度、选择概率等集合以 0 值进行初始化
        v = 2**self.chromosome_size - 1
        for i in range(self.size):
            if i > 0:
                ind = random.randint(0, v)
                bin_str = bin(ind)[2:]
                individual = list(
                    map(int,
                        (self.chromosome_size - len(bin_str)) * '0' + bin_str))
            self.individuals.append(individual)
            self.new_individuals.append([0] * len(individual))
            self.fitness.append(0)
            self.selector_probability.append(0)

    def decode(self, chromosome):
        # n = float(2**self.chromosome_size - 1)
        for i in range(len(self.links)):
            link = self.links[i]
            road = link.split('-')[1]
            gene = ''.join([str(k) for k in chromosome[i * 10:(i + 1) * 10]])
            w = int(gene, 2) / 1024
            self.RN.road_info[road].W[link] = w
        return

    def fitness_func(self, chromosome):  # 将训练集的mre总和的倒数作为适应度
        time_period = self.time_period
        self.decode(chromosome)
        RN = self.RN
        unknown_roads = [r for r in RN.roads if RN.known[r] == False]
        indexes = RN.road_info[list(RN.roads.keys())[0]].V.index
        indice = 0
        mre_sum = 0
        while indice / len(indexes) < self.train_rate:
            indice += 1

            for r in unknown_roads:
                v_est = RN.online_est(r, indexes[indice], time_period,
                                      self.train_rate)

                v_ori = RN.road_info[r].V[time_period][indexes[indice]]
                mre_sum += abs(v_ori - v_est) / v_ori

        return (indice + 1) * len(unknown_roads) / mre_sum

    def evaluate(self):
        sp = self.selector_probability
        # road_info_pre = self.RN.road_info
        for i in range(self.size):
            self.fitness[i] = self.fitness_func(self.individuals[i])
            # print(self.individuals[i])
            # print(self.fitness[i])
            # print(self.RN.road_info['45'].W)
        ft_sum = sum(self.fitness)
        for i in range(self.size):
            sp[i] = self.fitness[i] / float(ft_sum)
        for i in range(1, self.size):
            sp[i] = sp[i] + sp[i - 1]

    def select(self):
        (t, i) = (random.random(), 0)
        for p in self.selector_probability:
            if p > t:
                break
            i = i + 1
        return i

    def cross(self, chrom1, chrom2):
        p = random.random()
        if chrom1 != chrom2 and p < self.crossover_probability:
            t = random.randint(1, self.chromosome_size - 1)

            (chrom1, chrom2) = (chrom1[:t] + chrom2[t:],
                                chrom2[:t] + chrom1[t:])
        return (chrom1, chrom2)

    def mutate(self, chrom):
        p = random.random()
        if p < self.mutation_probability:
            t = random.randint(0, self.chromosome_size - 1)
            try:
                chrom[t] = 1 - chrom[t]
            except:
                print(t, len(chrom), self.chromosome_size)
        return chrom

    def reproduct_elitist(self):
        # 与当前种群进行适应度比较，更新最佳个体
        j = 0
        for i in range(self.size):
            if self.elitist['fitness'] < self.fitness[i]:
                j = i
                self.elitist['fitness'] = self.fitness[i]
        if (j > 0):
            self.elitist['chromosome'] = self.individuals[j]
            self.elitist['age'] = self.age

        new_fitness = [self.fitness_func(v) for v in self.new_individuals]
        best_fitness = max(new_fitness)
        if self.elitist['fitness'] > best_fitness:
            # 寻找最小适应度对应个体
            j = 0
            for i in range(self.size):
                if best_fitness > new_fitness[i]:
                    j = i
                    best_fitness = new_fitness[i]
            self.new_individuals[j] = self.elitist['chromosome']

    def evolve(self):
        indvs = self.individuals
        new_indvs = self.new_individuals

        # 计算适应度及选择概率
        self.evaluate()

        # 进化操作
        i = 0
        while True:
            # 选择两名个体，进行交叉与变异，产生 2 名新个体
            idv1 = self.select()
            idv2 = self.select()

            # 交叉
            idv1 = indvs[idv1]
            idv2 = indvs[idv2]
            idv1, idv2 = self.cross(idv1, idv2)

            # 变异
            idv1 = self.mutate(idv1)
            idv2 = self.mutate(idv2)

            if random.randint(0, 1) == 0:
                new_indvs[i] = idv1
            else:
                new_indvs[i] = idv2

            # 判断进化过程是否结束
            i = i + 1
            if i >= self.size:
                break

        # 最佳个体保留
        self.reproduct_elitist()

        # 更新换代
        for i in range(self.size):
            self.individuals[i] = self.new_individuals[i]
        self.age += 1

    def run(self):
        for i in range(self.generation_max):
            self.evolve()
            # print(i, max(self.fitness),
            #       sum(self.fitness) / self.size, min(self.fitness))
            print(i, self.elitist['fitness'], self.elitist['age'])
